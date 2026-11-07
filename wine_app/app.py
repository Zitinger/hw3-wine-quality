import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


app_dir = Path(__file__).parent
st.set_page_config(page_title='качество вина', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(app_dir / 'model.joblib')


@st.cache_data
def load_data():
    test = pd.read_csv(app_dir / 'test.csv')
    config = json.loads((app_dir / 'config.json').read_text(encoding='utf-8'))
    return test, config


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def batch_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def check_csv(data, features):
    missing = [name for name in features if name not in data.columns]
    if missing:
        return None, 'не хватает столбцов: ' + ', '.join(missing)
    if data.empty:
        return None, 'в файле нет строк с данными'

    values = data[features].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return None, 'признаки должны быть числами без пропусков и бесконечности'
    if not values['is_red'].isin([0, 1]).all():
        return None, 'в столбце is_red допустимы только 0 и 1'
    return values, None


model = load_model()
test, config = load_data()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
probability_test = test_proba(model, X_test)

if 'threshold' not in st.session_state:
    st.session_state['threshold'] = float(config['threshold'])

st.title('качество вина')
st.write('модель помогает отбирать хорошие вина по результатам лабораторного анализа')
threshold = st.sidebar.slider(
    'порог для всех предсказаний', min_value=0.0, max_value=1.0,
    step=0.01, format='%.3f', key='threshold')
single_tab, threshold_tab, batch_tab, weights_tab = st.tabs([
    'одно вино', 'модельные метрики', 'пакетное предсказание', 'коэффициенты'
])


with threshold_tab:
    prediction_test = (probability_test >= threshold).astype(int)

    cols = st.columns(3)
    cols[0].metric('precision', f'{metrics.precision_score(y_test, prediction_test, zero_division=0):.3f}')
    cols[1].metric('recall', f'{metrics.recall_score(y_test, prediction_test, zero_division=0):.3f}')
    cols[2].metric('f1', f'{metrics.f1_score(y_test, prediction_test, zero_division=0):.3f}')

    matrix = metrics.confusion_matrix(y_test, prediction_test, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set(xlabel='предсказанный класс', ylabel='истинный класс',
           title='матрица ошибок')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # рабочая точка
    tn, fp, fn, tp = matrix.ravel()
    current_fpr = fp / (fp + tn) if fp + tn else 0
    current_tpr = tp / (tp + fn) if tp + fn else 0
    current_precision = tp / (tp + fp) if tp + fp else 0

    fpr, tpr, _ = metrics.roc_curve(y_test, probability_test)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(
        y_test, probability_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, label='модель')
    axes[0].plot([0, 1], [0, 1], '--', label='случайная модель')
    axes[0].scatter(current_fpr, current_tpr, color='red', label='текущий порог')
    axes[0].set(xlabel='fpr', ylabel='tpr', title='roc-кривая')
    axes[1].plot(recall_curve, precision_curve, label='модель')
    axes[1].axhline(y_test.mean(), linestyle='--', label='случайная модель')
    axes[1].scatter(current_tpr, current_precision, color='red', label='текущий порог')
    axes[1].set(xlabel='recall', ylabel='precision', title='pr-кривая')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write('при снижении порога модель отбирает больше вин: recall не убывает, '
             'но ложных рекомендаций становится больше. при повышении порога '
             'отбор становится строже, а recall не возрастает. precision часто '
             'растет, но на конечной выборке может меняться немонотонно.')
    st.caption('тестовая выборка здесь служит для демонстрации; исходный порог '
               'для модели выбран в ноутбуке по валидации')


with single_tab:
    st.write('введите характеристики вина')
    labels = {
        'fixed_acidity': 'фиксированная кислотность',
        'volatile_acidity': 'летучая кислотность',
        'citric_acid': 'лимонная кислота',
        'residual_sugar': 'остаточный сахар',
        'chlorides': 'хлориды',
        'free_sulfur_dioxide': 'свободный диоксид серы',
        'total_sulfur_dioxide': 'общий диоксид серы',
        'density': 'плотность',
        'pH': 'ph',
        'sulphates': 'сульфаты',
        'alcohol': 'алкоголь'
    }
    values = {}
    columns = st.columns(2)
    for i, name in enumerate(features):
        with columns[i % 2]:
            if name == 'is_red':
                color = st.radio('тип вина', ['белое', 'красное'])
                values[name] = int(color == 'красное')
            else:
                values[name] = st.number_input(
                    labels[name], min_value=float(X_test[name].min()),
                    max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()), format='%.5f')

    one_wine = pd.DataFrame([values], columns=features)
    st.caption(f'текущий порог: {threshold:.6f}')
    if st.button('получить прогноз'):
        probability = float(model.predict_proba(one_wine)[0, 1])
        prediction = int(probability >= threshold)
        st.metric('вероятность хорошего вина', f'{probability:.3f}')
        st.write('класс:', prediction)
        if prediction == 1:
            st.success('вино рекомендовано для подборки')
        else:
            st.info('вино не рекомендовано для подборки')


with batch_tab:
    st.write('загрузите csv с 12 признаками вина. столбец target не нужен.')
    st.caption(f'текущий порог: {threshold:.6f}')
    st.download_button('скачать пример входных данных',
                       X_test.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='wine_example.csv', mime='text/csv')
    uploaded_file = st.file_uploader('файл с признаками', type=['csv'])
    if uploaded_file is not None:
        try:
            batch = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        except (ValueError, UnicodeError, pd.errors.ParserError):
            st.error('не удалось прочитать csv: проверьте кодировку и разделитель')
        else:
            batch_X, error = check_csv(batch, features)
            if error is not None:
                st.error(error)
            else:
                batch_probability = batch_proba(model, batch_X)
                result = batch.copy()
                result['probability'] = batch_probability
                result['prediction'] = (batch_probability >= threshold).astype(int)
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    'скачать предсказания',
                    result.to_csv(index=False).encode('utf-8'),
                    file_name='wine_predictions.csv', mime='text/csv')


with weights_tab:
    coefficients = pd.Series(model[-1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(coefficients >= 0, 'steelblue', 'salmon')
    ax.barh(coefficients.index, coefficients.values, color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set(xlabel='коэффициент после стандартизации', ylabel='признак',
           title='коэффициенты логистической регрессии')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('при прочих равных положительный коэффициент повышает оценку '
             'хорошего вина, отрицательный - понижает. наибольшие положительные '
             'коэффициенты у остаточного сахара, фиксированной кислотности и '
             'алкоголя. наибольшие отрицательные - у плотности и летучей '
             'кислотности. признаки связаны между собой, поэтому эти '
             'коэффициенты не доказывают причинного влияния.')
