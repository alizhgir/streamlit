import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

st.title('Эй, кожаный, сыграем в регрессию?')
st.sidebar.header('Клади сюда файл .csv для обучающей выборки')
uploaded_train_file = st.sidebar.file_uploader("Выбери файл .csv для обучающей выборки", type=["csv"])

st.sidebar.header('Клади сюда файл .csv для тестовой выборки')
uploaded_test_file = st.sidebar.file_uploader("Выбери файл .csv для тестовой выборки", type=["csv"])

data = None
test_data = None

if uploaded_train_file is not None:
    data = pd.read_csv(uploaded_train_file, index_col=0)
    st.write('Пример данных обучающей выборки:')
    st.write(data.head(4))
else:
    st.write('Файл обучающей выборки не загружен, я не могу так работать')

if uploaded_test_file is not None:
    test_data = pd.read_csv(uploaded_test_file, index_col=0)
    st.write('Пример данных тестовой выборки:')
    st.write(test_data.head(4))

class LogReg:
    def __init__(self, learning_rate, n_features):
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.coef_ = np.random.uniform(-1, 1, n_features)
        self.intercept_ = np.random.uniform(-1, 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def bce_loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def fit(self, X, y, n_epochs=1000):
        for epoch in range(n_epochs):
            linear_model = X @ self.coef_ + self.intercept_  # Вычисляем линейное предсказание
            y_pred = self.sigmoid(linear_model)

            error = y_pred - y  # Вычисляем градиенты для BCE
            grad_w = (X.T @ error) / X.shape[0]
            grad_intercept = np.mean(error)

            self.coef_ -= self.learning_rate * grad_w # Обновляем веса и свободный член
            self.intercept_ -= self.learning_rate * grad_intercept

            if (epoch + 1) % 250 == 0: # Печатаем функцию потерь (опционально)
                loss = self.bce_loss(y, y_pred)
                st.markdown(
                    f'<p style="font-size: 10px;">Epoch {epoch+1}/{n_epochs} - Loss: {loss}</p>', 
                    unsafe_allow_html=True
                )
                
                
    def predict(self, X):
        # Вычисляем линейное предсказание и применяем сигмоиду
        linear_model = X @ self.coef_ + self.intercept_
        probas = self.sigmoid(linear_model)
        # Возвращаем предсказанные классы
        return np.where(probas >= 0.5, 1, 0)

def plot_graph(data, x, y, graph_type):
    if graph_type == 'scatter':
        plt.scatter(data[x], data[y], color='red', alpha=0.1)
    elif graph_type == 'bar':
        plt.bar(data[x], data[y], color='red', alpha=0.1)
    elif graph_type == 'plot':
        plt.plot(data[x], data[y], color='red', alpha=0.1)
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot(plt)


if data is not None:
    
    graph_type = st.selectbox('Выбери тип графика для обучающей выборки', ['scatter', 'bar', 'plot'])
    available_columns = data.columns.tolist()
    x_axis = st.selectbox('Выбери ось X', available_columns, index=0)
    y_axis = st.selectbox('Выбери ось Y', available_columns, index=min(1, len(available_columns)-1))
    plot_graph(data, x_axis, y_axis, graph_type)
    
    # Выбор столбцов для обучения
    columns = data.columns.tolist()
    selected_features = st.multiselect("Выбери признаки для обучения", columns, default=columns[:-1])
    target_column = st.selectbox("Выбери целевой столбец", columns, index=len(columns)-1)

    # Проверка на бинарный таргет
    if data[target_column].nunique() == 2:
        # Нормирование данных
        scaler = StandardScaler()
        X = scaler.fit_transform(data[selected_features])
        y = data[target_column].values
        
        # Обучение модели
        model = LogReg(learning_rate=0.01, n_features=len(selected_features))
        model.fit(X, y)
        
        # Вывод результатов
        weights = model.coef_
        intercept = model.intercept_
        st.subheader('Веса в уравнении:')
        results = {feature: weight for feature, weight in zip(selected_features, weights)}
        st.write(results, f'Коэффициент W0: {intercept}')
        X_test = X
        y_pred_test = model.predict(X_test)
        # Истинные значения y из датафрейма train
        y_true_test = y
        precision = precision_score(y_true_test, y_pred_test)
        recall = recall_score(y_true_test, y_pred_test)
        f1 = f1_score(y_true_test, y_pred_test)
        # accuracy = accuracy_score(y_true_test, y_pred_test)
        st.subheader('Результаты запуска модели:')
        st.write(f'Precision: {precision:.2%}')
        st.write(f'Recall: {recall:.2%}')
        st.write(f'F1 Score: {f1:.2%}')
        # st.write(f'Accuracy: {accuracy:.2%}')

    else:
        st.error("Целевой столбец должен быть бинарным (0 или 1).")


if test_data is not None:
    # Выбор столбцов для тестирования
    columns = test_data.columns.tolist()
    st.header('Проделаем упражнение для тестовой выборки')
    selected_features_test = st.multiselect("Выберите признаки для тестирования", columns, default=columns[:-1], key='test_features')
    target_column_test = st.selectbox("Выберите целевой столбец для тестирования", columns, index=len(columns)-1, key='test_target')

    # Проверка на бинарный таргет
    if test_data[target_column_test].nunique() == 2:
        # Нормирование данных
        scaler_test = StandardScaler()
        X_test = scaler_test.fit_transform(test_data[selected_features_test])
        y_test = test_data[target_column_test].values
        
        # Предсказания на тестовых данных
        y_pred_test = model.predict(X_test)
        
        # Истинные значения y из датафрейма test
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        st.subheader('Результаты запуска модели:')
        st.write(f'Precision: {precision_test:.2%}')
        st.write(f'Recall: {recall_test:.2%}')
        st.write(f'F1 Score: {f1_test:.2%}')
        st.write(f'Accuracy: {accuracy_test:.2%}')
    else:
        st.error("Целевой столбец должен быть бинарным (0 или 1).")
