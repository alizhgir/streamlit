import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


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

            if (epoch + 1) % 100 == 0: # Печатаем функцию потерь (опционально)
                loss = self.bce_loss(y, y_pred)
                print(f'Epoch {epoch+1}/{n_epochs} - Loss: {loss}')
                
    def predict(self, X):
        # Вычисляем линейное предсказание и применяем сигмоиду
        linear_model = X @ self.coef_ + self.intercept_
        probas = self.sigmoid(linear_model)
        # Возвращаем предсказанные классы
        return np.where(probas >= 0.5, 1, 0)

# Функция загрузки данных
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")
    return None

st.title("Интерактивное приложение для логистической регрессии")

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")
data = load_data(uploaded_file)

def plot_graph(data, x, y, graph_type):
    if graph_type == 'scatter':
        plt.scatter(data[x], data[y])
    elif graph_type == 'bar':
        plt.bar(data[x], data[y])
    elif graph_type == 'plot':
        plt.plot(data[x], data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot(plt)
    st.write(data)


if data is not None:
    
    graph_type = st.selectbox('Выберите тип графика', ['scatter', 'bar', 'plot'])
    available_columns = data.columns.tolist()
    x_axis = st.selectbox('Выберите ось X', available_columns, index=0)
    y_axis = st.selectbox('Выберите ось Y', available_columns, index=min(1, len(available_columns)-1))
    plot_graph(data, x_axis, y_axis, graph_type)
    
    # Выбор столбцов для обучения
    columns = data.columns.tolist()
    selected_features = st.multiselect("Выберите признаки для обучения", columns, default=columns[:-1])
    target_column = st.selectbox("Выберите целевой столбец", columns, index=len(columns)-1)

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
        results = {feature: weight for feature, weight in zip(selected_features, weights)}
        st.write(results, f'Коэффициент свободного члена: {intercept}')
        X_test = X
        y_pred_test = model.predict(X_test)
        # Истинные значения y из датафрейма test
        y_true_test = y
        precision = precision_score(y_true_test, y_pred_test)
        recall = recall_score(y_true_test, y_pred_test)
        f1 = f1_score(y_true_test, y_pred_test)
        # accuracy = accuracy_score(y_true_test, y_pred_test)
        st.write(f'Precision: {precision:.2%}')
        st.write(f'Recall: {recall:.2%}')
        st.write(f'F1 Score: {f1:.2%}')
        # st.write(f'Accuracy: {accuracy:.2%}')

    else:
        st.error("Целевой столбец должен быть бинарным (0 или 1).")
        
test_file = st.file_uploader("Загрузите ваш проверочный CSV файл", type="csv")
test_data = load_data(test_file)

if test_data is not None:
    # Выбор столбцов для тестирования
    columns = test_data.columns.tolist()
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
        
        st.write(f'Precision: {precision_test:.2%}')
        st.write(f'Recall: {recall_test:.2%}')
        st.write(f'F1 Score: {f1_test:.2%}')
        st.write(f'Accuracy: {accuracy_test:.2%}')
    else:
        st.error("Целевой столбец должен быть бинарным (0 или 1).")
