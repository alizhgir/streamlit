import streamlit as st
from skimage import io
import numpy as np

# Запросить URL изображения у пользователя
url = st.text_input("Пожалуйста введите адрес изображения:", 'https://www.purina.ru/sites/default/files/2021-10/abisinskaya-1.jpg')

if url:
    image = io.imread(url, as_gray=True)
    
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(image.shape)
    np.fill_diagonal(sigma, sing_values)

    top_k = st.slider("Выберите значение для сжатия:", 1, min(image.shape), 10)
    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    
    compressed_image = trunc_U @ trunc_sigma @ trunc_V

    # Нормализация изображения
    compressed_image = (compressed_image - compressed_image.min()) / (compressed_image.max() - compressed_image.min())

    # Использование функции columns в Streamlit для размещения двух изображений рядом
    col1, col2 = st.columns(2)

    compression_ratio = (top_k / len(sing_values)) * 100
    st.write(f"Вес изображения сократился до: {compression_ratio:.2f}%")

    with col1:
        st.image(image, caption="Оргинальное изображение", use_column_width=True, channels="GRAY")
    with col2:
        st.image(compressed_image, caption=f"Сжатое изображение", use_column_width=True, channels="GRAY")

