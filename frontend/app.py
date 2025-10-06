import io
from typing import Dict


DEVICE = "cuda"

"""Минимальный Streamlit-интерфейс для общения с FastAPI-бэкендом."""

import requests
import streamlit as st
from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000"


def call_text_endpoint(text: str) -> Dict[str, object]:
    """Отправляет текст на бэкенд и возвращает предсказание."""
    response = requests.post(f"{BACKEND_URL}/predict/text", json={"text": text}, timeout=60)
    response.raise_for_status()
    return response.json()


def call_image_endpoint(file_name: str, file_bytes: bytes) -> Dict[str, object]:
    """Отправляет изображение в формате multipart."""
    files = {"file": (file_name, file_bytes, "application/octet-stream")}
    response = requests.post(f"{BACKEND_URL}/predict/image", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="FastAPI + Streamlit демо", layout="centered")
st.title("Классификация текста и изображений")


tab_text, tab_image = st.tabs(["Текст", "Картинка"])

with tab_text:
    st.subheader("Классификация текстов из Telegram")
    st.write("Введите несколько предложений — модель E5 превратит их в эмбеддинг, а CatBoost определит тематику.")
    user_text = st.text_area("Текст сообщения", height=180, placeholder="Например: Новая выставка современного искусства откроется завтра")
    if st.button("Классифицировать", key="text_button"):
        if not user_text.strip():
            st.warning("Пожалуйста, введите текст перед отправкой.")
        else:
            with st.spinner("Обрабатывается текст..."):
                try:
                    result = call_text_endpoint(user_text)
                except requests.RequestException as exc:
                    st.error(f"Не удалось связаться с бэкендом: {exc}")
                else:
                    st.success(f"Предсказанный класс: {result['label']}")
                    table_data = [
                        {"Класс": label, "Вероятность": prob}
                        for label, prob in result["probas"].items()
                    ]
                    st.table(table_data)

with tab_image:
    st.subheader("Классификация изображений (ResNet18)")
    st.write("Загрузите картинку — ResNet18 выдаст топ-3 класса.")
    uploaded_file = st.file_uploader("Файл изображения", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=False)
    if st.button("Распознать", key="image_button"):
        if uploaded_file is None:
            st.warning("Сначала выберите файл.")
        else:
            file_bytes = uploaded_file.getvalue()
            try:
                preview_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Streamlit не смог прочитать файл как изображение: {exc}")
            else:
                st.image(preview_image, caption="Предпросмотр загруженного изображения", use_container_width=True)
                with st.spinner("Анализ изображения..."):
                    try:
                        result = call_image_endpoint(uploaded_file.name or "image.jpg", file_bytes)
                    except requests.RequestException as exc:
                        st.error(f"Не удалось получить ответ от бэкенда: {exc}")
                    else:
                        st.success(f"Топ-1 класс: {result['top1']['label']} ({result['top1']['score']})")
                        st.table([
                            {"Класс": item["label"], "Вероятность": item["score"]}
                            for item in result["top3"]
                        ])
