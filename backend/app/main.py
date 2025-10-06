import torch
assert torch.cuda.is_available(), "CUDA не обнаружена. Установите PyTorch с поддержкой CUDA (см. README_GPU.md)."
DEVICE = "cuda"

"""Основное FastAPI-приложение, которое собирает все части сервиса."""

from fastapi import FastAPI, File, HTTPException, UploadFile

from .models.image_classifier import predict_topk
from .models.text_classifier import TextClassifier
from .schemas import ImagePrediction, ImageResponse, TextRequest, TextResponse
from .utils.embeddings import load_e5_model
from .utils.images import load_image_from_bytes

# Инициализируем FastAPI-приложение с понятным описанием
app = FastAPI(
    title="FastAPI + Streamlit демо",
    description="Учебный пример классификации текстов и изображений на GPU",
    version="1.0.0",
)

# Загружаем модель эмбеддингов один раз при старте, чтобы не тратить время на каждый запрос
E5_MODEL, E5_TOKENIZER = load_e5_model(device=DEVICE)
# Создаём текстовый классификатор, который при первом запуске обучит CatBoost и сохранит артефакты
TEXT_CLASSIFIER = TextClassifier(E5_MODEL, E5_TOKENIZER, device=DEVICE)


@app.get("/health")
def healthcheck() -> dict:
    """Быстрая проверка того, что сервис жив и видит GPU."""
    return {"status": "ok", "device": DEVICE}


@app.post("/predict/text", response_model=TextResponse)
async def predict_text(payload: TextRequest) -> TextResponse:
    """Получает текст и возвращает предсказанную категорию с вероятностями."""
    try:
        label, probas = TEXT_CLASSIFIER.predict(payload.text)
    except ValueError as exc:  # Например, если текст пустой
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TextResponse(label=label, probas=probas)


@app.post("/predict/image", response_model=ImageResponse)
async def predict_image(file: UploadFile = File(...)) -> ImageResponse:
    """Принимает изображение, прогоняет через ResNet18 и возвращает топ-3 класса."""
    raw_bytes = await file.read()
    try:
        image = load_image_from_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    predictions = predict_topk(image, k=3)
    if not predictions:
        raise HTTPException(status_code=500, detail="Не удалось получить предсказания от модели.")
    top1 = predictions[0]
    top3 = [ImagePrediction(**item) for item in predictions[:3]]
    return ImageResponse(top1=ImagePrediction(**top1), top3=top3)
