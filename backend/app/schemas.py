import torch
assert torch.cuda.is_available(), "CUDA не обнаружена. Установите PyTorch с поддержкой CUDA (см. README_GPU.md)."
DEVICE = "cuda"

"""Pydantic-схемы для сериализации запросов и ответов."""

from typing import Dict, List

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """JSON-запрос для текста."""

    text: str = Field(..., description="Текст поста из Telegram")


class TextResponse(BaseModel):
    """Ответ с итоговой меткой и вероятностями."""

    label: str
    probas: Dict[str, float]


class ImagePrediction(BaseModel):
    """Одна строка с результатом классификации изображения."""

    label: str
    score: float


class ImageResponse(BaseModel):
    """Ответ, который возвращает топ-1 и топ-3 классы."""

    top1: ImagePrediction
    top3: List[ImagePrediction]
