import torch
assert torch.cuda.is_available(), "CUDA не обнаружена. Установите PyTorch с поддержкой CUDA (см. README_GPU.md)."
DEVICE = "cuda"

"""Утилиты для безопасной загрузки изображений."""

from io import BytesIO
from typing import Tuple

from PIL import Image


def load_image_from_bytes(raw_bytes: bytes) -> Image.Image:
    """Принимает сырые байты, проверяет, что это изображение, и переводит в RGB."""
    if not raw_bytes:
        raise ValueError("Файл пуст. Загрузите изображение повторно.")
    try:
        image = Image.open(BytesIO(raw_bytes))
        image = image.convert("RGB")  # Приводим к RGB, чтобы ResNet18 работал корректно
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("Не удалось открыть файл как изображение. Проверьте формат.") from exc
    return image
