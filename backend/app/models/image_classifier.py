import torch
assert torch.cuda.is_available(), "CUDA не обнаружена. Установите PyTorch с поддержкой CUDA (см. README_GPU.md)."
DEVICE = "cuda"

"""GPU-классификатор изображений на основе ResNet18."""

from typing import List

import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# Загружаем предобученные веса и сразу переносим модель на GPU
_WEIGHTS = ResNet18_Weights.DEFAULT
_MODEL = resnet18(weights=_WEIGHTS)
_MODEL.to(DEVICE)
_MODEL.eval()
_PREPROCESS = _WEIGHTS.transforms()  # Готовая функция подготовки изображения
_LABELS = _WEIGHTS.meta["categories"]


def predict_topk(image: Image.Image, k: int = 3) -> List[dict]:
    """Возвращает топ-k классов с вероятностями в удобном виде."""
    tensor = _PREPROCESS(image).unsqueeze(0).to(DEVICE)  # Готовим батч из одного изображения
    with torch.no_grad():
        logits = _MODEL(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_indices = probs.argsort()[::-1][: max(k, 1)]
    results: List[dict] = []
    for idx in top_indices:
        results.append({
            "label": _LABELS[idx],
            "score": float(round(probs[idx], 4)),
        })
    return results
