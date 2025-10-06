import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("⚠️ CUDA не обнаружена, используется CPU.")

"""Простые утилиты для работы с текстовыми эмбеддингами."""

from typing import List, Tuple

import numpy as np
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "d0rj/e5-base-en-ru"


def load_e5_model(device: str = DEVICE) -> Tuple[AutoModel, AutoTokenizer]:
    """Загружает модель E5 и токенизатор на указанный GPU."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)  # Переносим параметры модели на видеокарту
    model.eval()  # Переводим в режим инференса, чтобы отключить dropout
    return model, tokenizer


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Среднее по токенам с учётом маски, чтобы получить один вектор на текст."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)  # Складываем только значащие токены
    counts = mask.sum(dim=1).clamp(min=1e-9)  # Избегаем деления на ноль
    return summed / counts


def get_embeddings(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = DEVICE,
    batch_size: int = 8,
) -> np.ndarray:
    """Преобразует тексты в числовые векторы, описывающие смысл сообщения."""
    vectors: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():  # Отключаем градиенты, потому что делаем только инференс
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}  # Переносим входные тензоры на GPU
            outputs = model(**encoded)
            pooled = _mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])  # Получаем эмбеддинг текста
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))  # Возвращаем данные на CPU для CatBoost
    return np.vstack(vectors)
