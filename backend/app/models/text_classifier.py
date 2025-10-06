import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("⚠️ CUDA не обнаружена, используется CPU.")

"""GPU-классификатор текстов с обучением CatBoost при первом запуске."""

from ..utils.embeddings import get_embeddings

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "text_catboost.cbm"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"
TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "train_channels.csv"
VAL_PATH = Path(__file__).resolve().parent.parent / "data" / "val_channels.csv"


class TextClassifier:
    """Инкапсулирует обучение и инференс CatBoost-модели."""

    def __init__(self, embedding_model, tokenizer, device: str = DEVICE) -> None:
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.device = device
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            self._train_and_save()
        self.catboost, self.id2label = self._load_artifacts()

    def _train_and_save(self) -> None:
        """Обучаем CatBoost на GPU и сохраняем результат на диск."""
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)

        train_texts = train_df["content"].fillna("").astype(str).tolist()
        val_texts = val_df["content"].fillna("").astype(str).tolist()

        train_labels = train_df["label"].astype(str).tolist()
        val_labels = val_df["label"].astype(str).tolist()

        unique_labels = sorted(set(train_labels) | set(val_labels))  # Фиксируем порядок классов
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        train_targets = np.array([label2id[label] for label in train_labels], dtype=np.int32)
        val_targets = np.array([label2id[label] for label in val_labels], dtype=np.int32)

        train_vectors = get_embeddings(train_texts, self.embedding_model, self.tokenizer, device=self.device)
        val_vectors = get_embeddings(val_texts, self.embedding_model, self.tokenizer, device=self.device)

        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            task_type="GPU",
            devices="0",
            verbose=False,
        )
        model.fit(train_vectors, train_targets, eval_set=(val_vectors, val_targets), verbose=False)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_PATH))  # Сохраняем веса модели на диск
        with LABELS_PATH.open("w", encoding="utf-8") as file:
            json.dump(unique_labels, file, ensure_ascii=False, indent=2)

    def _load_artifacts(self) -> Tuple[CatBoostClassifier, Dict[int, str]]:
        """Загружает сохранённую модель и словарь меток."""
        with LABELS_PATH.open("r", encoding="utf-8") as file:
            labels: List[str] = json.load(file)
        id2label = {idx: label for idx, label in enumerate(labels)}
        model = CatBoostClassifier(task_type="GPU", devices="0")  # Говорим CatBoost использовать GPU и для инференса
        model.load_model(str(MODEL_PATH))
        return model, id2label

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Возвращает метку и вероятности классов для заданного текста."""
        if not text or not text.strip():
            raise ValueError("Пустой текст. Введите содержательное сообщение.")

        vectors = get_embeddings([text], self.embedding_model, self.tokenizer, device=self.device)
        probas = self.catboost.predict_proba(vectors)[0]
        top_idx = int(np.argmax(probas))
        label = self.id2label[top_idx]
        formatted = {self.id2label[idx]: float(round(score, 4)) for idx, score in enumerate(probas)}
        return label, formatted
