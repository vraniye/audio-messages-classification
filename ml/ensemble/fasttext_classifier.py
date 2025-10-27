#!/usr/bin/env python3
"""Инференс fastText-бейзлайна без чтения внешних файлов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fasttext  # type: ignore
import numpy as np

LABEL_PREFIX = "__label__"


def prepare_text(text: str) -> str:
    """Сжать последовательности пробелов, чтобы соответствовать обучению."""
    return " ".join(text.split())


def strip_label_prefix(label: str) -> str:
    """Удалить fastText-префикс и вернуть числовой идентификатор метки."""
    if label.startswith(LABEL_PREFIX):
        return label[len(LABEL_PREFIX) :]
    return label


class FastTextClassifier:
    """Обёртка над fastText-моделью с преобразованием меток."""

    def __init__(
        self,
        *,
        model_path: Path,
        label_index_path: Path,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Не найден fastText-модельный файл: {model_path}")
        if not label_index_path.exists():
            raise FileNotFoundError(f"Не найден label_index.json: {label_index_path}")

        self.model = fasttext.load_model(model_path.as_posix())

        with label_index_path.open("r", encoding="utf-8") as fp:
            mapping = json.load(fp)

        label_to_index = {str(k): int(v) for k, v in mapping["label_to_index"].items()}
        self.index_to_label = {int(v): str(k) for k, v in label_to_index.items()}
        self.label_ids = sorted(self.index_to_label.keys())
        self.id_to_position = {label_id: idx for idx, label_id in enumerate(self.label_ids)}
        self.label_sequence = [self.index_to_label[label_id] for label_id in self.label_ids]

    def predict(
        self,
        texts: Sequence[str],
        *,
        return_proba: bool = False,
    ) -> Tuple[List[str], np.ndarray] | List[str]:
        if not texts:
            raise ValueError("Список текстов пуст — нечего классифицировать.")

        prepared = [prepare_text(text) for text in texts]
        k = len(self.label_ids)
        raw_labels, raw_probs = self.model.predict(prepared, k=k)

        predictions: List[str] = []
        probabilities: List[np.ndarray] = []

        for label_list, prob_list in zip(raw_labels, raw_probs):
            if not label_list:
                raise RuntimeError("fastText не вернул ни одной метки.")

            proba_vector = np.zeros(k, dtype=float)
            for label, prob in zip(label_list, prob_list):
                label_id = int(strip_label_prefix(label))
                position = self.id_to_position[label_id]
                proba_vector[position] = prob

            top_label_id = int(strip_label_prefix(label_list[0]))
            predictions.append(self.index_to_label[top_label_id])
            probabilities.append(proba_vector)

        if return_proba:
            return predictions, np.vstack(probabilities)
        return predictions


def format_probabilities(prob_vector: np.ndarray, labels: Sequence[str]) -> str:
    pairs = [f"{label}: {prob:.3f}" for label, prob in zip(labels, prob_vector)]
    return " | ".join(pairs)


def main() -> None:
    root = Path(__file__).resolve().parent
    classifier = FastTextClassifier(
        model_path=root / "fasttext_ensemble.bin",
        label_index_path=root / "label_index.json",
    )

    input_text = "Напишите сюда текст для классификации официального или разговорного стиля."
    texts = [input_text]

    labels, proba = classifier.predict(texts, return_proba=True)

    for text, label, prob_vector in zip(texts, labels, proba):
        print("===")
        print(text)
        print(f"Класс: {label}")
        print(f"Вероятности: {format_probabilities(prob_vector, classifier.label_sequence)}")


if __name__ == "__main__":
    main()
