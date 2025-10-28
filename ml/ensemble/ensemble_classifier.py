#!/usr/bin/env python3
"""Пример инференса ансамбля логистической регрессии и TextCNN."""

from __future__ import annotations

import inspect
import json
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from pymorphy2 import MorphAnalyzer

if not hasattr(inspect, "getargspec"):
    from collections import namedtuple

    _ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):  # type: ignore[override]
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec  # type: ignore[assignment]
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.nn import functional as F

TEXT_COLUMN = "text"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE | re.VERBOSE)
MENTION_PATTERN = re.compile(r"@[\w_]+")
HASHTAG_PATTERN = re.compile(r"#[\w_]+")
EMOJI_PATTERN = re.compile(r"[🌀-🗿😀-🙏🚀-🛿🜀-🝿]")
PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation if ch not in "#@"})
STOP_CHARS = re.compile(r"[^а-яёa-z0-9#@\s]")

morph = MorphAnalyzer()


@lru_cache(maxsize=200000)
def lemmatize_token(token: str) -> str:
    """Вернуть лемму токена через pymorphy2 (кэшируется)."""
    return morph.parse(token)[0].normal_form


def normalize_text(text: str) -> str:
    """Очистить, лемматизировать и привести текст к виду как в обучении TF-IDF."""
    clean = text.lower()
    clean = URL_PATTERN.sub(" ", clean)
    clean = MENTION_PATTERN.sub(" ", clean)
    clean = clean.translate(PUNCT_TABLE)
    clean = STOP_CHARS.sub(" ", clean)
    tokens = [token for token in clean.split() if token]
    lemmas: List[str] = []
    for token in tokens:
        if token.startswith("#"):
            lemmas.append(token)
            continue
        lemmas.append(lemmatize_token(token))
    return " ".join(lemmas)


def count_emoji(text: str) -> int:
    """Посчитать количество emoji в тексте."""
    return len(EMOJI_PATTERN.findall(text))


def compute_numeric_features(frame: pd.DataFrame) -> np.ndarray:
    """Вычислить числовые признаки (длины, доли символов, количество знаков)."""
    texts = frame[TEXT_COLUMN].values
    normalized = frame["normalized_text"].values

    lengths = np.array([len(t) for t in texts])[:, None]
    word_counts = np.array([len(t.split()) for t in normalized])[:, None]
    emoji_counts = np.array([count_emoji(t) for t in texts])[:, None]
    hashtag_counts = np.array([t.count("#") for t in texts])[:, None]
    upper_ratio = np.array([
        sum(1 for ch in t if ch.isupper()) / max(1, len(t)) for t in texts
    ])[:, None]
    digit_ratio = np.array([
        sum(ch.isdigit() for ch in t) / max(1, len(t)) for t in texts
    ])[:, None]
    quote_ratio = np.array([
        t.count('"') / max(1, len(t)) for t in texts
    ])[:, None]
    sentence_marks = np.array([
        sum(t.count(mark) for mark in ".!?") for t in texts
    ])[:, None]

    return np.hstack([
        lengths,
        word_counts,
        emoji_counts,
        hashtag_counts,
        upper_ratio,
        digit_ratio,
        quote_ratio,
        sentence_marks,
    ])


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Выбор одного столбца DataFrame внутри конвейера sklearn."""

    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X, y=None):  # type: ignore[override]
        return self

    def transform(self, X):  # type: ignore[override]
        return X[self.column]


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """Выбор набора столбцов DataFrame."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)

    def fit(self, X, y=None):  # type: ignore[override]
        return self

    def transform(self, X):  # type: ignore[override]
        return X[self.columns]


class NumericFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Обёртка для вычисления числовых признаков внутри пайплайна."""

    def fit(self, X, y=None):  # type: ignore[override]
        return self

    def transform(self, X):  # type: ignore[override]
        return compute_numeric_features(X)


def tokenize_for_sequence(text: str) -> List[str]:
    """Токенизация текста для последовательностной модели."""
    return [tok for tok in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9#@]+", text.lower()) if tok]


def encode_document(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """Преобразовать текст в последовательность индексов с паддингом."""
    tokens = tokenize_for_sequence(text)
    indices = [vocab.get(tok, 1) for tok in tokens]  # 1 == <unk>
    if len(indices) >= max_len:
        return indices[:max_len]
    padded = indices + [0] * (max_len - len(indices))  # 0 == <pad>
    return padded


class TextCNN(nn.Module):
    """TextCNN из ноутбука: свёртки + max-pooling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        kernel_sizes: Sequence[int],
        num_filters: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size // 2) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        x = embedded.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            activated = F.relu(conv(x))
            pooled.append(F.max_pool1d(activated, kernel_size=activated.shape[-1]).squeeze(-1))
        merged = torch.cat(pooled, dim=1)
        merged = self.dropout(merged)
        return self.fc(merged)


class EnsembleClassifier:
    """Ансамбль логистической регрессии и TextCNN с усреднением вероятностей."""

    def __init__(
        self,
        *,
        artifacts_root: Path,
        linear_weight: float = 0.5,
        device: str | torch.device | None = None,
    ) -> None:
        if not 0.0 <= linear_weight <= 1.0:
            raise ValueError("linear_weight должен лежать в диапазоне [0, 1].")
        self.linear_weight = float(linear_weight)
        self.cnn_weight = 1.0 - self.linear_weight

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        root = artifacts_root.resolve()
        feature_union_path = root / "feature_union.joblib"
        linear_path = root / "logreg_elasticnet.joblib"
        textcnn_path = root / "textcnn_state.pt"
        label_index_path = root / "label_index.json"
        vocab_json = root / "sequence_vocab.json"

        self.feature_union = joblib.load(feature_union_path)
        self.linear_model = joblib.load(linear_path)

        state = torch.load(textcnn_path, map_location=self.device)
        config = state["config"]
        dropout = state.get("config", {}).get("embedding_dropout", 0.3)
        self.max_len = int(config["sequence_max_len"])

        self.vocab: Dict[str, int]
        if "vocab" in state and isinstance(state["vocab"], dict):
            self.vocab = {str(k): int(v) for k, v in state["vocab"].items()}
        else:
            with open(vocab_json, "r", encoding="utf-8") as fp:
                raw_vocab = json.load(fp)
            self.vocab = {str(k): int(v) for k, v in raw_vocab.items()}

        self.textcnn = TextCNN(
            vocab_size=int(config["vocab_size"]),
            embed_dim=int(config["embed_dim"]),
            num_classes=int(config["num_classes"]),
            kernel_sizes=config["kernel_sizes"],
            num_filters=int(config["num_filters"]),
            dropout=float(dropout),
        )
        self.textcnn.load_state_dict(state["model_state_dict"])
        self.textcnn.to(self.device)
        self.textcnn.eval()

        if "label_to_index" in state:
            label_to_index = {str(lbl): int(idx) for lbl, idx in state["label_to_index"].items()}
        else:
            with open(label_index_path, "r", encoding="utf-8") as fp:
                mapping = json.load(fp)
            label_to_index = {str(lbl): int(idx) for lbl, idx in mapping["label_to_index"].items()}

        self.index_to_label = {int(idx): str(label) for label, idx in label_to_index.items()}
        self.target_indices = sorted(self.index_to_label.keys())
        self.label_sequence = [self.index_to_label[idx] for idx in self.target_indices]

        classes = np.array(self.linear_model.classes_, dtype=int)
        class_positions = {int(cls): position for position, cls in enumerate(classes)}
        self.linear_alignment = [class_positions[idx] for idx in self.target_indices]

    def _prepare_dataframe(self, texts: Sequence[str]) -> pd.DataFrame:
        frame = pd.DataFrame({TEXT_COLUMN: list(texts)})
        frame["normalized_text"] = frame[TEXT_COLUMN].apply(normalize_text)
        return frame

    def _linear_predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        frame = self._prepare_dataframe(texts)
        features = self.feature_union.transform(frame)
        probabilities = self.linear_model.predict_proba(features)
        return probabilities[:, self.linear_alignment]

    def _textcnn_predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        sequences = [encode_document(text, self.vocab, self.max_len) for text in texts]
        batch = torch.tensor(sequences, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.textcnn(batch)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs[:, self.target_indices]

    def predict(self, texts: Sequence[str], *, return_proba: bool = False) -> Tuple[List[str], np.ndarray] | List[str]:
        if not texts:
            raise ValueError("Список текстов пуст — нечего классифицировать.")

        linear_probs = self._linear_predict_proba(texts)
        cnn_probs = self._textcnn_predict_proba(texts)
        combined = self.linear_weight * linear_probs + self.cnn_weight * cnn_probs
        argmax_indices = combined.argmax(axis=1)
        labels = [self.label_sequence[idx] for idx in argmax_indices]
        if return_proba:
            return labels, combined
        return labels


def format_probabilities(prob_vector: np.ndarray, labels: Sequence[str]) -> str:
    pairs = [f"{label}: {prob:.3f}" for label, prob in zip(labels, prob_vector)]
    return " | ".join(pairs)


def main() -> None:
    artifacts_root = Path(__file__).resolve().parent
    input_text = "Напишите сюда текст для классификации официального или разговорного стиля."
    texts = [input_text]

    classifier = EnsembleClassifier(artifacts_root=artifacts_root, linear_weight=0.5)
    labels, proba = classifier.predict(texts, return_proba=True)

    for text, label, prob_vector in zip(texts, labels, proba):
        print("===")
        print(text)
        print(f"Класс: {label}")
        print(f"Вероятности: {format_probabilities(prob_vector, classifier.label_sequence)}")


if __name__ == "__main__":
    main()
