import os
import logging
import joblib
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, List

from scipy.sparse import csr_matrix
from transformers import AutoTokenizer, AutoModel

from server.models.fasttext.fasttext_classifier import FastTextClassifier
from server.models.ensemble.ensemble_classifier import EnsembleClassifier

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2, num_filters=50, filter_sizes=[3, 4, 5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(-1)).squeeze(-1) for c in conv_outs]
        x = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(x))


class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classifier")


class TextClassifier:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.tokenizers = {}
        self.vocabs = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.label_names = {
            0: "Разговорная речь",
            1: "Официально-деловая речь"
        }

        # Конфигурация моделей — пути относительно корня проекта
        self.model_config = {
            "logistic": {
                "model_path": str(self.models_dir / "lr" / "lr_model.pkl"),
                "vectorizer_path": str(self.models_dir / "lr" / "tfidf_vectorizer.pkl"),
                "type": "sklearn",
            },
            "cnn": {
                "model_path": str(self.models_dir / "cnn" / "cnn_best.pth"),
                "vocab_path": str(self.models_dir / "cnn" / "vocab.pkl"),
                "type": "torch_cnn",
                "embed_dim": 128,
                "num_filters": 50,
                "filter_sizes": [3, 4, 5],
            },
            "bert": {
                "model_path": str(self.models_dir / "bert" / "bert_classifier_best.pth"),
                "type": "torch_bert",
                "num_classes": 2,
            },
            "fasttext": {
                "model_path": str(self.models_dir / "fasttext" / "fasttext_ensemble.bin"),
                "label_index_path": str(self.models_dir / "fasttext" / "label_index.json"),
                "type": "fasttext",
            },
            "ensemble": {
                "artifacts_dir": str(self.models_dir / "ensemble"),
                "linear_weight": 0.5,
                "type": "ensemble",
            },
        }

    def load_model(self, model_type: str = "logistic") -> bool:
        if model_type in self.models:
            logger.info(f"Модель {model_type} уже загружена")
            return True

        try:
            config = self.model_config.get(model_type)
            if not config:
                logger.error(f"Неизвестный тип модели: {model_type}")
                return False

            model_kind = config.get("type")

            if model_kind == "sklearn":
                return self._load_sklearn_model(model_type, config)
            elif model_kind == "torch_cnn":
                return self._load_torch_cnn(model_type, config)
            elif model_kind == "torch_bert":
                return self._load_torch_bert(model_type, config)
            elif model_kind == "fasttext":
                return self._load_fasttext_model(model_type, config)
            elif model_kind == "ensemble":
                return self._load_ensemble_model(model_type, config)
            else:
                logger.error(f"Неизвестный тип модели: {model_kind}")
                return False

        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_type}: {e}")
            return False

    def _load_sklearn_model(self, model_type, config):
        model_file = config["model_path"]
        vectorizer_file = config.get("vectorizer_path")

        if not os.path.exists(model_file):
            logger.error(f"Файл модели не найден: {model_file}")
            return False

        self.models[model_type] = joblib.load(model_file)
        logger.info(f"Sklearn-модель {model_type} загружена")

        if vectorizer_file and os.path.exists(vectorizer_file):
            self.vectorizers[model_type] = joblib.load(vectorizer_file)
            logger.info(f"Векторизатор {model_type} загружен")
        else:
            self.vectorizers[model_type] = None
            logger.warning(f"Векторизатор для {model_type} отсутствует")

        return True

    def _load_torch_cnn(self, model_type, config):
        vocab_path = config["vocab_path"]
        if not os.path.exists(vocab_path):
            logger.error(f"Vocab не найден: {vocab_path}")
            return False

        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        if '<PAD>' not in vocab:
            vocab['<PAD>'] = 0
        if '<UNK>' not in vocab:
            vocab['<UNK>'] = 1

        self.vocabs[model_type] = vocab

        model = CNNClassifier(
            vocab_size=len(vocab),
            embed_dim=config.get("embed_dim", 128),
            num_filters=config.get("num_filters", 50),
            filter_sizes=config.get("filter_sizes", [3, 4, 5]),
            num_classes=2
        )
        model.load_state_dict(torch.load(config["model_path"], map_location=self.device, weights_only=True))
        model.eval()
        model.to(self.device)
        self.models[model_type] = model
        logger.info(f"CNN-модель {model_type} загружена на {self.device}")
        return True

    def _load_torch_bert(self, model_type, config):
        # Загружаем оригинальный RuBERT и токенизатор из Hugging Face
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        bert_backbone = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased", use_safetensors=True)

        model = BERTClassifier(bert_backbone, num_classes=config.get("num_classes", 2))
        model.load_state_dict(
            torch.load(config["model_path"], map_location=self.device, weights_only=True)
        )
        model.eval()
        model.to(self.device)

        self.models[model_type] = model
        self.tokenizers[model_type] = tokenizer
        logger.info(f"BERT-модель {model_type} загружена на {self.device}")
        return True

    def _load_fasttext_model(self, model_type, config):
        model_path = Path(config["model_path"])
        label_index_path = Path(config["label_index_path"])
        if not model_path.exists():
            logger.error(f"Файл fastText модели не найден: {model_path}")
            return False
        if not label_index_path.exists():
            logger.error(f"Файл индексов fastText не найден: {label_index_path}")
            return False

        classifier = FastTextClassifier(
            model_path=model_path,
            label_index_path=label_index_path,
        )
        self.models[model_type] = classifier
        logger.info(f"fastText-модель {model_type} загружена (device-independent)")
        return True

    def _load_ensemble_model(self, model_type, config):
        artifacts_dir = Path(config["artifacts_dir"])
        if not artifacts_dir.exists():
            logger.error(f"Каталог артефактов ансамбля не найден: {artifacts_dir}")
            return False

        linear_weight = config.get("linear_weight", 0.5)
        model = EnsembleClassifier(
            artifacts_root=artifacts_dir,
            linear_weight=float(linear_weight),
            device=self.device,
        )
        self.models[model_type] = model
        logger.info(f"Ансамбль {model_type} загружен на {self.device}")
        return True

    def predict(self, text: str, model: str = "logistic") -> Dict[str, Any]:
        if model == "all":
            return self._predict_all(text)

        if model not in self.model_config:
            logger.error(f"Неизвестный тип модели: {model}")
            return self._fallback_prediction(text, model=model)

        if model not in self.models:
            success = self.load_model(model)
            if not success:
                return self._fallback_prediction(text, model=model)

        try:
            config = self.model_config.get(model)
            model_kind = config.get("type")

            if model_kind == "ensemble":
                return self._predict_ensemble(text, model)
            elif model_kind == "torch_cnn":
                return self._predict_torch_cnn(text, model)
            elif model_kind == "torch_bert":
                return self._predict_torch_bert(text, model)
            elif model_kind == "fasttext":
                return self._predict_fasttext(text, model)
            elif model_kind == "sklearn":
                return self._predict_sklearn(text, model)
            else:
                raise ValueError(f"Неизвестный тип модели: {model_kind}")

        except Exception as e:
            logger.error(f"Ошибка предсказания моделью {model}: {e}")
            return self._fallback_prediction(text)

    def _predict_sklearn(self, text: str, model: str) -> Dict[str, Any]:
        processed_text = self._preprocess_text(text)
        vectorizer = self.vectorizers.get(model)
        if vectorizer:
            features = vectorizer.transform([processed_text])
        else:
            features = csr_matrix((1, 1))

        current_model = self.models[model]
        prediction = current_model.predict(features)[0]
        probability = self._get_probability(current_model, features)

        return {
            "success": True,
            "label": int(prediction),
            "label_name": self.label_names.get(prediction, "Неизвестный"),
            "confidence": float(probability),
            "text_length": len(text),
            "model": model
        }

    def _predict_all(self, text: str) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for model_name in self.model_config.keys():
            result = self.predict(text, model=model_name)
            results.append(dict(result))

        if not results:
            return self._fallback_prediction(text, model="all")

        summary = [
            f"{res.get('model')}: {res.get('label_name', 'Неизвестный')} ({res.get('confidence', 0.0):.2f})"
            for res in results
        ]

        votes: Dict[int, Dict[str, Any]] = {}
        for res in results:
            if not res.get("success"):
                continue
            label_value = res.get("label")
            if label_value is None:
                continue
            try:
                label_idx = int(label_value)
            except (TypeError, ValueError):
                continue

            entry = votes.setdefault(
                label_idx,
                {
                    "count": 0,
                    "confidence_sum": 0.0,
                    "best_confidence": -1.0,
                    "best_model": None,
                },
            )
            entry["count"] += 1
            confidence = float(res.get("confidence", 0.0))
            entry["confidence_sum"] += confidence
            if confidence > entry["best_confidence"]:
                entry["best_confidence"] = confidence
                entry["best_model"] = res.get("model")

        if not votes:
            return self._fallback_prediction(text, model="all")

        def _vote_score(item):
            label_idx, data = item
            avg_conf = data["confidence_sum"] / max(1, data["count"])
            return (data["count"], avg_conf, data["best_confidence"])

        winning_label, winning_data = max(votes.items(), key=_vote_score)
        best_model_name = winning_data["best_model"]
        best_model_result = next(
            (res for res in results if res.get("model") == best_model_name and int(res.get("label", -1)) == winning_label),
            None,
        )
        if best_model_result is None:
            best_model_result = max(
                (res for res in results if int(res.get("label", -1)) == winning_label),
                key=lambda item: item.get("confidence", 0.0),
                default=None,
            )

        winning_confidence = winning_data["best_confidence"] if winning_data["best_confidence"] >= 0 else (
            best_model_result.get("confidence", 0.0) if best_model_result else 0.0
        )
        label_name = self.label_names.get(winning_label, "Неизвестный")

        success = any(res.get("success") for res in results)
        fallback_flags = [bool(res.get("fallback")) for res in results if "fallback" in res]
        fallback = bool(fallback_flags) and all(fallback_flags)

        return {
            "success": success,
            "label": winning_label,
            "label_name": label_name,
            "confidence": winning_confidence,
            "text_length": len(text),
            "model": "all",
            "best_model": best_model_name,
            "details": results,
            "summary": summary,
            "votes": {label: data["count"] for label, data in votes.items()},
            "fallback": fallback,
        }

    def _predict_torch_cnn(self, text: str, model: str) -> Dict[str, Any]:
        vocab = self.vocabs[model]
        max_len = 256

        seq = [vocab.get(w, vocab['<UNK>']) for w in text.split()]
        seq = seq[:max_len]
        if len(seq) < max_len:
            seq += [vocab['<PAD>']] * (max_len - len(seq))
        x = torch.tensor([seq], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.models[model](x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0][pred].item()

        return {
            "success": True,
            "label": pred,
            "label_name": self.label_names.get(pred, "Неизвестный"),
            "confidence": confidence,
            "text_length": len(text),
            "model": model
        }

    def _predict_torch_bert(self, text: str, model: str) -> Dict[str, Any]:
        tokenizer = self.tokenizers[model]
        enc = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.models[model](input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0][pred].item()

        return {
            "success": True,
            "label": pred,
            "label_name": self.label_names.get(pred, "Неизвестный"),
            "confidence": confidence,
            "text_length": len(text),
            "model": model
        }

    def _predict_fasttext(self, text: str, model: str) -> Dict[str, Any]:
        classifier: FastTextClassifier = self.models[model]
        labels, probabilities = classifier.predict([text], return_proba=True)
        proba_vector = probabilities[0]
        best_idx = int(np.argmax(proba_vector))
        raw_label = classifier.label_sequence[best_idx]
        label = self._normalize_label(raw_label)
        confidence = float(proba_vector[best_idx])

        return {
            "success": True,
            "label": label,
            "label_name": self.label_names.get(label, str(raw_label)),
            "confidence": confidence,
            "text_length": len(text),
            "model": model,
        }

    def _preprocess_text(self, text: str) -> str:
        return ' '.join(text.strip().lower().split())

    def _basic_features(self, text: str):
        return csr_matrix((1, 1))

    def _get_probability(self, model, features) -> float:
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                return float(np.max(proba))
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(features)[0]
                return float(1 / (1 + np.exp(-abs(decision))))
        except Exception as e:
            logger.warning(f"Не удалось получить вероятность: {e}")
        return 0.8

    def _fallback_prediction(self, text: str, model: str = "logistic") -> Dict[str, Any]:
        official_keywords = ['уважаемый', 'заявление', 'документ', 'официальный', 'служебный', 'прошу', 'предоставить']
        informal_keywords = ['привет', 'пока', 'спасибо', 'пожалуйста', 'дружище', 'норм', 'окей', 'класс']

        text_lower = text.lower()
        official_count = sum(1 for word in official_keywords if word in text_lower)
        informal_count = sum(1 for word in informal_keywords if word in text_lower)

        if official_count > informal_count:
            label = 1
            confidence = min(0.7 + official_count * 0.1, 0.95)
        elif informal_count > official_count:
            label = 0
            confidence = min(0.7 + informal_count * 0.1, 0.95)
        else:
            label = 1 if len(text) > 100 else 0
            confidence = 0.6

        return {
            "success": True,
            "label": label,
            "label_name": self.label_names.get(label, "Неизвестный"),
            "confidence": confidence,
            "text_length": len(text),
            "fallback": True,
            "model": model
        }

    def get_available_models(self) -> List[str]:
        available: List[str] = []
        for model_type, config in self.model_config.items():
            model_kind = config.get("type")
            if model_kind == "sklearn":
                if os.path.exists(config["model_path"]):
                    available.append(model_type)
            elif model_kind == "torch_cnn":
                if os.path.exists(config["model_path"]) and os.path.exists(config["vocab_path"]):
                    available.append(model_type)
            elif model_kind == "torch_bert":
                if os.path.exists(config["model_path"]):
                    available.append(model_type)
            elif model_kind == "fasttext":
                model_path = Path(config["model_path"])
                label_index_path = Path(config["label_index_path"])
                if model_path.exists() and label_index_path.exists():
                    available.append(model_type)
            elif model_kind == "ensemble":
                artifacts_dir = Path(config["artifacts_dir"])
                if artifacts_dir.exists():
                    available.append(model_type)
        if available:
            available.append("all")
        return available

    def preload_models(self, model_types: list = None):
        if model_types is None:
            model_types = list(self.model_config.keys())
        for model_type in model_types:
            self.load_model(model_type)

    def _predict_ensemble(self, text: str, model: str) -> Dict[str, Any]:
        current_model = self.models[model]
        labels, proba = current_model.predict([text], return_proba=True)
        raw_label = labels[0]
        label_idx = self._normalize_label(raw_label)
        confidence = float(np.max(proba[0]))
        label_name = self.label_names.get(label_idx, str(raw_label))
        return {
            "success": True,
            "label": label_idx,
            "label_name": label_name,
            "confidence": confidence,
            "text_length": len(text),
            "model": model
        }

    @staticmethod
    def _normalize_label(value: Any) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            logger.warning("Не удалось преобразовать метку %s к int, используем 0", value)
            return 0



text_classifier = TextClassifier()
