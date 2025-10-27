# Ансамбль логрег + TextCNN

## Артефакты
- `feature_union.joblib` — пайплайн TF-IDF + числовые признаки, сохранённый через `joblib`.
- `logreg_elasticnet.joblib` — обученная логистическая регрессия на TF-IDF.
- `textcnn_state.pt` — чекпоинт TextCNN (PyTorch state_dict + конфигурация и словарь).
- `label_index.json` — маппинг меток к индексам (`label_to_index`, `index_to_label`).
- `sequence_vocab.json` — словарь токенов для TextCNN (если отсутствует в checkpoint).

Все файлы должны лежать рядом со скриптом `ensemble_classifier.py`.

## Скрипт `ensemble_classifier.py`
- В начале определены функции предобработки (`normalize_text`, `count_emoji`) и классы-помощники для совпадения пайплайна TF-IDF.
- `TextCNN` — реализация сверточной модели, используемой в ноутбуке.
- `EnsembleClassifier` загружает артефакты, выравнивает порядок классов и считает усреднённые вероятности (`linear_weight` регулирует вклад логистической регрессии).
- `main` задаёт входной текст в переменную `input_text`, классифицирует его и печатает метку и вероятности.

## Зависимости
Скопируйте список в `requirements.txt`, либо установите пакеты вручную:

```
joblib
numpy
pandas
torch
pymorphy2
scikit-learn
```
