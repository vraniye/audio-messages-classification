#!/usr/bin/env python3
"""
CLI-утилита для локального прогона текстов через все модели классификатора.

Примеры:
    python tools/classify_text.py "Нам необходимо подготовить отчёт и согласовать сроки."
    python tools/classify_text.py "Привет! Как дела?" -m logistic cnn
    python tools/classify_text.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _dump_result(result: dict, compact: bool) -> None:
    if compact:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n▶ Модель: {result.get('model')}")
    print(
        f"  Итоговый класс: {result.get('label_name')} "
        f"(label={result.get('label')}, confidence={result.get('confidence', 0.0):.2f})"
    )

    details: Iterable[dict] = result.get("details") or []
    if details:
        print("  Детализация по моделям:")
        for item in details:
            print(
                f"    - {item.get('model')}: {item.get('label_name')} "
                f"(label={item.get('label')}, confidence={item.get('confidence', 0.0):.2f})"
            )

    votes = result.get("votes")
    if votes:
        print("  Голоса:", ", ".join(f"{label} → {count}" for label, count in votes.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальная классификация текстов.")
    parser.add_argument(
        "text",
        nargs="?",
        help="Текст для классификации. Если не передан, используется stdin.",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        nargs="+",
        default=["all"],
        help="Список моделей (по умолчанию all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать доступные модели и выйти.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Выводить результат в JSON.",
    )

    args = parser.parse_args()

    _ensure_repo_on_path()
    from server.classifier import TextClassifier  # pylint: disable=C0415

    classifier = TextClassifier()

    if args.list:
        available = classifier.get_available_models()
        print("Доступные модели:", ", ".join(available or ["<нет артефактов>"]))
        return

    text = args.text or sys.stdin.read()
    if not text.strip():
        parser.error("Не передан текст для классификации.")

    models: List[str] = args.models
    for model_name in models:
        result = classifier.predict(text, model=model_name)
        _dump_result(result, compact=args.json)


if __name__ == "__main__":
    main()

