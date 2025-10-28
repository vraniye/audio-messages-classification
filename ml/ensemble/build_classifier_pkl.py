"""Utility script to pack the ensemble classifier into a single joblib file."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

import sys

from ensemble_classifier import (
    ColumnSelector,
    ColumnsSelector,
    EnsembleClassifier,
    NumericFeaturesTransformer,
)

current_module = sys.modules[__name__]
current_module.ColumnSelector = ColumnSelector
current_module.ColumnsSelector = ColumnsSelector
current_module.NumericFeaturesTransformer = NumericFeaturesTransformer


def build_classifier(output: Path, *, linear_weight: float) -> Path:
    """Create `classifier.pkl` with all ensemble components on CPU."""
    artifacts_root = Path(__file__).resolve().parent

    classifier = EnsembleClassifier(
        artifacts_root=artifacts_root,
        linear_weight=linear_weight,
        device="cpu",
    )
    classifier.textcnn.to("cpu")

    joblib.dump(classifier, output, compress=3)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack ensemble artifacts into classifier.pkl for deployment.",
    )
    parser.add_argument(
        "--linear-weight",
        type=float,
        default=0.5,
        help="Weight for the linear model in the ensemble (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "classifier.pkl",
        help="Target path for the serialized classifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_classifier(args.output, linear_weight=args.linear_weight)
    print(f"Ensemble classifier saved to {output_path}")


if __name__ == "__main__":
    main()
