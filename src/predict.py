"""Command-line entry point for a single LungAI prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .inference import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD, predict_image
except ImportError:  # Supports `python src/predict.py ...`.
    from inference import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD, predict_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify a chest X-ray with LungAI V4.")
    parser.add_argument("image", type=Path, help="Path to a JPEG or PNG chest X-ray")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = predict_image(args.image, model_path=args.model, threshold=args.threshold)
    print(f"Prediction: {result.label}")
    print(f"Pneumonia score: {result.pneumonia_score:.4f}")
    print(f"Decision threshold: {result.threshold:.2f}")
    print("Research prototype only — not a medical diagnosis.")


if __name__ == "__main__":
    main()
