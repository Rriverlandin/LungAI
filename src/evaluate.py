"""Evaluate LungAI V4 on a NORMAL/PNEUMONIA directory tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from .inference import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD
except ImportError:
    from inference import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory with NORMAL/ and PNEUMONIA/")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels="inferred",
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="grayscale",
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)
    model = tf.keras.models.load_model(args.model, compile=False)
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in dataset]).astype(int)
    probabilities = model.predict(dataset, verbose=1).ravel()
    y_pred = (probabilities >= args.threshold).astype(int)

    print(f"Threshold: {args.threshold:.2f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], digits=4))


if __name__ == "__main__":
    main()
