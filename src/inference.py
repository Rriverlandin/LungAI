"""Single-image inference for the LungAI V4 classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.60
DEFAULT_MODEL_PATH = Path("notebooks/models/LungAI_V4_final.keras")


@dataclass(frozen=True)
class Prediction:
    label: str
    pneumonia_score: float
    threshold: float


def prepare_image(image_path: str | Path) -> np.ndarray:
    """Load an image as the 224x224 single-channel tensor expected by V4."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        grayscale = image.convert("L").resize(IMAGE_SIZE)
        array = np.asarray(grayscale, dtype=np.float32)
    return array[np.newaxis, ..., np.newaxis]


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a saved Keras model without restoring its training configuration."""
    import tensorflow as tf

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. See models/README.md.")
    return tf.keras.models.load_model(path, compile=False)


def predict_image(
    image_path: str | Path,
    *,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> Prediction:
    """Classify one chest X-ray as NORMAL or PNEUMONIA."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    classifier = model if model is not None else load_model(model_path)
    output = classifier.predict(prepare_image(image_path), verbose=0)
    score = float(np.asarray(output).squeeze())
    label = "PNEUMONIA" if score >= threshold else "NORMAL"
    return Prediction(label=label, pneumonia_score=score, threshold=threshold)
