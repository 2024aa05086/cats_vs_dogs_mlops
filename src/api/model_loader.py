"""Model loading and prediction utilities used by FastAPI."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


def load_label_map(path: str | Path) -> Dict[int, str]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    # stored as strings for JSON compatibility
    return {int(k): v for k, v in data.items()}


def load_model(model_path: str | Path) -> tf.keras.Model:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}. Train first or copy model into /models.")
    return tf.keras.models.load_model(p)


def predict_image(model: tf.keras.Model, img_rgb: np.ndarray) -> Tuple[int, float]:
    """Return (pred_label_int, prob_dog).

    Convention:
      0 -> cat
      1 -> dog
    """
    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError("img_rgb must be HxWx3 RGB array")

    img = tf.image.resize(img_rgb, [224, 224]).numpy().astype("float32")
    x = np.expand_dims(img, axis=0)
    prob_dog = float(model.predict(x, verbose=0).reshape(-1)[0])
    pred = int(prob_dog >= 0.5)
    return pred, prob_dog
