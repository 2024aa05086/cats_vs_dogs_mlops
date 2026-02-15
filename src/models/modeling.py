"""Model architectures used for the assignment.

We provide two choices:
1) baseline_cnn: simple CNN from scratch (good baseline, fast to train)
2) mobilenetv2_transfer: transfer learning with ImageNet-pretrained MobileNetV2
"""

from __future__ import annotations
from dataclasses import dataclass
import tensorflow as tf


def build_baseline_cnn(input_shape=(224, 224, 3)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_mobilenetv2_transfer(input_shape=(224, 224, 3), fine_tune_at: int = -1) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")

    if fine_tune_at is not None and fine_tune_at >= 0:
        # Unfreeze from layer fine_tune_at onward
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    return model
