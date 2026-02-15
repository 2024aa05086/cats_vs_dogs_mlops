"""TensorFlow dataset utilities for reading the processed dataset."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import tensorflow as tf


def load_splits_csv(processed_dir: Path) -> pd.DataFrame:
    csv_path = processed_dir / "splits.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing splits.csv at {csv_path}. Run preprocess stage.")
    return pd.read_csv(csv_path)


def make_tf_dataset(df: pd.DataFrame, processed_dir: Path, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    paths = [str((processed_dir / p).resolve()) for p in df["rel_path"].tolist()]
    labels = df["label"].map({"cat": 0, "dog": 1}).astype("int32").tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path: tf.Tensor, label: tf.Tensor):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2000, len(paths)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
