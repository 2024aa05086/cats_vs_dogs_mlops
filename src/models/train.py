"""Train a Cats vs Dogs classifier and log everything to MLflow.

Stage: dvc repro train

Logs:
- parameters (hyperparams, model type, augment flag)
- metrics (accuracy, precision, recall, f1)
- artifacts:
  - confusion_matrix.png
  - training_curves.png
  - saved model (model.keras)
  - label map (label_map.json)
  - metrics.json (for DVC metrics)

The MLflow tracking URI defaults to local file store `mlruns`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import tensorflow as tf

from src.utils.config import load_yaml
from src.models.dataset import load_splits_csv, make_tf_dataset, augmentation_layer
from src.models.modeling import build_baseline_cnn, build_mobilenetv2_transfer
from src.models.eval import compute_metrics, save_confusion_matrix, save_training_curves, save_metrics_json


def compile_model(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to params.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.params)
    processed_dir = Path(cfg["data"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    model_path = Path(cfg["paths"]["model_path"])
    label_map_path = Path(cfg["paths"]["label_map_path"])
    mlruns_dir = Path(cfg["paths"]["mlruns_dir"])

    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["learning_rate"])
    model_type = str(cfg["train"]["model_type"])
    fine_tune_at = int(cfg["train"]["fine_tune_at"])
    augment = bool(cfg["data"]["augment"])

    mlflow.set_tracking_uri(str(mlruns_dir))
    mlflow.set_experiment("cats_vs_dogs")

    df = load_splits_csv(processed_dir)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    train_ds = make_tf_dataset(df_train, processed_dir, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(df_val, processed_dir, batch_size=batch_size, shuffle=False)
    test_ds = make_tf_dataset(df_test, processed_dir, batch_size=batch_size, shuffle=False)

    # Build model
    if model_type == "baseline_cnn":
        model = build_baseline_cnn(input_shape=(224, 224, 3))
    elif model_type == "mobilenetv2_transfer":
        model = build_mobilenetv2_transfer(input_shape=(224, 224, 3), fine_tune_at=fine_tune_at)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Optional data augmentation (only on train)
    if augment:
        aug = augmentation_layer()
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = aug(inputs)
        outputs = model(x)
        model = tf.keras.Model(inputs, outputs, name=f"{model.name}_aug")

    model = compile_model(model, lr=lr)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    with mlflow.start_run():
        mlflow.log_params({
            "model_type": model_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "augment": augment,
            "fine_tune_at": fine_tune_at,
        })

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

        # Evaluate on test
        y_true = []
        y_prob = []
        for xb, yb in test_ds:
            prob = model.predict(xb, verbose=0).reshape(-1)
            y_prob.extend(prob.tolist())
            y_true.extend(yb.numpy().reshape(-1).tolist())

        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_true, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Artifacts
        reports_dir.mkdir(parents=True, exist_ok=True)
        cm_path = reports_dir / "confusion_matrix.png"
        curves_path = reports_dir / "training_curves.png"
        save_confusion_matrix(y_true, y_pred, cm_path, labels=("cat", "dog"))
        save_training_curves(history, curves_path)

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(curves_path))

        # Save model + label map
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        label_map = {"0": "cat", "1": "dog"}
        label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(label_map_path))

        # DVC metrics file
        metrics_path = reports_dir / "metrics.json"
        save_metrics_json(metrics, metrics_path)
        mlflow.log_artifact(str(metrics_path))

    print("Training complete.")
    print(f"Model saved to: {model_path.resolve()}")
    print(f"Reports saved to: {reports_dir.resolve()}")
    print(f"MLflow runs: {mlruns_dir.resolve()}")


if __name__ == "__main__":
    main()
