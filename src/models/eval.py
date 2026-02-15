"""Evaluation helpers for classification metrics and plots."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score


def save_confusion_matrix(y_true, y_pred, out_path: Path, labels=("cat", "dog")) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_training_curves(history, out_path: Path) -> None:
    fig = plt.figure()
    for k in ["loss", "val_loss", "accuracy", "val_accuracy"]:
        if k in history.history:
            plt.plot(history.history[k], label=k)
    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_metrics_json(metrics: Dict[str, float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
