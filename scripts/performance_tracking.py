"""Post-deployment model performance tracking (simulated).

Goal (M5):
- Collect a small batch of requests + true labels
- Compute accuracy and log to MLflow (as a separate run)

Input format:
CSV with columns:
  image_path,true_label
where true_label is 'cat' or 'dog'

Example:
  python scripts/performance_tracking.py --csv data/post_deploy_samples.csv --base-url http://127.0.0.1:8000
"""

from __future__ import annotations
import argparse
import time
import pandas as pd
import requests
import mlflow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with image_path,true_label")
    ap.add_argument("--base-url", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    correct = 0
    total = 0
    latencies = []

    for _, row in df.iterrows():
        img_path = row["image_path"]
        true_label = str(row["true_label"]).strip().lower()

        with open(img_path, "rb") as f:
            t0 = time.time()
            r = requests.post(f"{args.base_url}/predict", files={"file": f}, timeout=30)
            dt = time.time() - t0
            latencies.append(dt)

        r.raise_for_status()
        pred = r.json()["label"].strip().lower()
        correct += int(pred == true_label)
        total += 1

    acc = correct / max(1, total)
    avg_latency = sum(latencies) / max(1, len(latencies))

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("cats_vs_dogs_post_deploy")

    with mlflow.start_run():
        mlflow.log_metric("post_deploy_accuracy", acc)
        mlflow.log_metric("avg_latency_seconds", avg_latency)

    print({"post_deploy_accuracy": acc, "avg_latency_seconds": avg_latency})


if __name__ == "__main__":
    main()
