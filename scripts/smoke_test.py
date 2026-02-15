"""Post-deploy smoke test.

Checks:
1) /health returns ok/model_not_loaded
2) /predict succeeds for a given image
Fails with non-zero exit code if any check fails.

Usage:
  python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --image tests/assets/sample_dog.jpg
"""

from __future__ import annotations
import argparse
import sys
import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    # health
    r = requests.get(f"{args.base_url}/health", timeout=10)
    if r.status_code != 200:
        print("Health check failed:", r.status_code, r.text)
        return 1
    print("Health:", r.json())

    # predict
    with open(args.image, "rb") as f:
        files = {"file": (args.image, f, "image/jpeg")}
        r = requests.post(f"{args.base_url}/predict", files=files, timeout=30)

    if r.status_code != 200:
        print("Predict failed:", r.status_code, r.text)
        return 1

    print("Predict OK:", r.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
