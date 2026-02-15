"""
Download dataset using kagglehub (CI friendly).

DVC stage:
    dvc repro download
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import kagglehub

from src.utils.config import load_yaml


def download_dataset(dataset_slug: str, raw_dir: Path) -> None:
    """
    Downloads dataset using kagglehub and copies into data/raw.

    kagglehub automatically caches downloads, so repeated
    CI runs DO NOT re-download large files.
    """
    print(f"Downloading dataset: {dataset_slug}")

    # kagglehub returns local cached path
    dataset_path = kagglehub.dataset_download(dataset_slug)

    dataset_path = Path(dataset_path)
    print("Cached dataset location:", dataset_path)

    if not dataset_path.exists():
        raise RuntimeError("Dataset download failed.")

    # Clean target directory
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    shutil.copytree(dataset_path, raw_dir)

    print(f"Dataset copied to: {raw_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.params)

    dataset_slug = cfg["data"]["kaggle_dataset"]
    raw_dir = Path(cfg["data"]["raw_dir"])

    raw_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(dataset_slug, raw_dir)


if __name__ == "__main__":
    main()
