"""Preprocess raw images into 224x224 RGB, train/val/test splits, and an index file.

Assumptions about raw dataset:
- The dataset contains images in some nested directory structure.
- Filenames or folder names contain class hints like 'cat'/'dog' OR there exist
  top-level folders 'cats' and 'dogs'.

This script tries to infer labels robustly:
1) If parent folder name is cats/dogs (case-insensitive), uses that.
2) Else if filename contains 'cat' or 'dog', uses that.
3) Otherwise skips the file (reported).

Outputs:
- data/processed/train|val|test/<class>/*.jpg (resized, RGB)
- data/processed/splits.csv with relative paths and split labels

Run via DVC:
    dvc repro preprocess
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

from src.utils.config import load_yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_label(path: Path) -> str | None:
    name = path.name.lower()
    parent = path.parent.name.lower()

    if parent in {"cat", "cats"}:
        return "cat"
    if parent in {"dog", "dogs"}:
        return "dog"
    if "cat" in name:
        return "cat"
    if "dog" in name:
        return "dog"
    return None


def resize_rgb(in_path: Path, out_path: Path, size: int) -> bool:
    """Load an image, convert to RGB, resize to (size,size), and save as JPEG.

    Returns True on success, False if the image couldn't be read.
    """
    img_bgr = cv2.imread(str(in_path))
    if img_bgr is None:
        return False
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save back as BGR for OpenCV imwrite
    img_out = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return True


def list_images(raw_dir: Path) -> List[Path]:
    files = []
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def split_indices(n: int, train: float, val: float, test: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    assert abs((train + val + test) - 1.0) < 1e-6, "Splits must sum to 1.0"
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(n * train)
    n_val = int(n * val)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to params.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.params)
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    size = int(cfg["data"]["image_size"])
    splits = cfg["data"]["split"]
    seed = int(cfg["data"]["seed"])

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}. Run download stage first.")

    # Collect labeled images
    all_imgs = list_images(raw_dir)
    labeled = []
    skipped = 0
    for p in all_imgs:
        lab = infer_label(p)
        if lab is None:
            skipped += 1
            continue
        labeled.append((p, lab))

    if not labeled:
        raise RuntimeError(f"No labeled images found under {raw_dir}.")
    print(f"Found {len(labeled)} labeled images; skipped {skipped} unlabeled.")

    # Split
    train_idx, val_idx, test_idx = split_indices(
        len(labeled), float(splits["train"]), float(splits["val"]), float(splits["test"]), seed
    )

    # Clean output and write
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                for p in child.rglob("*"):
                    if p.is_file():
                        p.unlink()
                child.rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {}
    for i in train_idx: split_map[i] = "train"
    for i in val_idx: split_map[i] = "val"
    for i in test_idx: split_map[i] = "test"

    # Process + save split index
    csv_path = out_dir / "splits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "rel_path", "source_path"])

        for i, (src, lab) in enumerate(tqdm(labeled, desc="Preprocessing")):
            split = split_map[i]
            rel = Path(split) / lab / f"{src.stem}_{i}.jpg"
            dst = out_dir / rel

            ok = resize_rgb(src, dst, size=size)
            if not ok:
                continue

            w.writerow([split, lab, str(rel).replace("\\", "/"), str(src)])

    print(f"Processed dataset written to: {out_dir.resolve()}")
    print(f"Split index: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
