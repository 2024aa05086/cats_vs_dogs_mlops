from pathlib import Path
import numpy as np
import cv2
from src.data.preprocess import resize_rgb


def test_resize_rgb_creates_output(tmp_path: Path):
    # Create a dummy BGR image and save it
    img = np.zeros((50, 80, 3), dtype=np.uint8)
    src = tmp_path / "in.jpg"
    cv2.imwrite(str(src), img)

    dst = tmp_path / "out.jpg"
    ok = resize_rgb(src, dst, size=224)

    assert ok is True
    assert dst.exists()
    out = cv2.imread(str(dst))
    assert out.shape[:2] == (224, 224)
