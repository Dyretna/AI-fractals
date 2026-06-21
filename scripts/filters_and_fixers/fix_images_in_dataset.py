# scripts/fix_images_in_dataset.py
"""
Batch-processing script for fractal images based on filename pattern
matching (e.g., colormap names). Applies visibility enhancements such
as darkening, CLAHE, saturation boost, and sharpening.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from ai_fractals.processing import HistogramEqualizers

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def darken(img: np.ndarray, factor: float = 0.80) -> np.ndarray:
    """Darken the image globally."""
    return (img.astype("float32") * factor).clip(0, 255).astype("uint8")


def apply_clahe(img: np.ndarray, clip=3.0, grid=(4, 4)) -> np.ndarray:
    """Apply CLAHE enhancement using HistogramEqualizers."""
    return HistogramEqualizers.clahe_color(img, clip=clip, grid=grid)


def boost_saturation(img: np.ndarray, factor: float = 2.5) -> np.ndarray:
    """Boost saturation in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[..., 1] *= factor
    hsv = hsv.clip(0, 255).astype("uint8")
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Apply a simple sharpening kernel."""
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )
    return cv2.filter2D(img, -1, kernel)


def find_matching_images(directory: Path, patterns: list[str]) -> list[Path]:
    """
    Return a list of image paths whose filenames contain ANY of the patterns.
    """
    pngs = list(directory.glob("*.png"))
    return [p for p in pngs if any(pat in p.name for pat in patterns)]


# ------------------------------------------------------------
# main functions
# ------------------------------------------------------------


def fix_cmaps(img: np.ndarray) -> np.ndarray:
    img = darken(img)
    img = apply_clahe(img)
    img = boost_saturation(img)
    img = sharpen(img)

    return img


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    if not PROJECT_ROOT.is_dir():
        raise IsADirectoryError("check PROJECT_ROOT in .env")

    # User configuration
    img_dir = PROJECT_ROOT / "dataset" / "mandelbrot" / "1024_1024_iter1024"
    out_dir = img_dir / "fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find matching images
    targets = find_matching_images(
        directory=img_dir,
        patterns=[
            "Pastel1",
            "Pastel1_r",
            "Pastel2",
            "Pastel2_r",
        ],
    )
    print(f"Found {len(targets)} matching images.")

    for path in targets:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read: {path}")
            continue

        # Apply processing pipeline
        img = fix_cmaps(img)

        # Save result
        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), img)

        print(f"[OK] Processed: {path.name}")

    print("Done.")
