# scripts/fix_colormap_images.py
"""
fix_colormap_images.py

Generic batch-processing script for fractal images based on filename
pattern matching (e.g., colormap names). Applies local histogram
equalization (CLAHE) using HistogramEqualizers to improve visibility.

mainly implemented to "save" ottherwise interesting fractals
with difficult cmaps, such as the Pastel-series.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from ai_fractals.processing import HistogramEqualizers

if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

    if not PROJECT_ROOT.is_dir():
        raise IsADirectoryError("check PROJECT_ROOT in .env")

    # ------------------------------------------------------------
    # User configuration
    # ------------------------------------------------------------
    KEYWORDS = [
        "Pastel1",
        "Pastel1_r",
        "Pastel2",
        "Pastel2_r",
    ]

    img_dir = PROJECT_ROOT / "dataset" / "mandelbrot" / "1024_1024_iter1024"
    out_dir = img_dir / "fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------
    pngs = list(img_dir.glob("*.png"))
    targets = [p for p in pngs if any(k in p.name for k in KEYWORDS)]

    print(f"Found {len(targets)} matching images.")

    for path in targets:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read: {path}")
            continue

        # Darken globally
        dark = (img.astype("float32") * 0.80).clip(0, 255).astype("uint8")

        # Apply CLAHE enhancement
        enh = HistogramEqualizers.clahe_color(
            dark,
            clip=3.0,  # more contrast
            grid=(4, 4),  # bigger tiles -> bigger local effect
        )

        # Saturation boost
        hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 1] *= 2.5  # increase saturation
        hsv = hsv.clip(0, 255).astype("uint8")
        enh = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enh = cv2.filter2D(enh, -1, kernel)

        # Save with suffix
        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), enh)

        print(f"[OK] Processed: {path.name}")

    print("Done.")
