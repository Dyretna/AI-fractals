# scripts/filter_dataset.py
"""
filter_dataset.py

Pre-processing script for fractal datasets.

This script uses DatasetFilterManager to remove unwanted images
(colormap, depth, score, etc.) *before* the Dataset registration.

Run this on the raw builder output directory (PNG + JSON) to clean
the dataset before registration.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ai_fractals.data import OUT_FILTERED, DatasetFilterManager

load_dotenv()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))


if __name__ == "__main__":
    if not PROJECT_ROOT.is_dir():
        raise IsADirectoryError("check PROJECT_ROOT in .env")

    # Input/output directories
    img_dir = PROJECT_ROOT / "dataset" / "mandelbrot" / "1024_1024_iter1024"
    out_dir = PROJECT_ROOT / "dataset" / "out" / "1024_1024_iter1024"

    # Create filter manager
    mgr = DatasetFilterManager(img_dir)

    # Example: remove unwanted colormaps
    mgr.remove_cmaps(out_dir, OUT_FILTERED)
    mgr.remove_depth_range(out_dir, 11, 15)
    # mgr.remove_low_score(threshold=0.15)

    print("Filtering complete.")
