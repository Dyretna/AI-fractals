# scripts/register_to_csv.py

"""
register_to_csv.py

This module performs dataset registration by scanning a directory
containing fractal images and their JSON metadata. All metadata is
loaded, normalized, and exported as a flat CSV file.

This script assumes that the dataset has already been cleaned by any
optional filtering steps. Filenames are preserved exactly as they
exist on disk, and no renaming or ID assignment is performed.

Pipeline recommendation:
    1. Generate raw images with the dataset builder (PNG + JSON)
    2. Optionally run filtering scripts to remove unwanted images
    3. Run this script to:
         - register all remaining images
         - load and flatten metadata
         - export a complete metadata CSV
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ai_fractals.data import DatasetRegistryManager

load_dotenv()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
DATASET_DIR = PROJECT_ROOT / "dataset"

mgr = DatasetRegistryManager(DATASET_DIR)
mgr.register_images()
mgr.write_csv(DATASET_DIR / "metadata.csv")
