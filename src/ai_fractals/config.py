# config.py

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
IMAGES_DIR = PROJECT_ROOT / "images"

FRACTAL_IMG_DIR = PROJECT_ROOT / "images" / "fractal_images"
SHORELINE_IMG_DIR = PROJECT_ROOT / "images" / "shoreline_images"
