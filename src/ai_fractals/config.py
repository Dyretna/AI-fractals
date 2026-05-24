# config.py

import os
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------
# Load the path to the project.
# Set the Paths for input and output images
# ----------------------------------------------------------

load_dotenv()
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
IMAGES_DIR = PROJECT_DIR / "images"

FRACTAL_IMG_DIR = PROJECT_DIR / IMAGES_DIR / "fractal_images"
SHORELINE_IMG_DIR = PROJECT_DIR / IMAGES_DIR / "shoreline_images"


for path in [FRACTAL_IMG_DIR, SHORELINE_IMG_DIR]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is not correct")
