"""
dataset_filter_manager.py

General-purpose image filtering manager for fractal datasets.

This class is intended to be used *before* dataset registration.
It supports removing images based on:

    - colormap
    - depth
    - score (via JSON metadata)
    - filename patterns
    - custom user-defined predicates

If ID prefixes are present (<id>_<cmap>_iter...), the manager can
automatically strip them before applying filters.

Typical pipeline:
    1. Generate raw PNG + JSON files with the dataset builder
    2. Run ImageFilterManager to remove unwanted images
    3. Run Dataset Registry Manager to assign IDs and build CSV
"""

import json
import shutil
from pathlib import Path


class DatasetFilterManager:
    def __init__(self, root: Path):
        self.root = Path(root)

    # -------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------

    @staticmethod
    def strip_id_prefix(filename: str) -> str:
        """
        Return the filename without its extension.
        No prefix stripping is performed.
        """
        return filename.split(".", 1)[0]

    def extract_cmap(self, filename: str) -> str:
        """
        Extract colormap from filename, with or without ID prefix.
        """

        name = self.strip_id_prefix(filename)
        return name.split("_iter")[0]

    def extract_depth(self, filename: str) -> int:
        """
        Extract depth from filename.
        Example:
            twilight_iter1024_d07.png -> 7
        """

        name = self.strip_id_prefix(filename)
        return int(name.split("_d")[1])

    # -------------------------------------------------------------
    # Core filtering engine
    # -------------------------------------------------------------

    def move_if(self, out_dir: Path, predicate):
        """
        Move all files for which predicate(filename) returns True.
        """

        out_dir.mkdir(parents=True, exist_ok=True)
        moved = 0

        for file in self.root.iterdir():
            if not file.is_file():
                continue

            if predicate(file.name):
                shutil.move(str(file), str(out_dir / file.name))
                moved += 1

        print(f"Moved {moved} files.")

    # -------------------------------------------------------------
    # Filter methods
    # -------------------------------------------------------------

    def remove_cmaps(self, out_dir: Path, bad_cmaps: list):
        """
        Remove all images whose colormap is in bad_cmaps.
        """

        self.move_if(out_dir, lambda fn: self.extract_cmap(fn) in bad_cmaps)

    def remove_depth_range(self, out_dir: Path, min_d: int, max_d: int):
        """
        Remove all images with depth in [min_d, max_d].
        """

        self.move_if(out_dir, lambda fn: min_d <= self.extract_depth(fn) <= max_d)

    def remove_low_score(self, out_dir: Path, threshold: float):
        """
        Remove images whose JSON metadata score is below threshold.
        """

        def pred(fn):
            json_path = self.root / fn.replace(".png", ".json")
            if not json_path.exists():
                return False
            with open(json_path) as f:
                meta = json.load(f)
            return meta["score"] < threshold

        self.move_if(out_dir, pred)
