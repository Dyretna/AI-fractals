# ai_fractals/registry/dataset_registry.py
"""
FractalRecord

A flat metadata container for CSV export. All fields are simple
primitives to ensure compatibility with tabular formats.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FractalRecord:
    """
    FractalRecord

    A flat metadata container representing a single fractal image.
    All nested JSON fields are expanded into top-level attributes.
    """

    timestamp: str
    relative_filepath: str
    fractal_type: str
    colormap: str
    width: int
    height: int
    max_iter: int
    depth: int
    bounds_xmin: float
    bounds_xmax: float
    bounds_ymin: float
    bounds_ymax: float
    score: float
    inside_ratio: float
    boundary_present: bool
    fractal_dimension: float
    entropy: float
    std: float
    edge_density: float
    dim_score: float
    entropy_score: float
    variance_score: float
    edge_score: float


# ================================================================
# DatasetRegistryManager
# ---------------------------------------------------------------
# A registration pipeline that:
#   1) Discovers PNG + JSON metadata pairs
#   2) Loads and normalizes metadata
#   3) Stores all entries as FractalRecord objects
#   4) Exports a CSV file
#
# ================================================================


class DatasetRegistryManager:
    """
    DatasetRegistryManager

    A dataset registration pipeline that collects fractal metadata
    and exports it as a flat CSV file. Filenames are preserved
    exactly as they appear in the dataset directory.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.records: list[FractalRecord] = []

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict:
        """
        Load a JSON file and return its contents as a dictionary.
        """
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------
    # 1. Register images
    # ------------------------------------------------------------

    def register_images(self):
        """
        register_images()

        Scans the dataset directory for PNG + JSON pairs.
        Loads metadata and stores all entries internally as
        FractalRecord objects.
        """
        png_files = sorted(self.root.rglob("*.png"))

        for png in png_files:
            json_file = png.with_suffix(".json")
            if not json_file.exists():
                continue

            meta = self._load_json(json_file)
            mx = meta["metrics"]

            rec = FractalRecord(
                timestamp=meta["timestamp"],
                relative_filepath=str(png.relative_to(self.root)),
                fractal_type=meta["fractal_type"],
                colormap=meta["colormap"],
                width=meta["width"],
                height=meta["height"],
                max_iter=meta["max_iter"],
                depth=meta["depth"],
                bounds_xmin=meta["bounds"][0],
                bounds_xmax=meta["bounds"][1],
                bounds_ymin=meta["bounds"][2],
                bounds_ymax=meta["bounds"][3],
                score=meta["score"],
                inside_ratio=mx["inside_ratio"],
                boundary_present=mx["boundary_present"],
                fractal_dimension=mx["fractal_dimension"],
                entropy=mx["entropy"],
                std=mx["std"],
                edge_density=mx["edge_density"],
                dim_score=mx["dim_score"],
                entropy_score=mx["entropy_score"],
                variance_score=mx["variance_score"],
                edge_score=mx["edge_score"],
            )

            self.records.append(rec)

    # ------------------------------------------------------------
    # 2. Write CSV
    # ------------------------------------------------------------

    def write_csv(self, path: Path):
        """
        write_csv(path)

        Writes all registered records to a flat CSV file.
        The CSV is rebuilt from scratch every time.
        """
        if not self.records:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.records[0].__dict__.keys())
            for r in self.records:
                writer.writerow(r.__dict__.values())
