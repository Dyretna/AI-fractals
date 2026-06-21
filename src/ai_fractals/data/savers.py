# src/ai_fractals/data/savers.py

import json
from pathlib import Path

import cv2


class BaseSaver:
    """Base class for saving fractal-related images and metadata."""

    def __init__(self, output_dir, fractal_type, colormap, max_iter):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fractal_type = fractal_type
        self.colormap = colormap
        self.max_iter = max_iter
        self.img_filetype = ".png"

    def save_img(self, img, name: Path):
        raise NotImplementedError

    def save_metadata(self, name: Path, bounds, score, metrics, ts, cid):
        raise NotImplementedError


class RGBSaver(BaseSaver):
    """Saver for RGB fractal renders."""

    def save_img(self, img, name: Path):
        cv2.imwrite(
            str(name.with_suffix(self.img_filetype)),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

    def save_metadata(self, name: Path, resolution, bounds, score, metrics, ts, cid):
        meta = {
            "timestamp": ts,
            "compact_id": cid,
            "img_type": "rgb",
            "fractal_type": self.fractal_type,
            "colormap": self.colormap,
            "max_iter": self.max_iter,
            "resolution": resolution,
            "bounds": bounds,
            "score": score,
            "metrics": metrics,
        }

        with open(name.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)


class ShorelineSaver(BaseSaver):
    """Saver for shoreline (grayscale) images."""

    def save_img(self, shoreline, name: Path):
        cv2.imwrite(str(name.with_suffix(self.img_filetype)), shoreline)

    def save_metadata(self, name: Path, resolution, bounds, score, metrics, ts, cid):
        meta = {
            "timestamp": ts,
            "compact_id": cid,
            "img_type": "shoreline",
            "fractal_type": self.fractal_type,
            "colormap": self.colormap,
            "max_iter": self.max_iter,
            "resolution": resolution,
            "bounds": bounds,
            "score": score,
            "metrics": metrics,
        }

        with open(name.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
