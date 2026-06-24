# ai_fractals/data/rgb_dataset_builder.py

from __future__ import annotations

from datetime import datetime

from ai_fractals.data.savers import RGBSaver

from .base import BaseDatasetBuilder


class RGBDatasetBuilder(BaseDatasetBuilder):
    """
    Dataset builder for high-resolution RGB fractal renders.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.saver = RGBSaver(
            output_dir=self.output_dir,
            fractal_type=self.fractal_type,
            colormap=self.colormap,
            max_iter=self.hires_generator.max_iter,
        )

    # --------------------------------------------------------------
    # Tile processing
    # --------------------------------------------------------------
    def _process_tile(self, chosen: dict):
        xmin, xmax, ymin, ymax = chosen["bounds"]

        # High-res RGB render
        rgb = self.hires_generator.generate(xmin, xmax, ymin, ymax)

        # TileSearch already evaluated the raw tile
        score = chosen["score"]
        passed = chosen["accept"]
        metrics = chosen.get("metrics", {})

        return rgb, score, passed, metrics

    # --------------------------------------------------------------
    # Saving
    # --------------------------------------------------------------
    def _save(self, rgb, chosen, score, metrics):
        ts = datetime.now().isoformat()
        cid = datetime.fromisoformat(ts).strftime("%y%m%d%H%M%S")

        root = f"{cid}_{self.colormap}_iter{self.hires_generator.max_iter}"
        name = self.output_dir / f"{root}_d{self.depth:02d}"

        self.saver.save_img(rgb, name)

        if self.save_metadata:
            self.saver.save_metadata(
                name=name,
                resolution=f"({self.hires_generator.width}, {self.hires_generator.height})",
                bounds=chosen["bounds"],
                score=score,
                metrics=metrics,
                ts=ts,
                cid=cid,
            )
