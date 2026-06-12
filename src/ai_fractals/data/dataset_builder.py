import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.generators import (
    BaseFractalGenerator,
    JuliaGenerator,
    MandelbrotGenerator,
)


class FractalDatasetBuilder:
    """
    Automated batch fractal dataset generator using tile-search refinement.
    """

    def __init__(
        self,
        fractal_type="mandelbrot",
        save_min_depth: int = 2,
        save_max_depth: int = 10,
        width: int = 1200,
        height: int = 1200,
        max_iter: int = 900,
        tile_resolution: int = 200,
        quality_threshold: float = 0.3,
        n_tiles: int = 5,
        colormap: str = "twilight",
        output_dir: Path = None,
    ):
        # set paths
        load_dotenv()
        project_root = Path(os.getenv("PROJECT_ROOT"))
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = Path(
                project_root / "dataset" / "fractals" / fractal_type
            ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # set params
        self.fractal_type = fractal_type
        self.save_min_depth = save_min_depth
        self.save_max_depth = save_max_depth
        self.max_iter = max_iter
        self.n_tiles = n_tiles
        self.colormap = colormap

        # Generators and evaluator
        self.tile_gen: BaseFractalGenerator = self._create_generator(
            fractal_type=self.fractal_type,
            width=tile_resolution,
            height=tile_resolution,
            max_iter=max_iter,
            colormap=colormap,
        )
        self.hires_gen: BaseFractalGenerator = self._create_generator(
            fractal_type=self.fractal_type,
            width=width,
            height=height,
            max_iter=max_iter,
            colormap=self.colormap,
        )

        self.evaluator = FractalQualityEvaluator(quality_threshold=quality_threshold)

        # Bounds
        self.bounds = (-2.0, 1.0, -1.5, 1.5)
        self.depth = 0

        # stuck detection
        self.consectuive_fallbacks = 0
        self.max_fallbacks = 10

        # logging
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.log.info(f"Output dir: {self.output_dir}")
        self.log.info(f"Dir exists: {self.output_dir.exists()}")

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def run(self, n_images):
        """
        Generate exactly n_images saved fractal images.
        """
        saved = self._count_existing_pngs()
        target = saved + n_images

        pbar = tqdm(total=n_images, desc="Generating fractals")

        while saved < target:
            saved_before = saved
            saved += self.step()

            if saved > saved_before:
                pbar.update(1)

            if self.consectuive_fallbacks >= self.max_fallbacks:
                self.log.warning("Too many fallbacks - resetting search space")
                self.reset()

        pbar.close()
        self.log.info(f"Completed: {n_images} new images saved.")

    def step(self):
        """
        Perform one tile-search refinement step.
        Returns 1 of an image was saved, else 0
        """
        # dynamic threshold
        self.evaluator.quality_threshold = 0.12 if self.depth == 0 else 0.3

        tiles = self._tile_and_score(*self.bounds)
        candidates = [t for t in tiles if t["accept"]]

        # Fallback: pich top 3
        pool = (
            candidates
            if candidates
            else sorted(tiles, key=lambda t: t["score"], reverse=True)[:3]
        )
        chosen = random.choice(pool)

        if chosen["accept"]:
            self.consectuive_fallbacks = 0
        else:
            self.consectuive_fallbacks += 1

        # update bounds
        self.bounds = chosen["bounds"]
        self.depth += 1

        # high-res render
        xmin, xmax, ymin, ymax = chosen["bounds"]
        hires = self.hires_gen.generate(xmin, xmax, ymin, ymax)

        # save?
        if (
            chosen["accept"]
            and self.save_min_depth <= self.depth <= self.save_max_depth
        ):
            self._save_with_metadata(hires, chosen)

            # reset when reaching max depth
            if self.depth == self.save_max_depth:
                self.reset()

            return 1

        return 0

    def reset(self):
        self.bounds = self._default_bounds(self.fractal_type)
        self.depth = 0
        self.consectuive_fallbacks = 0

    # ---------------------------------------------------------------------------
    # Saving
    # ---------------------------------------------------------------------------

    def _save_with_metadata(self, hires, chosen):
        xmin, xmax, ymin, ymax = chosen["bounds"]
        base = f"d{self.depth:02d}_{xmin:.5f}_{ymin:.5f}"

        img = hires
        fname = self.output_dir / f"{base}_{self.colormap}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        meta = {
            "fractal_type": self.fractal_type,
            "depth": self.depth,
            "bounds": chosen["bounds"],
            "score": chosen["score"],
            "metrics": chosen.get("metrics", {}),
            "max_iter": self.max_iter,
            "width": hires.shape[1],
            "height": hires.shape[0],
            "colormap": self.colormap,
            "timestamp": datetime.now().isoformat(),
        }

        with open(fname.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ---------------------------------------------------------------------------
    # Tile search
    # ---------------------------------------------------------------------------

    def _tile_and_score(self, xmin, xmax, ymin, ymax):
        xs = np.linspace(xmin, xmax, self.n_tiles + 1)
        ys = np.linspace(ymin, ymax, self.n_tiles + 1)

        tiles = []

        for row in range(self.n_tiles):
            for col in range(self.n_tiles):
                bounds = (xs[col], xs[col + 1], ys[row], ys[row + 1])
                raw = self.tile_gen.generate_raw(*bounds)
                score, accept, metrics = self.evaluator.evaluate(raw)

                tiles.append(
                    {
                        "bounds": bounds,
                        "score": score,
                        "accept": accept,
                        "metrics": metrics,
                    }
                )

        return tiles

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _create_generator(
        self,
        fractal_type: str,
        width: int,
        height: int,
        max_iter: int,
        colormap: Optional[str],
    ) -> BaseFractalGenerator:
        if fractal_type == "mandelbrot":
            return MandelbrotGenerator(width, height, max_iter, colormap)
        elif fractal_type == "julia":
            return JuliaGenerator(width, height, max_iter, colormap)
        else:
            raise ValueError(f"Unknown fractal type: {fractal_type}")

    def _default_bounds(self, fractal_type):
        if fractal_type == "mandelbrot":
            return (-2.0, 1.0, -1.5, 1.5)
        if fractal_type == "julia":
            return (-2.0, 2.0, -2.0, 2.0)
        raise ValueError(fractal_type)

    def _count_existing_pngs(self):
        return len(list(self.output_dir.glob("*.png")))

    # ---------------------------------------------------------------------------
    # repr and str
    # ---------------------------------------------------------------------------

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"fractal_type={self.fractal_type!r}, "
            f"save_min_depth={self.save_min_depth}, "
            f"save_max_depth={self.save_max_depth}, "
            f"max_iter={self.max_iter}, "
            f"n_tiles={self.n_tiles}, "
            f"colormap={self.colormap}, "
            f"output_dir={str(self.output_dir)!r}"
            ")"
        )

    def __str__(self):
        rows = []
        rows.append(f"{self.__class__.__name__}")
        rows.append(f"fractal_type={self.fractal_type}, ")
        rows.append(f"depth={self.depth}, bounds={self.bounds}, ")
        rows.append(f"output_dir='{self.output_dir}'")
        return "\n".join(rows)
