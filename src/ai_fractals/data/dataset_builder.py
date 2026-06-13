import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.generators import (
    BaseFractalGenerator,
    create_fractal_state,
    create_generator,
    get_default_bounds,
)
from ai_fractals.logging_config import get_logger


class FractalDatasetBuilder:
    """
    Automated batch fractal dataset generator using tile-search refinement.
    """

    def __init__(
        self,
        fractal_type="mandelbrot",
        save_min_depth: int = 2,
        save_max_depth: int = 10,
        width: int = 1024,
        height: int = 1024,
        max_iter: int = 1024,
        tile_resolution: int = 200,
        quality_threshold: float = 0.1,
        n_tiles: int = 5,
        colormap: str = "twilight_shifted",
        log_level: int = logging.WARNING,
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

        # set type and its state (c for julia, etc)
        self.fractal_type = fractal_type
        self.state = create_fractal_state(self.fractal_type)

        self.save_min_depth = save_min_depth
        self.save_max_depth = save_max_depth
        self.max_iter = max_iter
        self.n_tiles = n_tiles
        self.colormap = colormap

        # Generators and evaluator
        self.tile_gen: BaseFractalGenerator = create_generator(
            fractal_type=self.fractal_type,
            width=tile_resolution,
            height=tile_resolution,
            max_iter=max_iter,
            colormap=colormap,
            log_level=log_level,
            use_supersampling=False,
            state=self.state,
        )
        self.hires_gen: BaseFractalGenerator = create_generator(
            fractal_type=self.fractal_type,
            width=width,
            height=height,
            max_iter=max_iter,
            colormap=self.colormap,
            log_level=log_level,
            use_supersampling=True,
            state=self.state,
        )

        self.evaluator = FractalQualityEvaluator(quality_threshold)

        # Bounds
        self.bounds = get_default_bounds(self.fractal_type, self.state)
        self.depth = 0

        # stuck detection
        self.consecutive_fallbacks = 0
        self.max_fallbacks = 10

        # pytorch device and CUDA
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if self.has_cuda else torch.device("cpu")

        # logging
        self.log = get_logger(__name__, level=log_level)
        self._init_log()

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

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

            if self.consecutive_fallbacks >= self.max_fallbacks:
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
        # lower thres at starting point, for more variety
        self.evaluator.quality_threshold = 0.12 if self.depth == 0 else 0.3

        # create tiled candidates for further depth search
        # Random choice, since best metrics is not necessarily most interesting
        # Fallback if no tile is over threshold: pick 1 random of top 3
        tiles = self._tile_and_score(*self.bounds)
        candidates = [t for t in tiles if t["accept"]]
        pool = (
            candidates
            if candidates
            else sorted(tiles, key=lambda t: t["score"], reverse=True)[:3]
        )
        chosen = random.choice(pool)

        if chosen["accept"]:
            self.consecutive_fallbacks = 0
        else:
            self.consecutive_fallbacks += 1

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
        self.bounds = get_default_bounds(self.fractal_type, self.state)
        self.depth = 0
        self.consecutive_fallbacks = 0

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

    # --------------------------------------------------------------------------
    # Tile search
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _init_log(self):
        self.log.info("======== Dataset Builder =========")
        rows = []
        rows.append(f"Device: {self.device}")
        rows.append(f"Colormap: {self.colormap}")
        rows.append(f"Output dir: {self.output_dir}")

        self.log.info("\n - ".join(rows))

    def _count_existing_pngs(self):
        return len(list(self.output_dir.glob("*.png")))

    # --------------------------------------------------------------------------
    # repr and str
    # --------------------------------------------------------------------------

    def __repr__(self):
        rows = []
        rows.append(f"{self.__class__.__name__}")
        rows.append(f"Device: {self.device}")
        rows.append(f"fractal_type={self.fractal_type}")
        rows.append(f"save_min_depth={self.save_min_depth}")
        rows.append(f"save_max_depth={self.save_max_depth}")
        rows.append(f"max_iter={self.max_iter}")
        rows.append(f"n_tiles={self.n_tiles}")
        rows.append(f"colormap={self.colormap}")
        rows.append(f"output_dir='{self.output_dir}'")
        return "\n  ".join(rows)

    def __str__(self):
        rows = []
        rows.append(f"{self.__class__.__name__}")
        rows.append(f"Device:       {self.device}")
        rows.append(f"fractal_type: {self.fractal_type}")
        rows.append(f"colormap:     {self.colormap}")
        rows.append(f"output_dir:   {self.output_dir}")
        return "\n  ".join(rows)
