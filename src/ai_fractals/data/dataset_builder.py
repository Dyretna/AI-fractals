import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data.savers import RGBSaver
from ai_fractals.data.tile_search import TileSearch
from ai_fractals.generators import (
    BaseFractalGenerator,
    create_fractal_state,
    create_generator,
    get_default_bounds,
)
from ai_fractals.logging_config import get_logger


class FractalDatasetBuilder:
    """
    Automated batch fractal dataset generator using tile-search strategy.
    """

    def __init__(
        self,
        fractal_type="mandelbrot",
        save_min_depth: int = 2,
        save_max_depth: int = 10,
        width: int = 1024,
        height: int = 1024,
        max_iter: int = 1024,
        tile_resolution: int = 256,
        quality_threshold: float = 0.1,
        n_tiles: int = 5,
        colormap: str = "twilight_shifted",
        log_level: int = logging.WARNING,
        output_dir: Path = None,
    ):
        # set type and its state (c for julia, etc)
        self.fractal_type = fractal_type
        self.state = create_fractal_state(self.fractal_type)

        # rendering config
        self.width = width
        self.height = height
        self.save_min_depth = save_min_depth
        self.save_max_depth = save_max_depth
        self.max_iter = max_iter
        self.n_tiles = n_tiles
        self.colormap = colormap

        # evaluator
        self.evaluator = FractalQualityEvaluator(quality_threshold)

        # tile-search generator
        self.tile_search = TileSearch(
            tile_gen=create_generator(
                fractal_type=self.fractal_type,
                width=tile_resolution,
                height=tile_resolution,
                max_iter=max_iter,
                colormap=colormap,
                log_level=log_level,
                use_supersampling=False,
                state=self.state,
            ),
            evaluator=self.evaluator,
            n_tiles=self.n_tiles,
            top_k=3,
        )

        # high-res generator
        self.hires_gen: BaseFractalGenerator = create_generator(
            fractal_type=self.fractal_type,
            width=self.width,
            height=self.height,
            max_iter=max_iter,
            colormap=self.colormap,
            log_level=log_level,
            use_supersampling=True,
            state=self.state,
        )

        # Bounds + depth
        self.bounds = get_default_bounds(self.fractal_type, self.state)
        self.depth = 0

        # fallback detection
        self.consecutive_fallbacks = 0
        self.max_fallbacks = 10

        # pytorch device and CUDA
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if self.has_cuda else torch.device("cpu")

        # output directory
        load_dotenv()
        project_root = Path(os.getenv("PROJECT_ROOT"))
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = Path(
                project_root
                / "fractals"
                / "dataset"
                / fractal_type
                / f"{width}_{height}_iter{max_iter}"
            ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.saver = RGBSaver(
            output_dir=self.output_dir,
            fractal_type=self.fractal_type,
            colormap=self.colormap,
            max_iter=self.max_iter,
        )

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

        # tile-search via strategy
        chosen = self.tile_search.run(self.bounds)

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
            self._save(hires, chosen)

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

    def _save(self, img, chosen):
        # timestamp + compact ID
        ts = datetime.now().isoformat()
        cid = datetime.fromisoformat(ts).strftime("%y%m%d%H%M%S")

        # build filename root
        root = f"{cid}_{self.colormap}_iter{self.max_iter}"
        name = self.output_dir / f"{root}_d{self.depth:02d}"

        # save image
        self.saver.save_img(img, name)

        # save metadata
        self.saver.save_metadata(
            name=name,
            resolution=f"({self.width}, {self.height})",
            bounds=chosen["bounds"],
            score=chosen["score"],
            metrics=chosen.get("metrics", {}),
            ts=ts,
            cid=cid,
        )

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
        rows = [
            f"{self.__class__.__name__}",
            f"Device={self.device}",
            f"fractal_type={self.fractal_type}",
            f"save_min_depth={self.save_min_depth}",
            f"save_max_depth={self.save_max_depth}",
            f"max_iter={self.max_iter}",
            f"n_tiles={self.n_tiles}",
            f"colormap={self.colormap}",
            f"output_dir='{self.output_dir}'",
        ]
        return "\n".join(rows)

    def __str__(self):
        rows = [
            f"{self.__class__.__name__}",
            f"Device:       {self.device}",
            f"fractal_type: {self.fractal_type}",
            f"resolution:   {self.width}_{self.height}",
            f"max_iter:     {self.max_iter}",
            f"colormap:     {self.colormap}",
            f"output_dir:   {self.output_dir}",
        ]
        return "\n".join(rows)
