"""
ShorelineDatasetBuilder
-----------------------

This module generates a dataset of shoreline images for training a
self-supervised CNN that learns geometric representations of fractals.

A shoreline is a grayscale edge map extracted directly from the raw iteration
data. It captures only the mathematical boundary structure of the fractal,
without color, shading, or rendering artifacts. This makes shorelines ideal
for learning shape-based embeddings.

Why this dataset exists:
- A CNN is trained on shoreline images to learn geometry-invariant features.
- The CNN outputs an embedding vector for each shoreline.
- Similar fractal structures produce similar embeddings.
- These embeddings are later clustered (e.g., K-Means) to form automatic
  fractal categories such as “spiral”, “bulb”, “seahorse valley”, etc.
- The resulting labels or embeddings are used to condition generative models
  (GANs, VAEs) so they can produce specific fractal types or explore the
  latent space between them.

Why we generate shorelines directly:
- Raw RGB renders are unnecessary for geometry learning.
- Shorelines are lightweight, fast to compute, and contain only the essential
  structural information.
- Evaluator filtering removes uninformative tiles (fully inside/outside,
  low edge density, low entropy).
- Minimal augmentation (flips) increases dataset
  diversity without distorting the underlying fractal geometry.

The builder performs:
1. Tile-search to explore the fractal space and locate interesting regions.
2. Raw fractal tile generation at a configurable resolution.
3. Shoreline extraction using edge detection.
4. Quality evaluation
5. Augmentation
6. Saving via ShoreLineSaver

The output is a clean, geometry-focused dataset suitable for training
self-supervised CNNs that produce robust fractal embeddings.
"""
# src/ai_fractals/data/shoreline_dataset_builder.py

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data.savers import ShorelineSaver
from ai_fractals.data.tile_search import TileSearch
from ai_fractals.generators import (
    BaseFractalGenerator,
    create_fractal_state,
    create_generator,
    get_default_bounds,
)
from ai_fractals.logging_config import get_logger
from ai_fractals.processing import EdgeDetector, ImageAugmenter


class ShorelineDatasetBuilder:
    """
    Automated shoreline dataset generator using tile-search strategy.
    """

    def __init__(
        self,
        fractal_type="mandelbrot",
        save_min_depth=2,
        save_max_depth=10,
        max_iter=256,
        tile_resolution=256,
        shoreline_resolution=512,
        quality_threshold=0.1,
        n_tiles=5,
        colormap="twilight_shifted",
        output_dir: Path | None = None,
        detector: EdgeDetector | None = None,
        evaluator: FractalQualityEvaluator | None = None,
        augmenter: ImageAugmenter | None = None,
        log_level: int = logging.WARNING,
        save_metadata: bool = False,
    ):
        self.fractal_type = fractal_type
        self.tile_resolution = tile_resolution
        self.shoreline_resolution = shoreline_resolution
        self.max_iter = max_iter
        self.n_tiles = n_tiles
        self.save_min_depth = save_min_depth
        self.save_max_depth = save_max_depth
        self.colormap = colormap

        # fractal state
        self.state = create_fractal_state(fractal_type)

        # tools
        self.detector = detector or EdgeDetector()
        self.evaluator = evaluator or FractalQualityEvaluator(quality_threshold)
        self.augmenter = augmenter or ImageAugmenter(
            horizontal_flip=True,
            vertical_flip=True,
            target_size=(256, 256),
        )

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

        # high-res generator (final shoreline)
        self.hires_gen: BaseFractalGenerator = create_generator(
            fractal_type=self.fractal_type,
            width=self.shoreline_resolution,
            height=self.shoreline_resolution,
            max_iter=max_iter,
            colormap=self.colormap,
            log_level=log_level,
            use_supersampling=True,
            state=self.state,
        )

        # Bounds + depth
        self.bounds = get_default_bounds(fractal_type, self.state)
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
                / "dataset"
                / "shorelines"
                / fractal_type
                / f"{augmenter.target_size[0]}_{augmenter.target_size[1]}_iter{max_iter}"
            ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # saver
        self.saver = ShorelineSaver(
            output_dir=self.output_dir,
            fractal_type=self.fractal_type,
            colormap=self.colormap,
            max_iter=self.max_iter,
        )

        # logging
        self.log = get_logger(__name__, level=log_level)
        self._init_log()

        self.save_metadata = save_metadata

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def run(self, n_images):
        saved = self._count_existing_pngs()
        target = saved + n_images

        pbar = tqdm(total=n_images, desc="Generating shorelines")

        while saved < target:
            saved_before = saved
            saved += self.step()

            if saved > saved_before:
                pbar.update(1)

            if self.consecutive_fallbacks >= self.max_fallbacks:
                self._reset_search()

        pbar.close()

    def step(self):
        # tile-search (same as RGB builder)
        chosen = self.tile_search.run(self.bounds)

        if chosen["accept"]:
            self.consecutive_fallbacks = 0
        else:
            self.consecutive_fallbacks += 1

        if self.consecutive_fallbacks >= self.max_fallbacks:
            self._reset_search()
            return 0

        # update bounds + depth
        self.bounds = chosen["bounds"]
        self.depth += 1

        # high-res raw iteration
        xmin, xmax, ymin, ymax = chosen["bounds"]
        hires_raw = self.hires_gen.generate_raw(xmin, xmax, ymin, ymax)

        # shoreline extraction
        shoreline = self.detector.detect(hires_raw)

        # evaluate shoreline
        score, passed, metrics = self.evaluator.evaluate(shoreline)
        if not passed:
            return 0

        # save?
        if self.save_min_depth <= self.depth <= self.save_max_depth:
            self._save_augmented(shoreline, chosen, score, metrics)

            if self.depth == self.save_max_depth:
                self._reset_search()

            return 1

        return 0

    # ----------------------------------------------------------------------
    # Saving
    # ----------------------------------------------------------------------

    def _save_augmented(self, shoreline, chosen, score, metrics):
        ts = datetime.now().isoformat()
        cid = datetime.fromisoformat(ts).strftime("%y%m%d%H%M%S")

        variants = self.augmenter.augment(shoreline)

        for tag, aug in variants.items():
            out = (aug * 255).astype(np.uint8)

            root = f"{cid}_{self.colormap}_iter{self.max_iter}"
            name = self.output_dir / f"{root}_d{self.depth:02d}_{tag}"

            self.saver.save_img(out, name)

            if self.save_metadata:
                self.saver.save_metadata(
                    name=name,
                    resolution=f"({self.shoreline_resolution}, {self.shoreline_resolution})",
                    bounds=chosen["bounds"],
                    score=score,
                    metrics=metrics,
                    ts=ts,
                    cid=cid,
                )

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------

    def _reset_search(self):
        self.bounds = get_default_bounds(self.fractal_type, self.state)
        self.depth = 0
        self.consecutive_fallbacks = 0

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _count_existing_pngs(self):
        return len(list(self.output_dir.glob("*.png")))

    def _init_log(self):
        self.log.info("======== Shoreline Builder =========")
        rows = []
        rows.append(f"Device: {self.device}")
        rows.append(f"Colormap: {self.colormap}")
        rows.append(f"Output dir: {self.output_dir}")

        self.log.info("\n - ".join(rows))

    def __str__(self):
        rows = [
            f"{self.__class__.__name__}",
            f"  fractal_type:   {self.fractal_type}",
            f"  tile_res:       {self.tile_resolution}",
            f"  shoreline_res:  {self.shoreline_resolution}",
            f"  final_aug_res:  {self.augmenter.target_size}",
            f"  max_iter:       {self.max_iter}",
            f"  colormap:       {self.colormap}",
            f"  output_dir:     {self.output_dir}",
        ]
        return "\n".join(rows)
