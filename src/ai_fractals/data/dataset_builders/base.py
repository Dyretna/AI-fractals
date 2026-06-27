# ai_fractals/data/base_dataset_builder.py

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.generators import BaseFractalGenerator
from ai_fractals.logging_config import get_logger
from ai_fractals.search.tile_search import BaseTileSearch


class BaseDatasetBuilder(ABC):
    """BaseDatasetBuilder

    Orchestrates fractal dataset generation using a tile-search exploration
    strategy. This class handles the recursive zooming process, depth control,
    fallback handling, and the main run/step loop. Subclasses implement the
    high resolution image processing and saving logic.

    Subclasses
    ----------
    RGBDatasetBuilder
        Produces full-resolution RGB fractal images. Uses the high-resolution
        generator together with a selected colormap to convert iteration counts
        into final RGB images. Saves PNGs (and optional metadata) for each
        accepted tile within the configured depth range.

    ShorelineDatasetBuilder
        Produces binary or grayscale shoreline/edge-map representations of the
        fractal. Instead of coloring the fractal, it extracts structural features
        such as edges, boundaries, or distance-based masks. Used for training
        AE or VAE models that operate on raw fractal geometry rather than
        colored images.

    Parameters
    ----------
    tile_search : BaseTileSearch
        The tile-search strategy responsible for exploring the fractal space.
        It generates low-resolution tiles, scores them, and selects the next
        region to zoom into.

    hires_generator : BaseFractalGenerator
        High-resolution fractal generator used to render the final saved images
        once a tile has been accepted and passes the quality evaluation.

    evaluator : FractalQualityEvaluator
        Evaluates raw tiles produced by the tile-search generator. Determines
        acceptance, scoring, and auxiliary metrics used during exploration.

    output_dir : Path
        Directory where generated images (and optional metadata) will be saved.

    save_min_depth : int, default=2
        Minimum zoom depth at which images are eligible to be saved. Depth is
        incremented each time a tile is accepted and exploration continues.

    save_max_depth : int, default=10
        Maximum zoom depth at which images are saved. When this depth is
        reached, the search is reset to the initial bounds.

    colormap : str, default="twilight_shifted"
        Name of the matplotlib colormap used when converting iteration counts
        to RGB images in subclasses.

    save_metadata : bool, default=False
        If True, store additional metadata (bounds, score, metrics, depth, etc.)
        alongside each saved image.

    Notes
    -----
    Subclasses must implement:
        - _process_tile(tile_result) -> (processed_image, score, passed, metrics)
        - _save(processed_image, tile_result, score, metrics)
    """

    def __init__(
        self,
        *,
        tile_search: BaseTileSearch,
        hires_generator: BaseFractalGenerator,
        evaluator: FractalQualityEvaluator,
        output_dir: Path,
        save_min_depth: int = 2,
        save_max_depth: int = 10,
        colormap: str = "twilight_shifted",
        save_metadata: bool = False,
    ):
        # tools
        self.tile_search = tile_search
        self.hires_generator = hires_generator
        self.evaluator = evaluator

        # fractal info
        self.fractal_type = hires_generator.fractal_type
        self.colormap = colormap

        # depth control
        self.save_min_depth = save_min_depth
        self.save_max_depth = save_max_depth

        # bounds + depth
        self.bounds = self.tile_search.tile_gen.default_bounds()
        self.depth = 0

        # fallback detection
        self.consecutive_fallbacks = 0
        self.max_fallbacks = 10

        # output
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # saver metadata flag
        self.save_metadata = save_metadata

        # logging
        self.log = get_logger(__name__, level=logging.WARNING)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def run(self, n_images: int):
        remaining = n_images
        produced = 0

        pbar = tqdm(total=n_images, desc=f"Generating {self.fractal_type}")

        while remaining > 0:
            produced = self.step()  # returns 1 if saved, else 0

            if produced:
                remaining -= 1
                pbar.update(1)

            if self.consecutive_fallbacks >= self.max_fallbacks:
                self._reset_search()

        pbar.close()

    def step(self) -> int:
        self.tile_search.depth = self.depth
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

        # let subclass process tile (RGB or shoreline)
        # only if we are in accepted depth
        if self.save_min_depth <= self.depth <= self.save_max_depth:
            processed, score, passed, metrics = self._process_tile(chosen)

            if not passed:
                return 0

            # save?
            self._save(processed, chosen, score, metrics)

            if self.depth == self.save_max_depth:
                self._reset_search()

            return 1

        return 0

    # ----------------------------------------------------------------------
    # Abstract hooks
    # ----------------------------------------------------------------------

    @abstractmethod
    def _process_tile(self, chosen: dict):
        """
        Must return:
            processed_image, score, passed, metrics
        """
        ...

    @abstractmethod
    def _save(self, chosen: dict):
        """Saves the image and optional metadata"""
        ...

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _reset_search(self):
        self.bounds = self.tile_search.tile_gen.default_bounds()
        self.depth = 0
        self.consecutive_fallbacks = 0

    def __str__(self):
        header = "\n" + "=" * 50 + f"\n{self.__class__.__name__}\n" + "=" * 50
        rows = [header]

        def indent(text: str) -> str:
            pad = " " * 4
            return "\n".join(pad + line for line in text.splitlines())

        def block(label, obj):
            rows.append(f"  {label}:")
            rows.append(indent(str(obj)))
            rows.append("")

        block("tile_search", self.tile_search)
        block("evaluator", self.evaluator)
        block("edge_detector", self.evaluator.detector)
        block("hires_gen", self.hires_generator)

        rows.append(
            f"  final_res: {self.hires_generator.width}, {self.hires_generator.height}"
        )
        rows.append(f"  fractal_type: {self.fractal_type}")
        rows.append(f"  max_iter: {self.hires_generator.max_iter}")
        rows.append(f"  colormap: {self.colormap}")
        rows.append(f"  output_dir: {self.output_dir}")
        rows.append("=" * 50)

        return "\n".join(rows)
