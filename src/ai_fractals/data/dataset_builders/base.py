# ai_fractals/data/base_dataset_builder.py

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.generators import BaseFractalGenerator
from ai_fractals.logging_config import get_logger
from ai_fractals.search.tile_search import BaseTileSearch


class BaseDatasetBuilder(ABC):
    """
    Abstract dataset generator using tile-search.
    Subclasses implement:
        - _process_tile()
        - _save()
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
        colormap: str = "twilight",
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

        # device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # output
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # saver metadata flag
        self.save_metadata = save_metadata

        # logging
        self.log = get_logger(__name__, level=logging.WARNING)
        self._init_log()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def run(self, n_images: int):
        saved = self._count_existing_pngs()
        target = saved + n_images

        pbar = tqdm(total=n_images, desc=f"Generating {self.fractal_type}")

        while saved < target:
            saved_before = saved
            saved += self.step()

            if saved > saved_before:
                pbar.update(1)

            if self.consecutive_fallbacks >= self.max_fallbacks:
                self._reset_search()

        pbar.close()

    def step(self) -> int:
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
        processed, score, passed, metrics = self._process_tile(chosen)

        if not passed:
            return 0

        # save?
        if self.save_min_depth <= self.depth <= self.save_max_depth:
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

    def _count_existing_pngs(self):
        return len(list(self.output_dir.glob("*.png")))

    def _init_log(self):
        self.log.info("======== Dataset Builder =========")
        rows = [
            f"Device: {self.device}",
            f"Fractal type: {self.fractal_type}",
            f"Output dir: {self.output_dir}",
        ]
        self.log.info("\n - ".join(rows))

    def _reset_search(self):
        self.bounds = self.tile_search.tile_gen.default_bounds()
        self.depth = 0
        self.consecutive_fallbacks = 0

    def __str__(self):
        rows = [f"{self.__class__.__name__}:"]

        def indent(text: str, n: int = 4) -> str:
            pad = " " * n
            return "\n".join(pad + line for line in text.splitlines())

        def block(label, obj):
            rows.append(f"  {label}:")
            rows.append(indent(str(obj), 4))
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

        return "\n".join(rows)
