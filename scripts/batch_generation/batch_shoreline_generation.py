"""
/scripts/batch_shoreline_generation.py

Batch shoreline generation using ShorelineDatasetBuilder.
Creates shorelines from scratch, without loading any images from disk.

This script is the composition root: it constructs generators, tile-search
strategies and evaluators, then composes a ShorelineDatasetBuilder for each
(fractal_type, colormap) combination and runs the batch job.

"""

from __future__ import annotations

from pathlib import Path

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data import ShorelineDatasetBuilder
from ai_fractals.generators import BaseFractalGenerator, create_generator
from ai_fractals.processing import EdgeDetector
from ai_fractals.search import BaseTileSearch, create_search_strategy


def run_shoreline_batch(cfg: dict) -> None:
    output_dir = Path(cfg["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = EdgeDetector(**cfg["detector"])
    evaluator = FractalQualityEvaluator(**cfg["evaluator"], detector=detector)

    # low-res generator for tile-search
    tile_gen: BaseFractalGenerator = create_generator(
        fractal_type=cfg["fractal_type"],
        colormap="twilight_shifted",
        **cfg["tile_gen"],
    )

    search_strategy_cls = create_search_strategy(cfg["search_strategy"])
    tile_search: BaseTileSearch = search_strategy_cls(
        tile_gen=tile_gen, evaluator=evaluator, **cfg["search_params"]
    )

    # high-res generator for final shoreline extraction
    hires_generator: BaseFractalGenerator = create_generator(
        fractal_type=cfg["fractal_type"],
        colormap="twilight_shifted",
        **cfg["hires_gen"],
    )

    builder = ShorelineDatasetBuilder(
        tile_search=tile_search,
        hires_generator=hires_generator,
        evaluator=evaluator,
        output_dir=output_dir,
        save_min_depth=cfg["min_depth"],
        save_max_depth=cfg["max_depth"],
        colormap="twilight_shifted",
        save_metadata=False,
    )
    print("\n", builder, "\n")
    builder.run(cfg["batch_size"])
