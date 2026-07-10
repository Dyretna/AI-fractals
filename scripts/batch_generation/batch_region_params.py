# scripts/batch_generation/batch_region.py

from __future__ import annotations

import logging
from pathlib import Path

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data.region_builder import RegionBuilder
from ai_fractals.generators import create_generator
from ai_fractals.processing import EdgeDetector
from ai_fractals.search import create_search_strategy

log = logging.getLogger(__name__)


def run_region_batch(cfg: dict) -> None:
    """
    Pure region generation step.
    Produces only region metadata JSON files.
    """

    region_dir = Path(cfg["output_dir"]).resolve()
    region_dir.mkdir(parents=True, exist_ok=True)

    # Detector + evaluator
    detector = EdgeDetector(**cfg["detector"])
    evaluator = FractalQualityEvaluator(**cfg["evaluator"], detector=detector)

    # Tile-search generator (low-res)
    tile_gen = create_generator(
        fractal_type=cfg["fractal_type"],
        colormap="twilight_shifted",  # unused for raw iteration counts
        **cfg["tile_gen"],
    )

    # Search strategy
    search_strategy_cls = create_search_strategy(cfg["search_strategy"])
    tile_search = search_strategy_cls(
        tile_gen=tile_gen,
        evaluator=evaluator,
        **cfg["search_params"],
    )

    # Region builder
    region_builder = RegionBuilder(
        tile_search=tile_search,
        evaluator=evaluator,
        output_dir=region_dir,
        save_min_depth=cfg["min_depth"],
        save_max_depth=cfg["max_depth"],
    )

    log.info("\n" + str(region_builder) + "\n")
    region_builder.run(cfg["batch_size"])

    log.info(f"\nRegion batch completed. Saved metadata to: {region_dir}\n")
