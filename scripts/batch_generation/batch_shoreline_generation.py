"""
/scripts/batch_shoreline_generation.py

Batch shoreline generation using ShorelineDatasetBuilder.
Creates shorelines from scratch, without loading any images from disk.

This script is the composition root: it constructs generators, tile-search
strategies and evaluators, then composes a ShorelineDatasetBuilder for each
(fractal_type, colormap) combination and runs the batch job.

"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data import ShorelineDatasetBuilder
from ai_fractals.generators import BaseFractalGenerator, create_generator
from ai_fractals.processing import EdgeDetector
from ai_fractals.search import BaseTileSearch, create_search_strategy


def run_shoreline_batch(cfg: dict, project_root: Path) -> None:
    output_root = project_root / "dataset" / "shorelines"
    output_root.mkdir(parents=True, exist_ok=True)

    detector = EdgeDetector(**cfg["detector"])
    evaluator = FractalQualityEvaluator(**cfg["evaluator"], detector=detector)

    for fractal_type in cfg["fractal_types"]:
        per_fractal_type = cfg["batch_img_gen"] // len(cfg["fractal_types"])

        # low-res generator for tile-search
        tile_gen: BaseFractalGenerator = create_generator(
            fractal_type=fractal_type,
            colormap="twilight_shifted",
            **cfg["tile_gen"],
        )

        search_strategy_cls = create_search_strategy(cfg["search_strategy"])
        tile_search: BaseTileSearch = search_strategy_cls(
            tile_gen=tile_gen, evaluator=evaluator, **cfg["search_params"]
        )

        # high-res generator for final shoreline extraction
        hires_generator: BaseFractalGenerator = create_generator(
            fractal_type=fractal_type,
            colormap="twilight_shifted",
            **cfg["hires_gen"],
        )

        out_dir = Path(output_root / fractal_type / "512_512_iter256")
        out_dir.mkdir(parents=True, exist_ok=True)

        builder = ShorelineDatasetBuilder(
            tile_search=tile_search,
            hires_generator=hires_generator,
            evaluator=evaluator,
            output_dir=out_dir,
            save_min_depth=cfg["min_depth"],
            save_max_depth=cfg["max_depth"],
            colormap=cfg["colormap"],
            save_metadata=False,
        )

        print("\n", builder, "\n")
        builder.run(per_fractal_type)
        print(f"Completed batch for {fractal_type}\n")


if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))
    yaml_path = project_root / "configs" / "shoreline_jittered_batch.yaml"

    cfg = yaml.safe_load(open(yaml_path))

    run_shoreline_batch(cfg, project_root)
