"""
/scripts/batch_generation/batch_rgb_generation.py

Batch RGB fractal generation using RGBDatasetBuilder.
Creates high-resolution RGB fractal renders from scratch.

This script is the composition root: it constructs generators,
tile-search strategies and evaluators, then composes an
RGBDatasetBuilder for each fractal_type and runs the batch job.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data import RGBDatasetBuilder
from ai_fractals.generators import BaseFractalGenerator, create_generator
from ai_fractals.processing import EdgeDetector
from ai_fractals.search import BaseTileSearch, create_search_strategy


def run_rgb_batch(cfg: dict) -> None:
    output_dir = Path(cfg["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = EdgeDetector(**cfg["detector"])
    evaluator = FractalQualityEvaluator(**cfg["evaluator"], detector=detector)

    total_batches = len(cfg["colormaps"])
    per_cfg_loop = cfg["batch_size"] // total_batches
    completed_cmaps = []
    remaining_cmaps = list(cfg["colormaps"])

    for colormap in cfg["colormaps"]:
        # low-res generator for tile-search
        tile_gen: BaseFractalGenerator = create_generator(
            fractal_type=cfg["fractal_type"],
            colormap=colormap,
            **cfg["tile_gen"],
        )
        search_strategy_cls = create_search_strategy(cfg["search_strategy"])
        tile_search: BaseTileSearch = search_strategy_cls(
            tile_gen=tile_gen, evaluator=evaluator, **cfg["search_params"]
        )

        # high-res generator for final RGB render
        hires_generator: BaseFractalGenerator = create_generator(
            fractal_type=cfg["fractal_type"],
            colormap=colormap,
            **cfg["hires_gen"],
        )

        builder = RGBDatasetBuilder(
            tile_search=tile_search,
            hires_generator=hires_generator,
            evaluator=evaluator,
            output_dir=output_dir,
            save_min_depth=cfg["min_depth"],
            save_max_depth=cfg["max_depth"],
            colormap=colormap,
            save_metadata=False,
        )

        # if first loop - print builder object
        if not completed_cmaps:
            print("\n", builder, "\n")

        print(f"Current cmap: {colormap}\n")
        builder.run(per_cfg_loop)

        # progress prints
        completed_cmaps.append(colormap)
        remaining_cmaps.remove(colormap)
        total_batches -= 1

        print(f"Completed batch for cmap {colormap}\n")
        print(f"Completed: {completed_cmaps}")
        print(f"Remaining: {remaining_cmaps}")
        print(f"Batches to go: {total_batches}")


if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))
    yaml_path = project_root / "configs" / "rgb_batch.yaml"

    cfg = yaml.safe_load(open(yaml_path))

    # set output
    output_dir = Path(project_root / cfg["output_dir"])
    cfg["output_path"] = Path(project_root / cfg["output_dir"]).resolve()

    run_rgb_batch(cfg)
