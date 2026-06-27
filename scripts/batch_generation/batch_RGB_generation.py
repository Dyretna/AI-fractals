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


def run_rgb_batch(cfg: dict, project_root: Path) -> None:
    output_root = project_root / "dataset" / "rgb"
    output_root.mkdir(parents=True, exist_ok=True)

    detector = EdgeDetector(**cfg["detector"])
    evaluator = FractalQualityEvaluator(**cfg["evaluator"], detector=detector)

    num_types = len(cfg["fractal_types"])
    num_cmaps = len(cfg["colormaps"])

    total_combinations = num_types * num_cmaps
    per_cfg_loop = cfg["batch_img_gen"] // total_combinations

    completed_cmaps = []
    remaining_cmaps = list(cfg["colormaps"])

    for fractal_type in cfg["fractal_types"]:
        for colormap in cfg["colormaps"]:
            # low-res generator for tile-search
            tile_gen: BaseFractalGenerator = create_generator(
                fractal_type=fractal_type,
                colormap=colormap,
                **cfg["tile_gen"],
            )
            search_strategy_cls = create_search_strategy(cfg["search_strategy"])
            tile_search: BaseTileSearch = search_strategy_cls(
                tile_gen=tile_gen, evaluator=evaluator, **cfg["search_params"]
            )

            # high-res generator for final RGB render
            hires_generator: BaseFractalGenerator = create_generator(
                fractal_type=fractal_type,
                colormap=colormap,
                **cfg["hires_gen"],
            )

            width = cfg["hires_gen"].get("width")
            height = cfg["hires_gen"].get("height")
            out_dir = Path(
                output_root
                / fractal_type
                / f"{width}_{height}_iter{cfg['hires_gen']['max_iter']}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            builder = RGBDatasetBuilder(
                tile_search=tile_search,
                hires_generator=hires_generator,
                evaluator=evaluator,
                output_dir=out_dir,
                save_min_depth=cfg["min_depth"],
                save_max_depth=cfg["max_depth"],
                colormap=colormap,
                save_metadata=False,
            )

            # if first loop - print builder object
            if not completed_cmaps:
                print("\n", builder, "\n")

            builder.run(per_cfg_loop)

            # progress prints
            print(f"[{fractal_type}] Completed: {completed_cmaps}")
            print(f"[{fractal_type}] Remaining: {remaining_cmaps}")
            print(f"Completed batch for {fractal_type}\n")
            completed_cmaps.append(colormap)
            remaining_cmaps.remove(colormap)
            total_combinations -= 1
            print(f"Batches to go: {total_combinations}")


if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))
    yaml_path = project_root / "configs" / "rgb_batch.yaml"

    cfg = yaml.safe_load(open(yaml_path))

    run_rgb_batch(cfg, project_root)
