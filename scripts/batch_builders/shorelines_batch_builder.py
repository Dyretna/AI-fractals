#!/usr/bin/env python3

"""
Batch shoreline generation using ShorelineDatasetBuilder.
Creates shorelines from scratch, without loading any images from disk.

This script initializes the required tools (detector, evaluator, augmenter)
and injects them into the builder. All logic for tile-search, shoreline
extraction, evaluation, augmentation, and saving is handled inside the builder.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data import CURATED_COLORMAPS, ShorelineDatasetBuilder
from ai_fractals.processing import EdgeDetector, ImageAugmenter


def run_shoreline_batch(
    detector: EdgeDetector,
    evaluator: FractalQualityEvaluator,
    augmenter: ImageAugmenter,
    colormap: str,
    n: int,
    output_dir: Path,
):
    # Create builder (mirrors FractalDatasetBuilder usage)
    builder = ShorelineDatasetBuilder(
        fractal_type="mandelbrot",
        tile_resolution=256,
        shoreline_resolution=512,  # resolution of highres shoreline before augmentation
        max_iter=256,
        n_tiles=5,
        save_min_depth=2,
        save_max_depth=10,
        colormap=colormap,
        detector=detector,
        evaluator=evaluator,
        augmenter=augmenter,
        output_dir=output_dir,
    )

    print("\n", builder, "\n")
    builder.run(n)
    print("Shoreline dataset generation complete.")


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

    # Output directory
    SHORELINES_ROOT = PROJECT_ROOT / "dataset" / "shorelines"
    OUTPUT_DIR = SHORELINES_ROOT / "mandelbrot" / "256_256_iter256"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Injected tools
    detector = EdgeDetector(
        canny_low=40,
        canny_high=120,
        apply_smoothing=True,
        smoothing_method="gaussian",
        smoothing_kernel=3,
        smoothing_sigma=0.8,
    )

    evaluator = FractalQualityEvaluator(
        quality_threshold=0.3,
        min_edge_ratio=0.03,
        max_edge_ratio=0.45,
        min_inside_ratio=0.0001,
        max_inside_ratio=0.9999,
    )

    augmenter = ImageAugmenter(
        horizontal_flip=True,
        vertical_flip=True,
        target_size=(256, 256),
    )

    TOTAL_FINAL = 10_000
    AUG_PER_IMAGE = 4

    TOTAL_RAW = TOTAL_FINAL // AUG_PER_IMAGE
    shorelines_per_cmap = TOTAL_RAW // len(CURATED_COLORMAPS)

    for cmap in CURATED_COLORMAPS:
        run_shoreline_batch(
            detector,
            evaluator,
            augmenter,
            cmap,
            shorelines_per_cmap,
            output_dir=OUTPUT_DIR,
        )
