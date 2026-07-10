# /scripts/batch_generation/batch_shoreline_from_meta.py
"""
Batch shoreline rendering from region metadata.

This script loads precomputed region metadata files (JSON with fractal bounds)
and renders high-resolution shoreline images for each region. A quality
evaluation is performed for every raw fractal render, and only regions that
pass the evaluator are saved.

The purpose of this batch is structural representation: extract clean
grayscale contour maps ("shorelines") suitable for downstream ML training.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.generators import create_generator
from ai_fractals.processing import EdgeDetector

log = logging.getLogger(__name__)


def pretty_cfg(cfg: dict) -> str:
    lines = ["\n=== Config " + "=" * 70]
    for key, val in cfg.items():
        if isinstance(val, dict):
            lines.append(f"{key}:")
            for k2, v2 in val.items():
                lines.append(f"    {k2:18}: {v2}")
        else:
            lines.append(f"{key:22}: {val}")
    lines.append("=" * 80 + "\n")
    return "\n".join(lines)


def run_shoreline_batch(cfg: dict) -> None:
    # explicit paths
    region_raw = Path(cfg["region_raw_dir"]).resolve()
    region_eval = Path(cfg["region_evaluated_dir"]).resolve()
    region_reject = Path(cfg["region_rejected_dir"]).resolve()

    shore_eval = Path(cfg["shoreline_evaluated_dir"]).resolve()
    shore_reject = Path(cfg["shoreline_rejected_dir"]).resolve()

    region_eval.mkdir(parents=True, exist_ok=True)
    region_reject.mkdir(parents=True, exist_ok=True)
    shore_eval.mkdir(parents=True, exist_ok=True)
    shore_reject.mkdir(parents=True, exist_ok=True)

    detector = EdgeDetector(**cfg["detector"])
    evaluator = FractalQualityEvaluator(**cfg["evaluator"], detector=detector)

    hires_gen = create_generator(
        fractal_type=cfg["fractal_type"],
        colormap="twilight",
        **cfg["hires_gen"],
    )

    log.info("\n" + pretty_cfg(cfg) + "\n")

    for region_path in tqdm(
        sorted(region_raw.glob("*.json")),
        desc="Processing regions",
        ncols=80,
    ):
        meta = json.load(open(region_path))
        xmin, xmax, ymin, ymax = meta["bounds"]

        raw = hires_gen.generate_raw(xmin, xmax, ymin, ymax)
        score, passed, metrics = evaluator.evaluate(raw)

        shoreline = detector.detect(raw).astype(np.uint8)
        img = Image.fromarray(shoreline, mode="L")

        cid = meta["compact_id"]

        shoreline_name = (
            f"{cid}_iter{cfg['hires_gen']['max_iter']}_d{meta['depth']}.png"
        )

        if passed:
            img.save(shore_eval / shoreline_name)
            shutil.move(region_path, region_eval / region_path.name)
            log.info(f"[OK] {cid} score={score:.3f}")
        else:
            img.save(shore_reject / shoreline_name)
            shutil.move(region_path, region_reject / region_path.name)
            log.info(f"[REJECT] {cid} score={score:.3f}")

    log.info("Shoreline Batch completed.")
