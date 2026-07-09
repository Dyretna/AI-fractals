#!/usr/bin/env python3
"""
Simplified RGB Batch Rendering

This version is designed for experimentation:
- Uses a single output directory from config.
- Reads fractal_type from each region JSON.
- Renders all regions with the given colormap(s).
- No automatic folder structure.
- No fractal-type grouping.
- No missing-file detection.
"""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from ai_fractals.generators import BaseFractalGenerator, create_generator

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


def run_rgb_batch(cfg: dict) -> None:
    """
    Render RGB fractal images for all regions in region_evaluated_dir.

    Parameters
    ----------
    cfg : dict
        Must contain:
            region_evaluated_dir : str
            output_dir : str
            colormaps : list[str]
            hires_gen : dict with width, height, max_iter, device, etc.
    """

    region_dir = Path(cfg["region_evaluated_dir"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_iter = cfg["hires_gen"]["max_iter"]
    cmap = cfg["colormap"]
    fractal_type = cfg["fractal_type"]

    log.info(pretty_cfg(cfg), "\n")

    # -----------------------------------------------------------
    # Load existing CIDs (missing-file detection)
    # -----------------------------------------------------------
    existing_cids = {img.stem.split("_")[0] for img in output_dir.glob("*.png")}

    # -----------------------------------------------------------
    # Iterate over ALL region JSON files
    # -----------------------------------------------------------
    region_paths: List[Path] = sorted(region_dir.glob("*.json"))

    existing_count = len(existing_cids)
    total_regions = len(region_paths)
    missing_count = total_regions - existing_count

    log.info(f"Found {total_regions} region metadata files.")
    log.info(f"Already rendered: {existing_count}")
    log.info(f"Missing:          {missing_count}")
    log.info(
        f"Progress:         {existing_count}/{total_regions} "
        f"({existing_count / total_regions * 100:.2f}%)\n"
    )

    for region_path in tqdm(region_paths, desc="regions", ncols=80):
        meta = json.load(open(region_path))

        cid = meta["compact_id"]
        depth = meta["depth"]
        xmin, xmax, ymin, ymax = meta["bounds"]

        # skip if already rendered
        if cid in existing_cids:
            continue

        # -------------------------------------------------------
        # Render
        # -------------------------------------------------------

        hires_rgb: BaseFractalGenerator = create_generator(
            fractal_type=fractal_type,
            colormap=cmap,
            **cfg["hires_gen"],
        )

        rgb = hires_rgb.generate(xmin, xmax, ymin, ymax)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")

        out_path = output_dir / f"{cid}_{cmap}_iter{max_iter}_d{depth}.png"
        img.save(out_path)

    log.info("\nRGB batch completed.\n")
