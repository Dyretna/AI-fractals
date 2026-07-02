#!/usr/bin/env python3
"""
RGB Batch Rendering from Region Metadata


This module renders high-resolution RGB fractal images based on
precomputed region metadata files. Each region JSON contains the
fractal bounds and the fractal type. The pipeline supports multiple
fractal types automatically.

The workflow:

1. Regions are grouped by fractal type.
2. For each fractal type, the module checks which RGB outputs
   already exist.
3. Only missing RGB images are rendered.
4. Each fractal type receives its own output directory:
       rgb_root/<fractal_type>/<width>_<height>_iter<max_iter>/

This module does not perform tile-search, evaluation or rejection.
All regions are assumed to be validated earlier in the pipeline.
"""

import json
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from ai_fractals.generators import BaseFractalGenerator, create_generator

# ===============================================================
# Region grouping
# ===============================================================


def iter_regions_by_type(region_dir: Path) -> Iterator[Tuple[str, List[Path]]]:
    """
    Group region metadata files by fractal type.

    Parameters
    ----------
    region_dir : Path
        Directory containing validated region JSON files.

    Yields
    ------
    (fractal_type, list_of_paths)
        A tuple containing the fractal type and all region JSONs
        belonging to that type.
    """
    groups = {}

    for p in sorted(region_dir.glob("*.json")):
        meta = json.load(open(p))
        ftype = meta["fractal_type"]
        groups.setdefault(ftype, []).append(p)

    for ftype, paths in groups.items():
        yield ftype, paths


# ===============================================================
# Missing RGB detection
# ===============================================================


def iter_missing_for_type(
    ftype: str,
    paths: List[Path],
    rgb_root: Path,
    width: int,
    height: int,
    max_iter: int,
) -> Iterator[Tuple[str, List[Path], Path]]:
    """
    Determine which regions for a given fractal type still need RGB output.

    Parameters
    ----------
    ftype : str
        Fractal type (e.g., "mandelbrot").
    paths : list[Path]
        Region JSON files belonging to this fractal type.
    rgb_root : Path
        Root directory for RGB outputs.
    width, height : int
        Output resolution.
    max_iter : int
        Iteration count used in rendering.

    Yields
    ------
    (fractal_type, missing_paths, output_dir)
        A tuple containing the fractal type, the list of missing region
        JSONs, and the output directory for this fractal type.
    """
    out_dir = rgb_root / ftype / f"{width}_{height}_iter{max_iter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = set()
    for img in out_dir.glob("*.png"):
        cid = img.stem.split("_")[0]
        existing.add(cid)

    missing = []
    for p in paths:
        meta = json.load(open(p))
        cid = meta["compact_id"]
        if cid not in existing:
            missing.append(p)

    if missing:
        yield ftype, missing, out_dir


# ===============================================================
# RGB batch rendering
# ===============================================================


def run_rgb_batch(cfg: dict) -> None:
    """
    Render RGB fractal images for all missing regions across all fractal types.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing:
            region_evaluated_dir : str
            rgb_root_dir : str
            colormaps : list[str]
            hires_gen : dict with width, height, max_iter, device, etc.

    Notes
    -----
    The function distributes missing regions evenly across colormaps.
    Each colormap receives a subset of regions for each fractal type.
    """
    region_dir = Path(cfg["region_evaluated_dir"]).resolve()
    rgb_root = Path(cfg["rgb_root_dir"]).resolve()
    rgb_root.mkdir(parents=True, exist_ok=True)

    width = cfg["hires_gen"]["width"]
    height = cfg["hires_gen"]["height"]
    max_iter = cfg["hires_gen"]["max_iter"]
    colormaps = cfg["colormaps"]

    print("\n=== RGB Batch Rendering ===\n")
    print(f"Resolution: {width} * {height}")
    print(f"Max iter:   {max_iter}")
    print(f"Colormaps:  {len(colormaps)}\n")

    # -----------------------------------------------------------
    # Iterate over fractal types
    # -----------------------------------------------------------
    for ftype, paths in iter_regions_by_type(region_dir):
        print(f"\nFractal type: {ftype}")

        # find missing RGBs for this type
        for ftype, missing, out_dir in iter_missing_for_type(
            ftype, paths, rgb_root, width, height, max_iter
        ):
            total_missing = len(missing)
            if total_missing == 0:
                print("  No missing RGBs.")
                continue

            print(f"  Missing regions: {total_missing}")

            # distribute evenly across colormaps
            k = len(colormaps)
            per_cmap = total_missing // k if total_missing >= k else 1

            print(f"  Regions per colormap: {per_cmap}")

            # slice missing regions
            cmap_to_regions = {}
            idx = 0
            for cmap in colormaps:
                subset = missing[idx : idx + per_cmap]
                cmap_to_regions[cmap] = subset
                idx += per_cmap

            # ---------------------------------------------------
            # Render RGBs
            # ---------------------------------------------------
            for cmap, subset in cmap_to_regions.items():
                if not subset:
                    continue

                print(f"  Rendering colormap: {cmap} ({len(subset)} regions)")

                hires_rgb: BaseFractalGenerator = create_generator(
                    fractal_type=ftype,
                    colormap=cmap,
                    **cfg["hires_gen"],
                )

                for region_path in tqdm(subset, desc=f"{ftype}:{cmap}", ncols=80):
                    meta = json.load(open(region_path))
                    xmin, xmax, ymin, ymax = meta["bounds"]

                    rgb = hires_rgb.generate(xmin, xmax, ymin, ymax)
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                    img = Image.fromarray(rgb, mode="RGB")

                    cid = meta["compact_id"]
                    depth = meta["depth"]
                    out_path = out_dir / f"{cid}_{cmap}_iter{max_iter}_d{depth}.png"
                    img.save(out_path)

    print("\nRGB batch completed.\n")
