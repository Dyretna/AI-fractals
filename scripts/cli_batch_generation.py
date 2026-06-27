#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import yaml
from batch_generation.batch_RGB_generation import run_rgb_batch
from batch_generation.batch_shoreline_generation import run_shoreline_batch
from dotenv import load_dotenv

from ai_fractals.data import (
    CURATED_COLORMAPS,
    DISCRETE_GRADIENTS,
    THEMED,
    THREE_COLOR_GRADIENTS,
)

# -----------------------------------------------------------------
# Handling and colormaps to insert into cfg dict
# -----------------------------------------------------------------

COLORMAP_GROUPS = {
    "curated_colormaps": CURATED_COLORMAPS,
    "discrete_gradients": DISCRETE_GRADIENTS,
    "three_color_gradients": THREE_COLOR_GRADIENTS,
    "themed": THEMED,
}


def resolve_colormap_group(cfg: dict) -> dict:
    """Resolve 'colormap_group' into a concrete 'colormaps' list."""
    group_key = cfg.get("colormap_group", None)

    if not group_key:
        return cfg  # nothing to do

    if group_key not in COLORMAP_GROUPS:
        raise ValueError(
            f"Unknown colormap_group '{group_key}'. "
            f"Available groups: {list(COLORMAP_GROUPS.keys())}"
        )

    cfg["colormaps"] = COLORMAP_GROUPS[group_key]
    return cfg


# -----------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))

    parser = argparse.ArgumentParser(
        description="Run one or more batch fractal generation jobs using a YAML configs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    python scripts/cli_batch_generation.py --config configs/rgb/rgb_cfg.yaml
    python scripts/cli_batch_generation.py --config configs/shoreline/shoreline_cfg.yaml
    python scripts/cli_batch_generation.py --config cfg1.yaml cfg2.yaml cfg3.yaml
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="One or more Paths to YAML configuration files.",
    )

    args = parser.parse_args()

    for cfg_path in args.config:
        # Load YAML
        cfg: dict = yaml.safe_load(open(cfg_path))
        cfg = resolve_colormap_group(cfg)

        # Determine job type
        job_type = cfg.get("job_type", None)
        if job_type is None:
            raise ValueError("Config file must contain a 'job_type' field.")

        print(f"\nLoaded config: {args.config}")
        print(f"Job type: {job_type}\n")

        # Dispatch to correct batch script
        if job_type == "shoreline":
            run_shoreline_batch(cfg, project_root)

        elif job_type == "rgb":
            run_rgb_batch(cfg, project_root)

        else:
            raise ValueError(f"Unknown job_type: {job_type} in {cfg_path}")


if __name__ == "__main__":
    main()
