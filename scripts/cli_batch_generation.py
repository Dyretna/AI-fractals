#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import yaml
from batch_generation.batch_region_params import run_region_batch
from batch_generation.batch_rgb_from_meta import run_rgb_batch
from batch_generation.batch_shoreline_from_meta import run_shoreline_batch
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


def resolve_output_paths(cfg: dict, project_root: Path):
    job_type = cfg.get("job_type", None)

    if job_type == "region":
        # region builder has ONE output-dir
        if "output_dir" not in cfg:
            raise ValueError("Config missing 'output_dir'.")
        cfg["output_path"] = (project_root / cfg["output_dir"]).resolve()
        return cfg

    elif job_type == "shoreline":
        # shoreline has FIVE paths
        required = [
            "region_raw_dir",
            "region_evaluated_dir",
            "region_rejected_dir",
            "shoreline_evaluated_dir",
            "shoreline_rejected_dir",
        ]
        for key in required:
            if key not in cfg:
                raise ValueError(f"Config missing '{key}'.")

        cfg["region_raw_dir"] = Path(project_root / cfg["region_raw_dir"]).resolve()
        cfg["region_evaluated_dir"] = Path(
            project_root / cfg["region_evaluated_dir"]
        ).resolve()
        cfg["region_rejected_dir"] = Path(
            project_root / cfg["region_rejected_dir"]
        ).resolve()
        cfg["shoreline_evaluated_dir"] = Path(
            project_root / cfg["shoreline_evaluated_dir"]
        ).resolve()
        cfg["shoreline_rejected_dir"] = Path(
            project_root / cfg["shoreline_rejected_dir"]
        ).resolve()
        return cfg

    elif job_type == "rgb":
        # rgb has ONE output-dir
        if "rgb_root_dir" not in cfg:
            raise ValueError("Config missing 'rgb_root_dir'.")
        cfg["rgb_root_dir"] = (project_root / cfg["rgb_root_dir"]).resolve()
        return cfg

    else:
        raise ValueError(f"Unknown job_type: {job_type}")


# -----------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run a batch job using a YAML configs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    python scripts/cli_batch_generation.py --config configs/region/region_cfg.yaml
    python scripts/cli_batch_generation.py --config configs/shoreline/shoreline_cfg.yaml
    python scripts/cli_batch_generation.py --config configs/rgb/rgb_cfg.yaml
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration files.",
    )
    args = parser.parse_args()

    # Load YAML
    config_path = args.config
    cfg: dict = yaml.safe_load(open(config_path))

    # Determine job type
    job_type = cfg.get("job_type", None)
    if job_type is None:
        raise ValueError("Config file must contain a 'job_type' field.")

    if job_type == "rgb":
        cfg = resolve_colormap_group(cfg)

    # set output path
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))
    cfg = resolve_output_paths(cfg, project_root)

    print(f"\nLoaded config: {config_path}")
    print(f"Job type: {job_type}\n")

    # Dispatch to correct batch script
    if job_type == "region":
        run_region_batch(cfg)

    elif job_type == "shoreline":
        run_shoreline_batch(cfg)

    elif job_type == "rgb":
        run_rgb_batch(cfg)

    else:
        raise ValueError(f"Unknown job_type: {job_type} in {config_path}")


if __name__ == "__main__":
    main()
