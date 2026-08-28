#!/usr/bin/env python3
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml
from batch_generation.batch_region_params import run_region_batch
from batch_generation.batch_rgb_from_meta import run_rgb_batch
from batch_generation.batch_shoreline_from_meta import run_shoreline_batch
from dotenv import load_dotenv
from utils import get_system_specs_str, resolve_output_paths, setup_logging

load_dotenv()
ROOT = Path(os.getenv("PROJECT_ROOT"))
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_logging(redirect_path=ROOT / "logs" / f"{ts}_batch.log")
log = logging.getLogger(__name__)
log.info(get_system_specs_str())


# -----------------------------------------------------------------


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

    # set output path
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
