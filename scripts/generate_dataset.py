#!/usr/bin/env python3
"""
Automatic Fractal Dataset Generation - Main Script

Usage:
    python scripts/generate_dataset.py --type mandelbrot --target 100 --quality-threshold 0.3
"""

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from env_loader import ensure_env_loaded  # noqa: E402

ensure_env_loaded()

import argparse  # noqa: E402

from ai_fractals.hardware_config import setup_hardware  # noqa: E402
from ai_fractals.pipeline import AutomaticFractalPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate fractal dataset")
    parser.add_argument(
        "--type", choices=["mandelbrot", "julia", "both"], default="mandelbrot"
    )
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="dataset")
    args = parser.parse_args()

    # Show hardware config
    print("\n" + "=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    setup_hardware(verbose=True)

    # Generate fractals
    if args.type in ["mandelbrot", "both"]:
        print("\n" + "=" * 60)
        print("GENERATING MANDELBROT SET FRACTALS")
        print("=" * 60)
        pipeline = AutomaticFractalPipeline(
            output_dir=args.output_dir,
            target_images=args.target,
            quality_threshold=args.quality_threshold,
            parallel_workers=1,
            fractal_type="mandelbrot",
        )
        pipeline.run()

    if args.type in ["julia", "both"]:
        print("\n" + "=" * 60)
        print("GENERATING JULIA SET FRACTALS")
        print("=" * 60)
        pipeline = AutomaticFractalPipeline(
            output_dir=args.output_dir,
            target_images=args.target,
            quality_threshold=args.quality_threshold,
            parallel_workers=1,
            fractal_type="julia",
        )
        pipeline.run()

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
