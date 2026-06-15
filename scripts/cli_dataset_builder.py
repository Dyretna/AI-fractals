#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from matplotlib import colormaps

from ai_fractals.data.dataset_builder import FractalDatasetBuilder


def main():
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))

    parser = argparse.ArgumentParser(
        description=(
            "Generate a fractal dataset using a tile-search refinement strategy.\n"
            "This tool renders low-resolution tiles to locate visually interesting\n"
            "regions, then produces high-resolution fractal images using\n"
            "supersampling and post-processing."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
Examples:
    Generate 50 Mandelbrot images in 4K:
        python cli_dataset_builder.py --n 50 --type mandelbrot --width 3840 --height 2160

    Generate Julia set images with a custom colormap:
        python cli_dataset_builder.py --type julia --cmap plasma

    Save output to a custom directory:
        python cli_dataset_builder.py --out /tmp/fractals

    Enable verbose logging:
        python cli_dataset_builder.py --verbose 1

{colormaps}
""",
    )

    # -----------------------------
    # General options
    # -----------------------------
    general = parser.add_argument_group("General options")
    general.add_argument(
        "--n", type=int, default=50, help="Number of images to generate (default: 50)"
    )
    general.add_argument(
        "--type",
        type=str,
        default="mandelbrot",
        help="Fractal type: mandelbrot or julia (default: mandelbrot)",
    )

    # -----------------------------
    # Rendering options
    # -----------------------------
    render = parser.add_argument_group("Rendering options")
    render.add_argument(
        "--width", type=int, default=1024, help="Output image width (default: 1024)"
    )
    render.add_argument(
        "--height", type=int, default=1024, help="Output image height (default: 1024)"
    )
    render.add_argument(
        "--max_iter",
        type=int,
        default=1024,
        help="Maximum iterations for escape-time algorithm (default: 1024)",
    )
    render.add_argument(
        "--cmap",
        type=str,
        default="twilight_shifted",
        help=("Matplotlib colormap to use (default: twilight_shifted)"),
    )

    # -----------------------------
    # Output Directory
    # -----------------------------
    general.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: dataset/fractals/<type>/<width>_<height>/)",
    )

    # -----------------------------
    # Debugging / Logging
    # -----------------------------
    debug = parser.add_argument_group("Debugging")
    debug.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level: 0=warnings only, 1=info logs (default: 0)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.out:
        output_dir = Path(args.out)
    else:
        output_dir = project_root / "dataset" / "fractals" / args.type

    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging level
    log_level = logging.INFO if args.verbose else logging.WARNING

    # Build dataset
    builder = FractalDatasetBuilder(
        fractal_type=args.type,
        width=args.width,
        height=args.height,
        max_iter=args.max_iter,
        colormap=args.cmap,
        log_level=log_level,
        output_dir=output_dir,
    )

    print("\n", builder, "\n")
    builder.run(args.n)


if __name__ == "__main__":
    main()
