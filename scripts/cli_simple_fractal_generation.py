#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from matplotlib import colormaps

# Core components
from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.data import RGBDatasetBuilder
from ai_fractals.generators import create_generator
from ai_fractals.processing import EdgeDetector
from ai_fractals.search import TileSearchBasic


def main():
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = str(device).strip().lower()

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
        help="Fractal type: mandelbrot (default: mandelbrot)",
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
        help="Output directory (default: dataset/rgb/<type>/<width>_<height>_<maxiter>/)",
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
        output_dir = (
            project_root
            / "dataset"
            / "rgb"
            / args.type
            / f"{args.width}_{args.height}_iter{args.max_iter}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Construct components
    # -----------------------------

    detector = EdgeDetector()
    evaluator = FractalQualityEvaluator(detector)

    tile_search = TileSearchBasic(
        tile_gen=create_generator(
            fractal_type=args.type,
            width=256,
            height=256,
            max_iter=256,
            colormap=args.cmap,
            use_supersampling=False,
            device=device,
        ),
        evaluator=evaluator,
        n_tiles=5,
        top_k=5,
    )

    hires = create_generator(
        fractal_type=args.type,
        width=args.width,
        height=args.height,
        max_iter=args.max_iter,
        colormap=args.cmap,
        use_supersampling=False,
        device=device,
    )

    # -----------------------------
    # Build dataset
    # -----------------------------
    builder = RGBDatasetBuilder(
        # tools
        tile_search=tile_search,
        hires_generator=hires,
        evaluator=evaluator,
        # output
        output_dir=output_dir,
        # cmap
        colormap=args.cmap,
    )

    print("\n", builder, "\n")
    builder.run(args.n)


if __name__ == "__main__":
    main()
