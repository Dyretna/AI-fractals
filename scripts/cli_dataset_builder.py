#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from ai_fractals.data.dataset_builder import FractalDatasetBuilder


def main():
    # Load environment variables
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))

    # CLI
    parser = argparse.ArgumentParser(
        description="Generate fractal dataset via tile-search."
    )
    parser.add_argument(
        "--n", type=int, default=50, help="Number of images to generate"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="mandelbrot",
        help="Fractal type: mandelbrot or julia",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="width of fractal",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1200,
        help="height of fractal",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=900,
        help="sets max iterations for the fractal generator",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="twilight",
        help="sets the colormap of the fractal (using matplotlibs cmaps)",
    )

    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Determine output directory
    if args.out:
        output_dir = Path(args.out)
    else:
        output_dir = project_root / "dataset" / "fractals" / args.type

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    builder = FractalDatasetBuilder(
        fractal_type=args.type,
        width=args.width,
        height=args.height,
        max_iter=args.max_iter,
        colormap=args.cmap,
        output_dir=output_dir,
    )

    builder.run(args.n)


if __name__ == "__main__":
    main()
