import os
from pathlib import Path

from dotenv import load_dotenv

from ai_fractals.data import CURATED_COLORMAPS, FractalDatasetBuilder


def main():
    load_dotenv()
    project_root = Path(os.getenv("PROJECT_ROOT"))

    fractal_type = "mandelbrot"
    width = 1024
    height = 1024

    # test and check qualities...
    for max_iter in (
        256,
        512,
        1024,
        # 2048,
    ):
        output_dir = (
            project_root
            / "fractals"
            / "dataset"
            / fractal_type
            / f"{width}_{height}_iter{max_iter}"
        )

        for colormap in CURATED_COLORMAPS:
            builder = FractalDatasetBuilder(
                fractal_type=fractal_type,
                width=width,
                height=height,
                max_iter=max_iter,
                save_max_depth=10,
                colormap=colormap,
                output_dir=output_dir,
            )

            print("\n", builder, "\n")
            builder.run(9)


if __name__ == "__main__":
    main()
