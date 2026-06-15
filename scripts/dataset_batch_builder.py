#!/usr/bin/env python3


from ai_fractals.data import CURATED_COLORMAPS, FractalDatasetBuilder


def dataset_batch_builder(colormaps):
    for colormap in colormaps:
        for max_iter in [
            512,
            # 1024, 2048
        ]:
            # Build dataset
            builder = FractalDatasetBuilder(
                fractal_type="mandelbrot",
                width=1024,
                height=1024,
                max_iter=max_iter,
                save_max_depth=15,
                colormap=colormap,
            )

            print("\n", builder, "\n")
            builder.run(15)


if __name__ == "__main__":
    dataset_batch_builder(CURATED_COLORMAPS)
