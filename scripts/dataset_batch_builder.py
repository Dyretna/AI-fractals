#!/usr/bin/env python3


from ai_fractals.data import CURATED_COLORMAPS, FractalDatasetBuilder


def dataset_batch_builder(colormaps):
    for max_iter in [
        256,
        512,
        1024,
        # 2048
    ]:
        for colormap in colormaps:
            # Build dataset
            builder = FractalDatasetBuilder(
                fractal_type="mandelbrot",
                width=1024,
                height=1024,
                max_iter=max_iter,
                save_max_depth=10,
                colormap=colormap,
            )

            print("\n", builder, "\n")
            builder.run(9)


if __name__ == "__main__":
    dataset_batch_builder(CURATED_COLORMAPS)
