"""
saving_shorelines.py

automates the saving of fractal images and their shorelines.
"""

# python std
import os

# 3rd party
import cv2

# local
from ai_fractals.config import FRACTAL_IMG_DIR, SHORELINE_IMG_DIR
from ai_fractals.generators import FractalGenerator, MandelbrotGenerator
from ai_fractals.io.img_saver import ImageSaver, PltImageSaver
from ai_fractals.processing import ShorelineProcessor


class FractalSaverPipeline:
    def __init__(
        self,
        generator: FractalGenerator,  # abstract
        saver: ImageSaver,  # abstract
        shoreline_processor: ShorelineProcessor,  # TODO:not abc
    ):
        self.generator = generator
        self.saver = saver
        self.shoreline_processor = shoreline_processor

    def generate_and_save(self, params_list, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(params_list)
        for i, params in enumerate(params_list):
            img = self.generator.generate(*params)
            path = os.path.join(output_dir, f"{i}.png")
            self.saver.save(path, img)
            print("image saved to: ", path)

    def extract_and_save_shorelines(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, name)
            img = cv2.imread(img_path)

            shoreline = self.shoreline_processor.extract(img)
            out_path = os.path.join(output_dir, name)
            self.shoreline_processor.save(out_path, shoreline)


if __name__ == "__main__":
    mandelbrot_params_list = [
        # xmin, xmax, ymin, ymax
        (-2.0, 1.0, -1.5, 1.5),
        (-1.8, 0.5, -1.2, 1.2),
        (-0.75, -0.6, 0.1, 0.3),
    ]

    Mandelbrot_pipeline = FractalSaverPipeline(
        generator=MandelbrotGenerator(width=800, height=600, max_iter=256),
        saver=PltImageSaver(),
        shoreline_processor=ShorelineProcessor(),
    )

    Mandelbrot_pipeline.generate_and_save(
        mandelbrot_params_list, FRACTAL_IMG_DIR / "mandelbrot"
    )
    Mandelbrot_pipeline.extract_and_save_shorelines(
        FRACTAL_IMG_DIR / "mandelbrot",
        SHORELINE_IMG_DIR / "mandelbrot",
    )
