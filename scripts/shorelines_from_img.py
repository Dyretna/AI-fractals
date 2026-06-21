# shorelines_from_img.py
"""
Build shorelines from fractal images.
Directly augments into wanted size and number of flips
without intermediate saving.

this was a first build, but since a CNN model needs many shorelines
(around 5000-10000, size 256x256), it would also require many source images, that take up a lot of space
(if they are worth saving, they are usually much bigger than 256x256).

the other "from scratch" solves this issue, but by creating shorelines
without first loading from images.

This script was a step for me to conceptualize the problem essentially.

Steps:
1. Load RGB fractal image.
2. Extract shoreline (in memory).
3. Evaluate shoreline quality.
4. If quality is acceptable, generate N augmentations.
5. Save augmented images.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.processing import EdgeDetector, ImageAugmenter


def shorelines_from_img(
    input_dir: Path,
    output_dir: Path,
    detector: Optional[EdgeDetector] = None,
    evaluator: Optional[FractalQualityEvaluator] = None,
    augmenter: Optional[ImageAugmenter] = None,
) -> None:
    detector = detector or EdgeDetector()
    evaluator = evaluator or FractalQualityEvaluator()
    augmenter = augmenter or ImageAugmenter()

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob("*.png"):
        rgb = cv2.imread(str(img_path))
        if rgb is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        # Step 1: extract shoreline
        shoreline = detector.detect(rgb)

        # Step 2: evaluate quality
        score, passed, _ = evaluator.evaluate(shoreline)
        if not passed:
            print(f"[SKIP] Low-quality shoreline: {img_path.name} (score={score:.3f})")
            continue

        # step 3: augment
        variants = augmenter.augment(shoreline)

        # step 4: save
        orig_name = img_path.stem  # removes .png

        for tag, aug in variants.items():
            out = (aug * 255).astype(np.uint8)

            # new filename: originalname_tag.png
            filename = f"{orig_name}_{tag}.png"
            save_path = output_dir / filename

            cv2.imwrite(str(save_path), out)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    DATASET_DIR = PROJECT_ROOT / "dataset" / "fractals"

    input_dir = DATASET_DIR / "mandelbrot" / "1024_1024_iter1024"
    output_dir = DATASET_DIR / "shorelines" / "mandelbrot" / "1024_1024_iter1024"
    print("\ninput_dir: ", input_dir)
    print("Path is dir: ", input_dir.is_dir())

    detector = EdgeDetector(
        canny_low=40,
        canny_high=120,
        apply_smoothing=True,
        smoothing_method="gaussian",
        smoothing_kernel=3,
        smoothing_sigma=0.8,
    )
    evaluator = FractalQualityEvaluator(
        quality_threshold=0.3,
        min_edge_ratio=0.03,
        max_edge_ratio=0.45,
        min_inside_ratio=0.0001,
        max_inside_ratio=0.9999,
    )
    augmenter = ImageAugmenter(
        horizontal_flip=True, vertical_flip=True, target_size=(256, 256)
    )

    shorelines_from_img(
        input_dir=input_dir,
        output_dir=output_dir,
        detector=detector,
        evaluator=evaluator,
        augmenter=augmenter,
    )

    print("Shoreline augmentation pipeline complete.")
