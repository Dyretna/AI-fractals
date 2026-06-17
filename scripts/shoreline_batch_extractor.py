from pathlib import Path

import cv2

from ai_fractals.processing import EdgeDetector


class ShorelineBatchExtractor:
    """
    Batch-processes images and extracts edge maps using EdgeDetector.

    This class replaces the old ShorelineBatchExtractor and is fully
    generic: any edge detection configuration can be passed in.

    Example
    -------
        detector = EdgeDetector(canny_low=80, canny_high=160)
        batch = EdgeBatchExtractor(detector)
        batch.run("input_dir", "output_dir")
    """

    def __init__(self, detector: EdgeDetector | None = None):
        """
        Initialize the batch extractor.

        Parameters
        ----------
        detector : EdgeDetector, optional
            Custom edge detector instance. If None, a default EdgeDetector
            will be created.
        """
        self.detector = detector or EdgeDetector()

    def run(self, input_dir: str | Path, output_dir: str | Path) -> None:
        """
        Process all PNG images in input_dir and save edge maps to output_dir.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing input PNG images.
        output_dir : str or Path
            Directory where edge maps will be saved.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in input_dir.glob("*.png"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            edges = self.detector.detect(img)
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), edges)

            print(f"[OK] Processed: {img_path.name}")


if __name__ == "__main__":
    """
    Example usage:
        Run edge extraction on all PNG images in the input directory and
        save the resulting edge maps to the output directory.

        Adjust the paths below as needed.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()

    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    DATASET_DIR = PROJECT_ROOT / "dataset"

    # Configure detector (tweak thresholds as needed)
    detector = EdgeDetector(
        canny_low=80,
        canny_high=160,
        apply_smoothing=True,
        smoothing_method="gaussian",
        smoothing_kernel=5,
        smoothing_sigma=1.2,
    )

    extractor = ShorelineBatchExtractor(detector)

    input_dir = DATASET_DIR / "mandelbrot" / "1024_1024_iter1024"
    output_dir = DATASET_DIR / "mandelbrot" / "edges" / "1024_1024_iter1024"

    extractor.run(input_dir, output_dir)
    print("Edge extraction complete.")
