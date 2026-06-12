from pathlib import Path

import cv2


class ShorelineProcessor:
    def extract(self, img):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def save(self, path, shoreline):
        cv2.imwrite(path, shoreline)


class ShorelineBatchExtractor:
    def __init__(self, processor=None):
        self.processor = processor or ShorelineProcessor()

    def run(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in input_dir.glob("*.png"):
            img = cv2.imread(str(img_path))
            shoreline = self.processor.extract(img)
            out_path = output_dir / img_path.name
            self.processor.save(str(out_path), shoreline)
