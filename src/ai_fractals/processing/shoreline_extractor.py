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
