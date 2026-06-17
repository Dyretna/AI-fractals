"""
Edge detection operations.

Provides edge detection using Canny algorithm with optional preprocessing.
"""

import cv2
import numpy as np

from .filters import SmoothingFilter


class EdgeDetector:
    """
    Detects edges in images using Canny edge detection.

    Can optionally apply smoothing preprocessing before edge detection
    to reduce noise and improve edge quality.

    Example:
        >>> detector = EdgeDetector(
        ...     canny_low=100,
        ...     canny_high=200,
        ...     smoothing_method='median',
        ...     smoothing_kernel=5
        ... )
        >>> edges = detector.detect(image)
    """

    def __init__(
        self,
        canny_low: int = 100,
        canny_high: int = 200,
        apply_smoothing: bool = True,
        smoothing_method: str = "median",
        smoothing_kernel: int = 5,
        smoothing_sigma: float = 0,
    ):
        """
        Initialize edge detector.

        Args:
            canny_low: Lower threshold for Canny (0-255)
            canny_high: Upper threshold for Canny (0-255)
            apply_smoothing: Whether to apply smoothing before edge detection
            smoothing_method: Smoothing method ('median', 'gaussian', 'bilateral', 'blur')
            smoothing_kernel: Kernel size for smoothing
            smoothing_sigma: Sigma for gaussian/bilateral smoothing

        Raises:
            ValueError: If thresholds are invalid
        """
        if not (0 <= canny_low <= 255):
            raise ValueError(f"canny_low must be 0-255, got {canny_low}")

        if not (0 <= canny_high <= 255):
            raise ValueError(f"canny_high must be 0-255, got {canny_high}")

        if canny_low >= canny_high:
            raise ValueError(
                f"canny_low ({canny_low}) must be < canny_high ({canny_high})"
            )

        self.canny_low = canny_low
        self.canny_high = canny_high
        self.apply_smoothing = apply_smoothing

        # Create smoothing filter if enabled
        self.smoother = None
        if apply_smoothing:
            self.smoother = SmoothingFilter(
                method=smoothing_method,
                kernel_size=smoothing_kernel,
                sigma=smoothing_sigma,
            )

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in an image.

        Args:
            image: Input image (can be color or grayscale)

        Returns:
            Binary edge image (single channel)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply smoothing if enabled
        if self.apply_smoothing and self.smoother is not None:
            gray = self.smoother.apply(gray)

        # Detect edges
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        return edges

    def __str__(self) -> str:
        """Return detailed edge detector configuration."""
        rows = [
            "--- Edge Detection (Canny) ---",
            f"Canny threshold low: {self.canny_low}",
            f"Canny threshold high: {self.canny_high}",
        ]
        if self.apply_smoothing and self.smoother is not None:
            rows.append(f"Smoothing: {self.smoother.method}")
            rows.append(f"Kernel size: {self.smoother.kernel_size}")
            if self.smoother.sigma > 0:
                rows.append(f"Sigma: {self.smoother.sigma}")
        else:
            rows.append("Smoothing: disabled")

        return "\n".join(rows)

    def __repr__(self) -> str:
        smooth_info = (
            f", smoothing={self.smoother.method}"
            if self.apply_smoothing
            else ", no_smoothing"
        )
        return f"EdgeDetector(canny=({self.canny_low}, {self.canny_high}){smooth_info})"
