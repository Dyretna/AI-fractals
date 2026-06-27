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
        ...     smoothing_method='gaussian',
        ...     smoothing_kernel=3
        ... )
        >>> edges = detector.detect(image)
    """

    def __init__(
        self,
        canny_low: int = 40,
        canny_high: int = 120,
        apply_smoothing: bool = True,
        smoothing_method: str = "gaussian",
        smoothing_kernel: int = 3,
        smoothing_sigma: float = 0.8,
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

    def __str__(self):
        rows = [f"{self.__class__.__name__}:"]
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue

            if isinstance(v, (int, float, str, bool)):
                val = v
            else:
                val = type(v).__name__

            rows.append(f"  {k}: {val}")

        return "\n".join(rows)
