"""
Image filtering and smoothing operations.

Provides both functional and class-based interfaces for image smoothing.
"""

import cv2
import numpy as np


class SmoothingFilter:
    """
    Apply smoothing/blurring to images.

    Supports multiple smoothing methods:
    - median: Good for removing salt-and-pepper noise
    - gaussian: General purpose smoothing
    - bilateral: Edge-preserving smoothing
    - blur: Simple averaging blur

    Example:
        >>> smoother = SmoothingFilter(method='median', kernel_size=5)
        >>> blurred = smoother.apply(image)
    """

    VALID_METHODS = {"median", "gaussian", "bilateral", "blur"}

    def __init__(
        self,
        method: str = "median",
        kernel_size: int = 5,
        sigma: float = 0,
    ):
        """
        Initialize smoothing filter.

        Args:
            method: Type of smoothing ('bilateral', 'blur', 'gaussian', 'median')
            kernel_size: Size of the kernel (must be odd number for median/gaussian)
            sigma: Sigma value for Gaussian/bilateral (auto-calculated if 0)

        Raises:
            ValueError: If method is invalid or kernel_size is invalid
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got '{method}'"
            )

        if kernel_size < 1:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")

        if method in ("median", "gaussian") and kernel_size % 2 == 0:
            raise ValueError(f"{method} requires odd kernel_size, got {kernel_size}")

        self.method = method
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply smoothing filter to image.

        Args:
            image: Input image

        Returns:
            Smoothed image
        """
        if self.method == "bilateral":
            return cv2.bilateralFilter(
                image,
                self.kernel_size,
                self.sigma if self.sigma > 0 else 75,
                self.sigma if self.sigma > 0 else 75,
            )
        elif self.method == "blur":
            return cv2.blur(image, (self.kernel_size, self.kernel_size))
        elif self.method == "gaussian":
            return cv2.GaussianBlur(
                image,
                (self.kernel_size, self.kernel_size),
                sigmaX=self.sigma,
                sigmaY=self.sigma,
            )
        elif self.method == "median":
            return cv2.medianBlur(image, self.kernel_size)
        else:
            raise ValueError(f"Unexpected method: {self.method}")

    def __repr__(self) -> str:
        return (
            f"SmoothingFilter(method='{self.method}', "
            f"kernel_size={self.kernel_size}, sigma={self.sigma})"
        )
