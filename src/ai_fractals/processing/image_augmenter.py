# ImageAugmenter for Fractal ML Pipeline
"""
This module performs data augmentation on fractal images.

What the augmenter does:
- Loads images (grayscale).
- Applies controlled augmentations:
  - flipping
  - resizing
"""

from typing import Tuple

import cv2
import numpy as np


class ImageAugmenter:
    """Performs geometric data augmentation on images."""

    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.target_size = target_size

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def augment(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return a dict of augmented variants:
        {
            "orig": img,
            "h":    flipped horizontally,
            "v":    flipped vertically,
            "hv":   flipped both ways,
        }
        All variants are resized to target_size.
        """

        flipped_variants = self._flip_all(img)

        return {tag: self._resize(v) for tag, v in flipped_variants.items()}

    def _flip_all(self, img: np.ndarray) -> dict[str, np.ndarray]:
        variants = {}

        variants["orig"] = img

        if self.horizontal_flip:
            variants["h"] = cv2.flip(img, 1)

        if self.vertical_flip:
            variants["v"] = cv2.flip(img, 0)

        if self.horizontal_flip and self.vertical_flip:
            variants["hv"] = cv2.flip(img, -1)

        return variants

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
