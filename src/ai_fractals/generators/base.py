# ai_fractals/generators/base.py

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ai_fractals.logging_config import get_logger

if TYPE_CHECKING:
    from .factory import FractalState


class BaseFractalGenerator(ABC):
    """Abstract base class for fractal generators."""

    def __init__(
        self,
        state: FractalState,
        width=1024,
        height=1024,
        max_iter=1024,
        colormap="twilight",
        use_supersampling=True,
        log_level=logging.WARNING,
    ):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.colormap = colormap
        self.cmap = plt.get_cmap(colormap)
        self.use_supersampling = use_supersampling
        self.log_level = log_level

        self.state = state
        self.params = state.params

        # pytorch device, GPU if available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.log = get_logger(
            f"{self.__class__.__module__}.{id(self)}", level=log_level
        )
        self.log.info(f"using device: {self.device}")

    @abstractmethod
    def _compute(self, *args, **kwargs):
        """Raw escape-time iteration counts, shape (height, width)."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, *args, **kwargs):
        """RGB uint8 for display - article Section 4.1"""
        raise NotImplementedError

    @abstractmethod
    def generate_raw(self, *args, **kwargs):
        """Grayscale uint8 for analysis - avoids colormap distortion."""
        raise NotImplementedError

    def match_aspect(self, xmin, xmax, ymin, ymax):
        target_ratio = self.width / self.height
        x_range = xmax - xmin
        y_range = ymax - ymin

        current_ratio = x_range / y_range

        if abs(current_ratio - target_ratio) < 1e-9:
            return xmin, xmax, ymin, ymax

        if current_ratio > target_ratio:
            # too wide -> expand y
            new_y_range = x_range / target_ratio
            center_y = (ymin + ymax) / 2
            ymin = center_y - new_y_range / 2
            ymax = center_y + new_y_range / 2
        else:
            # too high -> expand x
            new_x_range = y_range * target_ratio
            center_x = (xmin + xmax) / 2
            xmin = center_x - new_x_range / 2
            xmax = center_x + new_x_range / 2

        return xmin, xmax, ymin, ymax

    def normalize_RGB(self, img):
        """normalizes and applies to 0-1 for colormap. converts to uint8 RGB"""
        x = img.astype(np.float32)

        # local contrast: diff between percentiles
        p1 = np.percentile(x, 1)
        p99 = np.percentile(x, 99)
        spread = max(p99 - p1, 1e-6)

        # adaptive gamma:
        gamma = np.clip(1.0 + (200.0 / spread), 0.7, 2.5)

        # normalise
        x = x / self.max_iter
        x = x ** (1.0 / gamma)

        colored = self.cmap(x)
        return (colored[:, :, :3] * 255).astype(np.uint8)

    def supersample(self, img):
        up = cv2.resize(
            img, (self.width * 2, self.height * 2), interpolation=cv2.INTER_LINEAR
        )

        down = cv2.resize(up, (self.width, self.height), interpolation=cv2.INTER_AREA)

        smooth = cv2.GaussianBlur(down, (3, 3), sigmaX=0.4)

        return smooth
