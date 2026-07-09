# ai_fractals/generators/base.py

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class BaseFractalGenerator(ABC):
    """Abstract base class for fractal generators."""

    fractal_type = "unknown (in base class)"

    def __init__(
        self,
        width=1024,
        height=1024,
        max_iter=1024,
        colormap="twilight_shifted",
        normalize_mode="global",
        normalize_gamma=0.5,
        use_supersampling=True,
        device="cpu",
    ):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.colormap = colormap
        self.cmap = plt.get_cmap(colormap)
        self.normalize_mode = normalize_mode
        self.normalize_gamma = normalize_gamma
        self.use_supersampling = use_supersampling
        self.device = device

        # pytorch device, GPU if available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.log = logging.getLogger(__name__)

        self.log.info(f"using device: {self.device}")

    # ------------------------------------------------------------
    # Public API-methods
    # ------------------------------------------------------------

    def generate(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> np.ndarray:
        """RGB uint8 for display"""

        if self.normalize_mode == "global":
            img = self._normalize_global(self._compute(xmin, xmax, ymin, ymax))
        else:
            img = self._normalize_local(self._compute(xmin, xmax, ymin, ymax))

        if self.use_supersampling:
            img = self._supersample(img)
        return img

    def generate_raw(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> np.ndarray:
        """Grayscale uint8 for analysis - avoids colormap distortion."""

        img = self._compute(xmin, xmax, ymin, ymax)
        return (img / self.max_iter * 255).astype(np.uint8)

    @abstractmethod
    def default_bounds(self) -> tuple[float, float, float, float]:
        """Return default bounds of fractal type"""
        raise NotImplementedError

    # ------------------------------------------------------------
    # Private
    # ------------------------------------------------------------

    @abstractmethod
    def _compute(self, *args, **kwargs) -> np.ndarray:
        """Raw escape-time iteration counts, shape (height, width)."""
        raise NotImplementedError

    def _match_aspect(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> tuple[float, float, float, float]:
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

    def _normalize_global(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        mn = x.min()
        mx = x.max()
        spread = max(mx - mn, 1e-6)
        x = (x - mn) / spread

        # contrast boost via gamma
        x = x**self.normalize_gamma

        colored = self.cmap(x)
        return (colored[:, :, :3] * 255).astype(np.uint8)

    def _normalize_local(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        p1 = np.percentile(x, 1)
        p99 = np.percentile(x, 99)
        spread = max(p99 - p1, 1e-6)
        gamma = np.clip(1.0 + (200.0 / spread), 0.7, 2.5)
        x = x / self.max_iter
        x = x ** (1.0 / gamma)
        colored = self.cmap(x)
        return (colored[:, :, :3] * 255).astype(np.uint8)

    def _supersample(self, img: np.ndarray) -> np.ndarray:
        """Upscales, downscales, and applies Gaussian blur"""
        up = cv2.resize(
            img, (self.width * 2, self.height * 2), interpolation=cv2.INTER_LINEAR
        )
        down = cv2.resize(up, (self.width, self.height), interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(down, (3, 3), sigmaX=0.4)

        return smooth

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
