from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class BaseFractalGenerator(ABC):
    """Abstract base class for fractal generators."""

    def __init__(self, width=800, height=600, max_iter=256, colormap="twilight"):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.colormap = colormap
        self.cmap = plt.get_cmap(colormap)

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


class MandelbrotGenerator(BaseFractalGenerator):
    def _compute(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)

        # escape iteration count (0 = not escaped yet)
        img = np.zeros(C.shape, dtype=np.int32)

        for i in range(1, self.max_iter):
            mask = img == 0  # only update pixels that haven't escaped
            Z[mask] = Z[mask] * Z[mask] + C[mask]

            escaped = (np.abs(Z) >= 2) & (img == 0)
            img[escaped] = i

        # pixels that never escaped
        img[img == 0] = self.max_iter

        return img

    def generate(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        img = self._compute(xmin, xmax, ymin, ymax)
        return self.normalize_RGB(img)

    def generate_raw(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        img = self._compute(xmin, xmax, ymin, ymax)
        return (img / self.max_iter * 255).astype(np.uint8)


class JuliaGenerator(BaseFractalGenerator):
    def _compute(self, c, xmin, xmax, ymin, ymax) -> np.ndarray:
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # 0 = not escaped yet
        img = np.zeros(Z.shape, dtype=np.int32)

        for i in range(1, self.max_iter):
            mask = img == 0  # only update active pixels
            Z[mask] = Z[mask] * Z[mask] + c

            escaped = (np.abs(Z) >= 2) & (img == 0)
            img[escaped] = i

        # pixels that never escaped
        img[img == 0] = self.max_iter

        return img

    def generate(self, c, xmin, xmax, ymin, ymax) -> np.ndarray:
        img = self._compute(c, xmin, xmax, ymin, ymax)
        return self.normalize_RGB(img)

    def generate_raw(self, c, xmin, xmax, ymin, ymax) -> np.ndarray:
        img = self._compute(c, xmin, xmax, ymin, ymax)
        return (img / self.max_iter * 255).astype(np.uint8)
