import matplotlib.pyplot as plt
import numpy as np


class FractalGenerator:
    """Abstract base class for fractal generators."""

    def generate(self, *args, **kwargs):
        raise NotImplementedError


class MandelbrotGenerator(FractalGenerator):
    def __init__(self, width=800, height=600, max_iter=256, colormap="twilight"):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.colormap = colormap
        self.cmap = plt.get_cmap(colormap)

    def _compute(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        # Raw escape-time iteration counts, shape (height, width).
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros(C.shape, dtype=complex)

        img = np.zeros(C.shape, dtype=int)
        for _ in range(self.max_iter):
            mask = np.abs(Z) < 2
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            img += mask

        return img

    def generate(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        # RGB uint8 for display - article Section 4.1
        img = self._compute(xmin, xmax, ymin, ymax)

        # Normalize to 0-1 for colormap
        img_normalized = img / self.max_iter

        # Apply colormap to get RGB
        img_colored = self.cmap(img_normalized)

        # Convert to uint8 RGB (drop alpha channel)
        img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)

        return img_rgb

    def generate_raw(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        # Grayscale uint8 for analysis - avoids colormap distortion.
        img = self._compute(xmin, xmax, ymin, ymax)
        return (img / self.max_iter * 255).astype(np.uint8)


class JuliaGenerator(FractalGenerator):
    def __init__(self, width=800, height=600, max_iter=256, colormap="twilight"):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.colormap = colormap
        self.cmap = plt.get_cmap(colormap)

    def _compute(self, c, xmin, xmax, ymin, ymax) -> np.ndarray:
        # Raw escape-time iteration counts, shape (height, width).
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        img = np.zeros(Z.shape, dtype=int)
        for _ in range(self.max_iter):
            mask = np.abs(Z) < 2
            Z[mask] = Z[mask] * Z[mask] + c
            img += mask

        return img

    def generate(self, c, xmin, xmax, ymin, ymax) -> np.ndarray:
        # RGB uint8 for display - article Section 4.1
        img = self._compute(c, xmin, xmax, ymin, ymax)

        # Normalize to 0-1 for colormap
        img_normalized = img / self.max_iter

        # Apply colormap to get RGB
        img_colored = self.cmap(img_normalized)

        # Convert to uint8 RGB (drop alpha channel)
        img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)

        return img_rgb

    def generate_raw(self, c, xmin, xmax, ymin, ymax) -> np.ndarray:
        # Grayscale uint8 for analysis - avoids colormap distortion.
        img = self._compute(c, xmin, xmax, ymin, ymax)
        return (img / self.max_iter * 255).astype(np.uint8)
