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

    def generate(self, xmin, xmax, ymin, ymax):
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

        # Normalize to 0-1 for colormap
        img_normalized = img / self.max_iter

        # Apply colormap to get RGB
        img_colored = self.cmap(img_normalized)

        # Convert to uint8 RGB (drop alpha channel)
        img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)

        return img_rgb


class JuliaGenerator(FractalGenerator):
    def __init__(self, width=800, height=600, max_iter=256, colormap="twilight"):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.colormap = colormap
        self.cmap = plt.get_cmap(colormap)

    def generate(self, c, xmin, xmax, ymin, ymax):
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        img = np.zeros(Z.shape, dtype=int)
        for _ in range(self.max_iter):
            mask = np.abs(Z) < 2
            Z[mask] = Z[mask] * Z[mask] + c
            img += mask

        # Normalize to 0-1 for colormap
        img_normalized = img / self.max_iter

        # Apply colormap to get RGB
        img_colored = self.cmap(img_normalized)

        # Convert to uint8 RGB (drop alpha channel)
        img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)

        return img_rgb
