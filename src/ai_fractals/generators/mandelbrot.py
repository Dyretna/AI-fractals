# ai_fractals/generators/mandelbrot.py
import numpy as np
import torch

from .base import BaseFractalGenerator


class MandelbrotCPU(BaseFractalGenerator):
    fractal_type = "mandelbrot"

    def default_bounds(self) -> tuple[float, float, float, float]:
        return (-2.0, 1.0, -1.5, 1.5)

    def _compute(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> np.ndarray:
        xmin, xmax, ymin, ymax = self._match_aspect(xmin, xmax, ymin, ymax)

        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        img = np.zeros(C.shape, dtype=np.int32)

        for i in range(1, self.max_iter):
            mask = img == 0
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            escaped = (np.abs(Z) >= 2) & (img == 0)
            img[escaped] = i

        img[img == 0] = self.max_iter
        return img


class MandelbrotGPU(BaseFractalGenerator):
    fractal_type = "mandelbrot"

    def default_bounds(self) -> tuple[float, float, float, float]:
        return (-2.0, 1.0, -1.5, 1.5)

    def _compute(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> np.ndarray:
        xmin, xmax, ymin, ymax = self._match_aspect(xmin, xmax, ymin, ymax)

        # double precision coordinates
        x = torch.linspace(
            xmin, xmax, self.width, device=self.device, dtype=torch.float64
        )
        y = torch.linspace(
            ymin, ymax, self.height, device=self.device, dtype=torch.float64
        )
        X, Y = torch.meshgrid(x, y, indexing="xy")

        # complex constant
        C = X + 1j * Y
        Z = torch.zeros_like(C, dtype=torch.complex128)

        img = torch.zeros(C.shape, dtype=torch.int32, device=self.device)

        for i in range(1, self.max_iter):
            mask = img == 0
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            escaped = (Z.real * Z.real + Z.imag * Z.imag >= 4) & (img == 0)
            img[escaped] = i

        img[img == 0] = self.max_iter
        return img.cpu().numpy()
