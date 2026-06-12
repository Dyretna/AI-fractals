# ai_fractals/generators/julia.py
import numpy as np
import torch

from .base import BaseFractalGenerator


class JuliaCPU(BaseFractalGenerator):
    def _compute(self, c, xmin, xmax, ymin, ymax):
        device = self.device
        xmin, xmax, ymin, ymax = self.match_aspect(xmin, xmax, ymin, ymax)

        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        img = np.zeros(Z.shape, dtype=np.int32)

        self.log.info(f"using device: {device}")
        self.log.info(f"Z dtype: ({Z.dtype}), device: {Z.device}")

        for i in range(1, self.max_iter):
            mask = img == 0
            Z[mask] = Z[mask] * Z[mask] + c
            escaped = (np.abs(Z) >= 2) & (img == 0)
            img[escaped] = i

        img[img == 0] = self.max_iter
        return img

    def generate(self, c, xmin, xmax, ymin, ymax):
        return self.normalize_RGB(self._compute(c, xmin, xmax, ymin, ymax))

    def generate_raw(self, c, xmin, xmax, ymin, ymax):
        img = self._compute(c, xmin, xmax, ymin, ymax)
        return (img / self.max_iter * 255).astype(np.uint8)


class JuliaGPU(BaseFractalGenerator):
    def _compute(self, c, xmin, xmax, ymin, ymax):
        device = self.device

        xmin, xmax, ymin, ymax = self.match_aspect(xmin, xmax, ymin, ymax)

        # double precision coordinates
        x = torch.linspace(xmin, xmax, self.width, device=device, dtype=torch.float64)
        y = torch.linspace(ymin, ymax, self.height, device=device, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="xy")

        # complex128 grid
        Z = (X + 1j * Y).to(torch.complex128)

        # complex128 constant
        c = torch.tensor(c, device=device, dtype=torch.complex128)

        img = torch.zeros(Z.shape, dtype=torch.int32, device=device)

        self.log.info(f"using device: {device}")
        self.log.info(f"Z dtype: ({Z.dtype}), device: {Z.device}")

        for i in range(1, self.max_iter):
            mask = img == 0
            Z[mask] = Z[mask] * Z[mask] + c
            escaped = (Z.real * Z.real + Z.imag * Z.imag >= 4) & (img == 0)
            img[escaped] = i

        img[img == 0] = self.max_iter
        return img.cpu().numpy()

    def generate(self, c, xmin, xmax, ymin, ymax):
        return self.normalize_RGB(self._compute(c, xmin, xmax, ymin, ymax))

    def generate_raw(self, c, xmin, xmax, ymin, ymax):
        img = self._compute(c, xmin, xmax, ymin, ymax)
        return (img / self.max_iter * 255).astype(np.uint8)
