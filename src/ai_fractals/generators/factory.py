# ai_fractals/generators/factory.py

import torch

from .julia import JuliaCPU, JuliaGPU
from .mandelbrot import MandelbrotCPU, MandelbrotGPU

GENERATOR_TABLE = {
    "mandelbrot": (MandelbrotGPU, MandelbrotCPU),
    "julia": (JuliaGPU, JuliaCPU),
}


def create_generator(
    fractal_type, width, height, max_iter, colormap, log_level, use_supersampling
):
    has_cuda = torch.cuda.is_available()

    if fractal_type not in GENERATOR_TABLE:
        raise ValueError(f"Unknown fractal type: {fractal_type}")

    GPUClass, CPUClass = GENERATOR_TABLE[fractal_type]
    cls = GPUClass if has_cuda else CPUClass

    return cls(width, height, max_iter, colormap, log_level, use_supersampling)
