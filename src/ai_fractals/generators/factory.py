import torch

from .julia import JuliaCPU, JuliaGPU
from .mandelbrot import MandelbrotCPU, MandelbrotGPU


def create_generator(
    fractal_type, width, height, max_iter, colormap, log_level, use_supersampling
):
    if fractal_type == "mandelbrot":
        return (
            MandelbrotGPU(
                width, height, max_iter, colormap, log_level, use_supersampling
            )
            if torch.cuda.is_available()
            else MandelbrotCPU(
                width, height, max_iter, colormap, log_level, use_supersampling
            )
        )

    if fractal_type == "julia":
        return (
            JuliaGPU(width, height, max_iter, colormap, log_level, use_supersampling)
            if torch.cuda.is_available()
            else JuliaCPU(
                width, height, max_iter, colormap, log_level, use_supersampling
            )
        )

    raise ValueError(f"Unknown fractal type: {fractal_type}")
