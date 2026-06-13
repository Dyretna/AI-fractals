"""Fractal generators module.

This module contains classes for generating various fractal types
including Mandelbrot and Julia sets.
"""

from .base import BaseFractalGenerator
from .factory import create_fractal_state, create_generator, get_default_bounds
from .julia import JuliaCPU, JuliaGPU
from .mandelbrot import MandelbrotCPU, MandelbrotGPU

__all__ = [
    "BaseFractalGenerator",
    "MandelbrotCPU",
    "MandelbrotGPU",
    "JuliaCPU",
    "JuliaGPU",
    # factory
    "create_generator",
    "create_fractal_state",
    "get_default_bounds",
]
