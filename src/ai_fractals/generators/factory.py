# ai_fractals/generators/factory.py

import random

import torch

from .julia import JuliaCPU, JuliaGPU
from .mandelbrot import MandelbrotCPU, MandelbrotGPU


class FractalState:
    def __init__(self, fractal_type, params):
        self.type = fractal_type
        self.params = params


# ----------------------------------------------------------------
# Metadata (parameters)
# ----------------------------------------------------------------
FRACTAL_META = {
    "mandelbrot": {"params": lambda: {}},
    "julia": {
        "params": lambda: {
            "c": complex(  # known intervall for Julia
                random.uniform(-0.7, 0.3), random.uniform(-0.6, 0.6)
            )
        },
    },
}

# ----------------------------------------------------------------
# Generator classes
# ----------------------------------------------------------------

GENERATOR_TABLE = {
    "mandelbrot": (MandelbrotGPU, MandelbrotCPU),
    "julia": (JuliaGPU, JuliaCPU),
}

# ----------------------------------------------------------------
# factory API
# ----------------------------------------------------------------


def create_fractal_state(fractal_type: str) -> FractalState:
    entry = FRACTAL_META[fractal_type]
    params = entry["params"]()
    return FractalState(fractal_type, params)


def get_default_bounds(fractal_type: str, state: FractalState):
    if fractal_type == "mandelbrot":
        return (-2.0, 1.0, -1.5, 1.5)

    if fractal_type == "julia":
        c = state.params["c"]
        cx, cy = c.real, c.imag
        size = 1.5
        return (cx - size, cx + size, cy - size, cy + size)


def create_generator(
    fractal_type,
    width,
    height,
    max_iter,
    colormap,
    log_level,
    use_supersampling,
    state: FractalState,
):
    if fractal_type not in FRACTAL_META:
        raise ValueError(f"Unknown fractal type: {fractal_type}")

    GPUClass, CPUClass = GENERATOR_TABLE[fractal_type]
    cls = GPUClass if torch.cuda.is_available() else CPUClass

    return cls(
        width=width,
        height=height,
        max_iter=max_iter,
        colormap=colormap,
        log_level=log_level,
        use_supersampling=use_supersampling,
        state=state,
    )
