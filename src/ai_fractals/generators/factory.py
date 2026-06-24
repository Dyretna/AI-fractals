# ai_fractals/generators/factory.py

from .mandelbrot import MandelbrotCPU, MandelbrotGPU

GENERATOR_TABLE = {
    "mandelbrot": (MandelbrotGPU, MandelbrotCPU),
}


def create_generator(
    fractal_type: str,
    *,
    width: int,
    height: int,
    max_iter: int,
    colormap: str,
    use_supersampling: bool,
    use_gpu: bool = False,
):
    if fractal_type not in GENERATOR_TABLE:
        raise ValueError(f"Unknown fractal type: {fractal_type}")

    GPUClass, CPUClass = GENERATOR_TABLE[fractal_type]

    cls = GPUClass if use_gpu else CPUClass

    return cls(
        width=width,
        height=height,
        max_iter=max_iter,
        colormap=colormap,
        use_supersampling=use_supersampling,
    )
