# ai_fractals/generators/factory.py

from .mandelbrot import MandelbrotCPU, MandelbrotGPU

GENERATOR_TABLE = {
    "mandelbrot": (MandelbrotGPU, MandelbrotCPU),
}


def format_available_types():
    rows = ["Available fractal generator types:"]
    for str_key, (GPUClass, CPUClass) in GENERATOR_TABLE.items():
        rows.append(f"{str_key:10} : GPU={GPUClass.__name__}, CPU={CPUClass.__name__}")
    return "    \n".join(rows)


def create_generator(
    *,
    fractal_type: str,
    width: int,
    height: int,
    max_iter: int,
    colormap: str,
    use_supersampling: bool,
    device: str,
):
    if fractal_type not in GENERATOR_TABLE:
        raise ValueError(
            f"Unregistered fractal generator type '{fractal_type}',"
            f"{format_available_types()}"
        )

    GPUClass, CPUClass = GENERATOR_TABLE[fractal_type]

    # get correct generator type according to device type
    match device:
        case "cpu":
            cls = CPUClass
        case "cuda":
            cls = GPUClass
        case _:
            raise ValueError(f"device must be 'cpu' or 'cuda', {device}")

    return cls(
        width=width,
        height=height,
        max_iter=max_iter,
        colormap=colormap,
        use_supersampling=use_supersampling,
    )
