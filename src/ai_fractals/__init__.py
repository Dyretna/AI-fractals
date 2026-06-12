"""AI-Fractals: AI-Enhanced Fractal Geometry

This package combines traditional fractal mathematics with machine learning
to generate and analyze fractal patterns.

Based on: Youvan (2024) "AI-Enhanced Fractal Geometry"

Modules:
    - generators: Fractal generation (Mandelbrot, Julia sets) - Section 4.1
    - analysis:   Quality metrics (fractal dimension, entropy, statistics) - Section 6
    - processing: Shoreline extraction via Canny - Section 4.2
    - io:         Image and metadata saving
"""

__version__ = "0.1.0"

from .data import FractalDatasetBuilder, ShorelineBatchExtractor
from .generators import BaseFractalGenerator, JuliaGenerator, MandelbrotGenerator

__all__ = [
    # Generators
    "BaseFractalGenerator",
    "MandelbrotGenerator",
    "JuliaGenerator",
    # Data
    "FractalDatasetBuilder",
    "ShorelineBatchExtractor",
]
