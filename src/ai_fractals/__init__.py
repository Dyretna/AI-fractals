"""AI-Fractals: AI-Enhanced Fractal Geometry

This package combines traditional fractal mathematics with machine learning
to generate and analyze fractal patterns.

Modules:
    - generators: Fractal generation (Mandelbrot, Julia sets)
    - io: Image saving and pipeline management
    - processing: Shoreline extraction and analysis
    - models: Neural networks for fractal learning (CNNs, GANs)
    - hardware_config: Dynamic GPU/CPU configuration
"""

__version__ = "0.1.0"

# Import from subpackages for convenient access
from .generators import FractalGenerator, JuliaGenerator, MandelbrotGenerator
from .hardware_config import (
    HardwareConfig,
    check_gpu_available,
    get_hardware_config,
    get_optimal_batch_size,
    setup_hardware,
)
from .io import FractalSaverPipeline, ImageSaver, OpenCVImageSaver, PltImageSaver
from .models import (
    CNNConfig,
    CNNTrainer,
    GANConfig,
    GANTrainer,
    TrainingConfig,
    build_cnn,
    build_discriminator,
    build_generator,
)
from .processing import ShorelineProcessor

__all__ = [
    # Generators
    "FractalGenerator",
    "MandelbrotGenerator",
    "JuliaGenerator",
    # I/O
    "ImageSaver",
    "PltImageSaver",
    "OpenCVImageSaver",
    "FractalSaverPipeline",
    # Processing
    "ShorelineProcessor",
    # Hardware Configuration
    "HardwareConfig",
    "setup_hardware",
    "get_hardware_config",
    "check_gpu_available",
    "get_optimal_batch_size",
    # Models - Architectures
    "build_cnn",
    "build_generator",
    "build_discriminator",
    # Models - Configs
    "TrainingConfig",
    "GANConfig",
    "CNNConfig",
    # Models - Trainers
    "GANTrainer",
    "CNNTrainer",
]
