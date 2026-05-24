"""Architecture module."""

from .cnn import build_cnn
from .gan import build_discriminator, build_generator

__all__ = ["build_cnn", "build_generator", "build_discriminator"]
