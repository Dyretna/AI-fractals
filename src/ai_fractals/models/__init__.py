"""AI models for fractal generation and analysis.

This module contains neural network architectures, training
utilities, and configurations for learning fractal patterns.
"""

# Architectures
from .architectures import build_cnn, build_discriminator, build_generator

# Configs
from .configs import CNNConfig, GANConfig, TrainingConfig

# Data utilities
from .data import create_data_generator, datagen, datagen_no_aug

# Trainers
from .trainers import BaseTrainer, CNNTrainer, GANTrainer

__all__ = [
    # Architectures
    "build_cnn",
    "build_generator",
    "build_discriminator",
    # Configs
    "TrainingConfig",
    "GANConfig",
    "CNNConfig",
    # Data
    "create_data_generator",
    "datagen",
    "datagen_no_aug",
    # Trainers
    "BaseTrainer",
    "GANTrainer",
    "CNNTrainer",
]
