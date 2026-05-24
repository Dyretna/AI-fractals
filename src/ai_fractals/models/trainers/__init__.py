"""Trainers module."""

from .base import BaseTrainer
from .cnn_trainer import CNNTrainer
from .gan_trainer import GANTrainer

__all__ = ["BaseTrainer", "GANTrainer", "CNNTrainer"]
