"""
Module for training ML-models: CNNs, VAEs and GANS

under construction!
"""

from .autoencoder_trainer import AutoencoderTrainer
from .early_stopping import EarlyStopping

__all__ = [
    "AutoencoderTrainer",
    "EarlyStopping",
]
