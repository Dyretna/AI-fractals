"""
Module for training ML-models: CNNs, VAEs and GANS

under construction!
"""

from .autoencoder_trainer import AutoencoderTrainer
from .early_stopping import EarlyStopping
from .self_supervised_trainer import SelfSupervisedTrainer
from .wgan_pg_trainer import WganGpTrainer

__all__ = [
    "AutoencoderTrainer",
    "EarlyStopping",
    "SelfSupervisedTrainer",
    "WganGpTrainer",
]
