"""
Module for training ML-models: CNNs, VAEs and GANS

under construction!
"""

from .early_stopping import EarlyStopping
from .self_supervised_trainer import SelfSupervisedTrainer
from .shoreline_vae_trainer import VAETrainer
from .wgan_pg_trainer import WganGpTrainer

__all__ = [
    "EarlyStopping",
    "SelfSupervisedTrainer",
    "VAETrainer",
    "WganGpTrainer",
]
