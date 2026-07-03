from .gans_interface import CriticBase, GeneratorBase
from .self_supervised_cnn import SelfSupervisedCNN
from .shoreline_vae import ShorelineVAE
from .wgan_gp import WganGpCritic, WganGpGenerator

__all__ = [
    "CriticBase",
    "GeneratorBase",
    "SelfSupervisedCNN",
    "ShorelineVAE",
    "WganGpCritic",
    "WganGpGenerator",
]
