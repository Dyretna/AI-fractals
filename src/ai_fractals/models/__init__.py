from .cond_wgan_gp import WganGpCritic, WganGpGenerator
from .gans_interface import CriticBase, GeneratorBase
from .self_supervised_cnn import SelfSupervisedCNN
from .shoreline_vae import ShorelineVAE

__all__ = [
    "CriticBase",
    "GeneratorBase",
    "SelfSupervisedCNN",
    "ShorelineVAE",
    "WganGpCritic",
    "WganGpGenerator",
]
