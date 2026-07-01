from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class GeneratorBase(nn.Module, ABC):
    """
    Abstract base class for all generator models.

    Requirements:
    - forward(z): must take a latent noise tensor of shape (B, z_dim)
      and return a batch of generated images of shape (B, C, H, W).
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass


class CriticBase(nn.Module, ABC):
    """
    Abstract base class for all critic/discriminator models.

    Requirements:
    - forward(x): must take a batch of images of shape (B, C, H, W)
      and return a batch of scalar scores of shape (B, 1).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
