"""
ShorelineVAE

This module implements a Variational Autoencoder (VAE) designed
specifically for fractal shoreline images. The architecture is tuned
for geometric data rather than texture-rich RGB images, and every
layer size is chosen to preserve global fractal structure while still
allowing meaningful compression into a continuous latent space.

ENCODER DESIGN
--------------
Input images are 256x256 single-channel shoreline masks. The encoder
reduces spatial resolution in four steps:

    256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16

Each step uses Conv2d(kernel=4, stride=2, padding=1), which performs
an halving of width and height. This downsampling schedule is
intentional:

    - Shorelines contain global geometric structure (branches, bulbs,
    filaments). If resolution is reduced too aggressively, these
    structures collapse and the latent space becomes meaningless.

    - If resolution is reduced too slowly, the bottleneck becomes too
    large and interpolation in latent space loses semantic coherence.

The chosen 256->16 progression preserves the fractal's global shape
while still compressing it enough for a VAE to learn a smooth manifold.

CHANNEL DEPTH
-------------
Feature depth increases at each stage:

    32 -> 64 -> 128 -> 256

This reflects the nature of shoreline geometry:

    - Early layers capture local contour details.
    - Mid layers capture branching and filament patterns.
    - Deep layers capture global fractal topology.

Shorelines have high geometric complexity but low texture complexity,
so increasing channel depth is more important than preserving color
information (there is none). The final encoder output is a 256x16x16
tensor containing rich geometric features.

BOTTLENECK
----------
The encoder flattens the 256x16x16 tensor (65,536 values) and maps it
to (mu, logvar) in a latent_dim-dimensional space. This bottleneck is
the key to interpolation:

    - Large enough to preserve global fractal structure.
    - Small enough to force the model to learn a continuous manifold.

DECODER DESIGN
--------------
The decoder mirrors the encoder using ConvTranspose2d layers:

    16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256

This symmetry ensures that synthetic shorelines have the same visual
style and geometric characteristics as real ones. The final Sigmoid
activation produces normalized 0-1 masks, matching the input format.

PIPELINE ROLE
-------------
The VAE is trained independently from the CNN and GAN. After training:

    - The decoder is used to generate synthetic shorelines.
    - These synthetic shorelines are encoded by the frozen CNN.
    - The resulting embeddings are fed into the GAN as geometry control.

This architecture ensures that synthetic geometry lies in the same
embedding manifold as real geometry, enabling smooth interpolation
between fractal types.

Conditioning on bounds ensures that synthetic geometry lies in the same
embedding manifold as real geometry, enabling smooth interpolation
between fractal types and stable GAN conditioning.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Kernel, Stride, Padding
K, S, P = 4, 2, 1


class ShorelineVAE(nn.Module):
    def __init__(self, latent_dim: int = 64, cond_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Encoder CNN
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, K, S, P),
            nn.ReLU(),
            nn.Conv2d(32, 64, K, S, P),
            nn.ReLU(),
            nn.Conv2d(64, 128, K, S, P),
            nn.ReLU(),
            nn.Conv2d(128, 256, K, S, P),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Conditional bottleneck
        self.fc_mu = nn.Linear(256 * 16 * 16 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16 + cond_dim, latent_dim)

        # Decoder FC (latent + bounds)
        self.fc_dec = nn.Linear(latent_dim + cond_dim, 256 * 16 * 16)

        # Decoder CNN
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, K, S, P),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, K, S, P),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, K, S, P),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, K, S, P),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor, bounds: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.enc(x)
        h = torch.cat([h, bounds], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, bounds: Tensor) -> Tensor:
        z = torch.cat([z, bounds], dim=1)
        h = self.fc_dec(z)
        h = h.view(-1, 256, 16, 16)
        return self.dec(h)

    def forward(self, x: Tensor, bounds: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x, bounds)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, bounds)
        return recon, mu, logvar
