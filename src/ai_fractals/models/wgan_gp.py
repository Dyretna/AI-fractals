"""
WGAN-GP generator and critic following Gulrajani et al. (2017),
"Improved Training of Wasserstein GANs".

This module implements lightweight DCGAN-style architectures adapted for
WGAN-GP stability:

- Generator:
    Uses ConvTranspose2d + BatchNorm + ReLU.
    Produces 128*128 RGB images from a latent vector z.
    Architecture mirrors the stable upsampling stack used in Gulrajani et al.

- Critic:
    Uses Conv2d + LeakyReLU without BatchNorm (as recommended in the paper).
    Outputs a scalar Wasserstein score instead of a probability.
    Architecture is intentionally lightweight to avoid critic overpowering.

These networks are designed to be trained with:
    - Adam( lr=1e-4, betas=(0.0, 0.9) )
    - Gradient penalty (λ = 10 in paper, often 5-10 in practice)
    - n_critic > 1 (paper uses 5, 3 is stable for 128*128)

The goal is to remain faithful to the training dynamics described in
Gulrajani et al. (2017) while supporting 128*128 fractal image synthesis.
"""

import torch.nn as nn

# Kernel, Stride, Padding
K, S, P = 4, 2, 1


class WganGpGenerator(nn.Module):
    """
    Generator network for WGAN-GP.

    Components:
    - Input:
        A latent noise vector z sampled from N(0, 1). This vector represents the
        seed for generating a fractal-like RGB image.

    - Upsampling stack:
        A sequence of transposed convolutions (ConvTranspose2d) that progressively
        increase spatial resolution while reducing feature dimensionality.
        Each block uses:
            - ConvTranspose2d
            - BatchNorm2d
            - ReLU activation

    - Output layer:
        A final convolution that maps features to 3 RGB channels, followed by a
        Tanh activation to produce pixel values in [-1, 1].

    Methods:
    - forward(z): returns a generated RGB image tensor.
    """

    def __init__(self, z_dim=128, img_channels=3, feature_maps=64):
        super().__init__()

        self.net = nn.Sequential(
            # z -> 4x4
            nn.ConvTranspose2d(z_dim, feature_maps * 16, 4, 1, 0),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, K, S, P),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, K, S, P),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, K, S, P),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, K, S, P),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(feature_maps, img_channels, K, S, P),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class WganGpCritic(nn.Module):
    """
    Critic network for WGAN-GP.

    Components:
    - Downsampling stack:
        A sequence of convolutional layers that progressively reduce spatial
        resolution while increasing feature dimensionality. Unlike a classifier,
        the critic does not use normalization layers (BatchNorm is avoided in
        WGAN-GP for stability).

        Each block uses:
            - Conv2d
            - LeakyReLU activation(0.2)
            (no BatchNorm — critical for WGAN-GP stability)

    - Output layer:
        A final linear layer that maps the flattened feature vector to a single
        scalar Wasserstein score.

    Methods:
    - forward(x): returns a scalar score for each input image.
    """

    def __init__(self, img_channels=3, feature_maps=64):
        super().__init__()

        self.net = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(img_channels, feature_maps, K, S, P),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(feature_maps, feature_maps * 2, K, S, P),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, K, S, P),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, K, S, P),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(feature_maps * 8, feature_maps * 16, K, S, P),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4x4 spatial, feature_maps * 16 channels
        self.fc = nn.Linear(feature_maps * 16 * 4 * 4, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
