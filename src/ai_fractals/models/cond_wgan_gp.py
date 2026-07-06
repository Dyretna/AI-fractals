"""
Conditional WGAN-GP generator and critic following Gulrajani et al. (2017),
"Improved Training of Wasserstein GANs", extended with embedding conditioning.

This module implements lightweight DCGAN-style architectures adapted for
WGAN-GP stability and conditional generation:

- Generator:
    Uses ConvTranspose2d + BatchNorm + ReLU.
    Produces 128*128 RGB images from a concatenated latent vector:
        [noise z || conditioning embedding e].
    Architecture mirrors the stable upsampling stack used in Gulrajani et al.,
    while allowing geometry control via the embedding.

- Critic:
    Uses Conv2d + LeakyReLU without BatchNorm (as recommended in the paper).
    Receives both the RGB image and the conditioning embedding.
    The embedding is concatenated with the flattened feature vector before
    the final linear layer, so the critic can learn a joint Wasserstein score
    over (image, embedding) pairs.
    Outputs a scalar Wasserstein score instead of a probability.
    Architecture is intentionally lightweight to avoid critic overpowering.

These networks are designed to be trained with:
    - Adam( lr=1e-4, betas=(0.0, 0.9) )
    - Gradient penalty (lambda = 10 in paper, often 5-10 in practice)
    - n_critic > 1 (paper uses 5, 3 is stable for 128*128)

The goal is to remain faithful to the training dynamics described in
Gulrajani et al. (2017) while supporting 128*128 fractal image synthesis
conditioned on CNN embeddings.
"""

import torch
import torch.nn as nn

# Kernel, Stride, Padding
K, S, P = 4, 2, 1


class WganGpGenerator(nn.Module):
    """
    Conditional generator network for WGAN-GP.

    Components:
    - Input:
        Two vectors:
            - z: latent noise vector sampled from N(0, 1), controls style.
            - e: conditioning embedding vector (e.g. CNN geometry embedding).
        These are concatenated along the feature dimension to form a single
        latent vector [z || e] that seeds the fractal-like RGB image.

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
    - forward(z, e): returns a generated RGB image tensor conditioned on e.
    """

    def __init__(self, noise_dim=128, embed_dim=256, img_channels=3, feature_maps=64):
        super().__init__()

        z_dim = noise_dim + embed_dim

        self.net = nn.Sequential(
            # [z || e] -> 4x4
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

        self.noise_dim = noise_dim
        self.embed_dim = embed_dim

    def forward(self, z, e):
        """
        Forward pass for the conditional generator.

        Args:
            z: tensor of shape (batch_size, noise_dim)
            e: tensor of shape (batch_size, embed_dim)

        Returns:
            Generated RGB image tensor of shape (batch_size, 3, 128, 128),
            conditioned on the embedding e.
        """
        x = torch.cat([z, e], dim=1)
        return self.net(x.view(x.size(0), x.size(1), 1, 1))


class WganGpCritic(nn.Module):
    """
    Conditional critic network for WGAN-GP.

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

    - Conditioning path:
        The conditioning embedding e (e.g. CNN geometry embedding) is concatenated
        with the flattened feature vector before the final linear layer. This
        allows the critic to learn a joint Wasserstein score over (image, e)
        pairs, enforcing consistency between RGB content and geometry embedding.

    - Output layer:
        A final linear layer that maps the concatenated feature + embedding
        vector to a single scalar Wasserstein score.

    Methods:
    - forward(x, e): returns a scalar score for each (image, embedding) pair.
    """

    def __init__(self, img_channels=3, feature_maps=64, embed_dim=256):
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
        feat_dim = feature_maps * 16 * 4 * 4
        self.fc = nn.Linear(feat_dim + embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, x, e):
        """
        Forward pass for the conditional critic.

        Args:
            x: tensor of shape (batch_size, img_channels, 128, 128)
            e: tensor of shape (batch_size, embed_dim)

        Returns:
            Scalar Wasserstein score tensor of shape (batch_size, 1),
            representing the critic's assessment of (x, e) pairs.
        """
        h = self.net(x)
        h = h.view(h.size(0), -1)
        h_cond = torch.cat([h, e], dim=1)
        return self.fc(h_cond)
