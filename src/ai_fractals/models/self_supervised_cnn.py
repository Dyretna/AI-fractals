"""
Self-supervised CNN for learning geometry-based embeddings from shoreline images.

This model implements a lightweight SimCLR-style architecture designed to learn
fractal geometry without labels. It takes a shoreline image, passes it through a
convolutional encoder to extract structural features, and then maps these
features into a normalized embedding space using a projection head.

The embeddings produced by this model can be used for:
- K-Means clustering of fractal shapes (spirals, bulbs, filaments, etc.)
- Unsupervised labeling for conditional GAN training
- Latent space exploration and similarity search
- Geometry-aware analysis of generated fractals

Unlike an autoencoder, this model does NOT reconstruct images. It learns
representations purely through contrastive learning (SimCLR), where two
augmentations of the same shoreline are encouraged to have similar embeddings.
"""

import torch.nn as nn
import torch.nn.functional as F


class SelfSupervisedCNN(nn.Module):
    """
    A self-supervised convolutional encoder with a projection head (SimCLR-style).

    Components:
    - Encoder:
        A small ResNet-like stack of convolutional layers that progressively
        downsamples the input shoreline image while extracting geometric features.
        The final AdaptiveAvgPool2d layer collapses spatial dimensions to produce
        a compact feature vector (256-dimensional).

    - Projection head:
        A 2-layer MLP that maps encoder features into a contrastive embedding
        space. This head is used only during training. The output is L2-normalized
        to place embeddings on the unit hypersphere, which stabilizes contrastive
        loss and makes cosine similarity meaningful.

    Methods:
    - forward(x): returns the projected embedding (used during training)
    - embed(x): returns the raw encoder embedding (used for clustering / GAN labels)
    """

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 16x16 -> 1x1
        )

        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        # Full SimCLR forward pass: encoder -> projector -> normalized embedding
        h = self.encoder(x).view(x.size(0), -1)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def embed(self, x):
        # Encoder-only embedding (used for clustering and GAN conditioning)
        h = self.encoder(x).view(x.size(0), -1)
        return F.normalize(h, dim=1)
