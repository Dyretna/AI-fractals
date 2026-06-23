"""
ai_fractals/models/autoencoder.py

Defines the AutoEncoder (AE) used for learning geometry-based embeddings
from shoreline images.
"""

import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 256x256
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def embed(self, x):
        return self.encoder(x)

    def __str__(self):
        rows = [
            "AutoEncoder",
            f"  encoder_layers: {len(self.encoder)}",
            f"  decoder_layers: {len(self.decoder)}",
            "  embedding_shape: 256x16x16",
        ]
        return "\n".join(rows)
