"""
VAE training loop for geometry-aware fractal shoreline modeling.

This trainer is designed for ShorelineVAE, a conditional variational
autoencoder that learns a continuous latent space over fractal shoreline
geometry. The model receives both the shoreline mask and its associated
fractal bounds, enabling position-aware latent representations.

Workflow:
- For each batch, load shoreline images and normalized bounds
- Forward pass through the VAE: recon, mu, logvar
- Compute VAE loss (reconstruction + KL divergence)
- Optimize encoder + decoder jointly

After training:
- Use model.decode(z, bounds) to generate synthetic shorelines
- Use model.encode(x, bounds) to obtain latent vectors
- Feed synthetic shorelines into the frozen CNN to obtain geometry embeddings
- Use embeddings as conditioning for GAN-based RGB fractal generation
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# VAE Loss (Reconstruction + KL Divergence)
# ---------------------------------------------------------------------------


def vae_loss(recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Computes the standard VAE loss:
    - Reconstruction loss (MSE)
    - KL divergence between q(z|x) and N(0, I)
    """
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


# ---------------------------------------------------------------------------
# VAE Trainer
# ---------------------------------------------------------------------------


class VAETrainer:
    """
    Trainer for conditional ShorelineVAE.

    Workflow:
    - For each batch, load shoreline images and bounds
    - Forward pass: recon, mu, logvar = model(x, bounds)
    - Compute VAE loss
    - Backprop + optimize

    After training:
    - model.decode(z, bounds) generates synthetic geometry
    - model.encode(x, bounds) provides latent vectors
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        lr: float = 1e-4,
        epochs: int = 50,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader

        self.lr = lr
        self.epochs = epochs
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        # logger
        self.log = logging.getLogger(__name__)
        self.log.info(self)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for batch in self.dataloader:
                # Expect batch = (shoreline, bounds)
                x, bounds = batch
                x = x.to(self.device)
                bounds = bounds.to(self.device)

                # Forward pass
                recon, mu, logvar = self.model(x, bounds)

                # VAE loss
                loss = vae_loss(recon, x, mu, logvar)

                # Backprop
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            self.log.info(f"Epoch {epoch + 1}/{self.epochs} - loss={avg_loss:.4f}")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def __str__(self):
        rows = [
            "VAETrainer",
            f"  model:   {self.model.__class__.__name__}",
            f"  device:  {self.device}",
            f"  lr:      {self.lr}",
            f"  epochs:  {self.epochs}",
            f"  batches: {len(self.dataloader)}",
            "  loss_fn: VAE (reconstruction + KL)",
        ]
        return "\n".join(rows)
