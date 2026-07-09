"""
Self-supervised training loop for SimCLR-style contrastive learning.

This trainer is designed for models like SelfSupervisedCNN, which learn
geometry-based embeddings from shoreline images without labels. The trainer
generates two augmented views of each input image, computes embeddings for both,
and optimizes the model using the NT-Xent contrastive loss.

The resulting encoder can be used for:
- K-Means clustering of fractal shapes
- Unsupervised labeling for conditional GANs
- Latent space exploration
- Similarity search and geometry-aware analysis
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# NT-Xent Loss (SimCLR)
# ---------------------------------------------------------------------------


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-Scaled Cross Entropy Loss (SimCLR).

    Encourages embeddings of two augmentations of the same image to be similar,
    while pushing apart embeddings from different images in the batch.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)

        # Cosine similarity matrix
        reps = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.matmul(reps, reps.T)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)

        # Positive pairs: diagonal of z1·z2 and z2·z1
        positives = torch.sum(z1 * z2, dim=-1)
        positives = torch.cat([positives, positives], dim=0)

        # Scale by temperature
        logits = sim_matrix / self.temperature
        positives = positives / self.temperature

        # Labels: each positive pair is at index 0 in its row
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)

        # Cross entropy loss
        loss = F.cross_entropy(
            torch.cat([positives.unsqueeze(1), logits], dim=1), labels
        )
        return loss


# ---------------------------------------------------------------------------
# Self-Supervised Trainer
# ---------------------------------------------------------------------------


class SelfSupervisedTrainer:
    """
    Trainer for SimCLR-style self-supervised learning.

    Workflow:
    - For each batch, generate two augmented views (x1, x2)
    - Compute embeddings z1 = model(x1), z2 = model(x2)
    - Compute NT-Xent contrastive loss
    - Optimize encoder + projection head

    After training:
    - Use model.embed(x) to extract geometry embeddings
    - Run K-Means on embeddings to obtain fractal shape clusters
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        augment_fn,
        lr: float = 1e-3,
        epochs: int = 50,
        temperature: float = 0.5,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.augment_fn = augment_fn

        self.lr = lr
        self.epochs = epochs
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = NTXentLoss(temperature)

        # logger
        self.log = logging.getLogger(__name__)
        self.log.info(self)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for batch in self.dataloader:
                batch = batch.to(self.device)

                # Two augmented views
                x1 = self.augment_fn(batch)
                x2 = self.augment_fn(batch)

                # Forward pass
                z1 = self.model(x1)
                z2 = self.model(x2)

                # Contrastive loss
                loss = self.loss_fn(z1, z2)

                # Backprop
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            self.log.info(f"Epoch {epoch + 1}/{self.epochs} - loss={avg_loss:.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def __str__(self):
        rows = [
            "SelfSupervisedTrainer",
            f"  model:   {self.model.__class__.__name__}",
            f"  device:  {self.device}",
            f"  lr:      {self.lr}",
            f"  epochs:  {self.epochs}",
            "  loss_fn: NTXentLoss",
            f"  batches: {len(self.dataloader)}",
        ]
        return "\n".join(rows)
