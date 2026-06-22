"""
ai_fractals/training/autoencoder_trainer.py

Trainer for autoencoder-style models.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .early_stopping import EarlyStopping


class AutoencoderTrainer:
    def __init__(
        self,
        model,
        dataloader,
        lr=1e-3,
        epochs=20,
        device=None,
        early_stop: EarlyStopping | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader

        self.lr = lr
        self.epochs = epochs
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.early_stop = early_stop

    def train(self):
        for epoch in range(self.epochs):
            for batch in self.dataloader:
                batch = batch.to(self.device)

                self.opt.zero_grad()
                out = self.model(batch)
                loss = self.loss_fn(out, batch)
                loss.backward()
                self.opt.step()

            train_loss = loss.item()
            print(f"Epoch {epoch + 1}: loss={train_loss:.4f}")

            # ---- EARLY STOPPING ----
            if self.early_stop is not None:
                if self.early_stop.check(train_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def __str__(self):
        rows = [
            "AutoencoderTrainer",
            f"  model:   {self.model.__class__.__name__}",
            f"  device:  {self.device}",
            f"  lr:      {self.lr}",
            f"  epochs:  {self.epochs}",
            f"  loss_fn: {self.loss_fn.__class__.__name__}",
            f"  batches: {len(self.dataloader)}",
            f"  early_stop: {self.early_stop if self.early_stop else None}",
        ]
        return "\n".join(rows)
