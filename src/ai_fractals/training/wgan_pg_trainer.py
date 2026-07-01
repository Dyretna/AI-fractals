"""
WGAN-GP training loop for learning RGB fractal distributions.

This trainer optimizes a generator and critic using the Wasserstein loss with
gradient penalty. The critic estimates the Wasserstein distance between real and
generated fractal images, while the generator learns to produce samples that
maximize the critic's score.

Workflow:
- For each batch:
    - Update the critic multiple times:
        * Sample noise z
        * Generate fake images
        * Compute critic scores for real and fake
        * Compute gradient penalty
        * Optimize critic
    - Update the generator once:
        * Sample noise z
        * Generate fake images
        * Maximize critic score on fake samples

After training:
- Use generator(z) to sample new fractal images
- Use critic(x) to evaluate sample quality
"""

import copy
import os
import platform
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ..logging_config import get_logger
from ..models.gans_interface import CriticBase, GeneratorBase


class WganGpTrainer:
    """
    Trainer for WGAN-GP generative adversarial networks.

    Components:
    - generator: produces RGB fractal images from latent noise
    - critic: scores images using the Wasserstein distance
    - dataloader: provides batches of real RGB fractals
    - gradient penalty: enforces Lipschitz constraint for stable training

    Workflow:
    - Update critic n_critic times per batch
    - Update generator once
    - Track Wasserstein distance and losses
    """

    def __init__(
        self,
        generator: GeneratorBase,
        critic: CriticBase,
        dataloader: DataLoader,
        z_dim: int = 128,
        lr_g: float = 1e-4,
        lr_c: float = 1e-4,
        epochs: int = 50,
        n_critic: int = 3,
        lambda_gp: float = 5.0,
        ema_decay: float = 0.999,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        sample_dir: Optional[Path] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.dataloader = dataloader

        self.z_dim = z_dim
        self.lr_g = lr_g
        self.lr_c = lr_c
        self.epochs = epochs
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.ema_decay = ema_decay
        self.generator_ema = copy.deepcopy(self.generator).to(self.device)

        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c, betas=(0.0, 0.9))

        # set and create chpoint and sample directories
        self.sample_dir = sample_dir
        if self.sample_dir is not None:
            self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # logger
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = log_dir / "wgan_train.log"
        self.logger = get_logger(name="WGAN_trainer", redirect_path=self.train_log_path)

        self._system_specs()
        self.logger.info(self)

    def train(self):
        self.generator.train()
        self.critic.train()

        for epoch in range(self.epochs):
            for batch_idx, real in enumerate(self.dataloader):
                real = real.to(self.device)
                batch_size = real.size(0)

                # -------------------------
                # Critic updates
                # -------------------------
                for _ in range(self.n_critic):
                    z = torch.randn(batch_size, self.z_dim, device=self.device)
                    fake = self.generator(z)

                    critic_real = self.critic(real)
                    critic_fake = self.critic(fake.detach())

                    gp = gradient_penalty(
                        self.critic, real, fake, self.device, self.lambda_gp
                    )

                    loss_c = -(critic_real.mean() - critic_fake.mean()) + gp

                    self.opt_c.zero_grad()
                    loss_c.backward()
                    self.opt_c.step()

                # -------------------------
                # Generator update
                # -------------------------
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake = self.generator(z)
                critic_fake = self.critic(fake)

                loss_g = -critic_fake.mean()

                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()

                # Ema update
                self._update_ema()
                # Batch progress (var 10:e batch)
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} | "
                        f"Batch {batch_idx}/{len(self.dataloader)} | "
                        f"Critic loss: {loss_c.item():.4f} | "
                        f"Generator loss: {loss_g.item():.4f}"
                    )

            if self.sample_dir is not None:
                self._samples_per_epoch(epoch)

                # Checkpoint per epoch
            if self.checkpoint_dir is not None:
                self._checkpoint_per_epoch(epoch)

            self.logger.info(
                f"[EPOCH DONE] {epoch + 1}/{self.epochs} | "
                f"Critic loss: {loss_c.item():.4f} | "
                f"Generator loss: {loss_g.item():.4f}"
            )

    def save(self, gen_path, critic_path, ema_path=None):
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.critic.state_dict(), critic_path)
        if ema_path is not None:
            torch.save(self.generator_ema.state_dict(), ema_path)

    def _update_ema(self):
        with torch.no_grad():
            for p_ema, p in zip(
                self.generator_ema.parameters(), self.generator.parameters()
            ):
                p_ema.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

    def _samples_per_epoch(self, epoch):
        with torch.no_grad():
            z = torch.randn(16, self.z_dim, device=self.device)
            samples = self.generator_ema(z)
            save_image(
                samples,
                self.sample_dir / f"samples_epoch_{epoch + 1}.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )

    def _checkpoint_per_epoch(self, epoch):
        torch.save(
            {
                "epoch": epoch + 1,
                "gen": self.generator.state_dict(),
                "gen_ema": self.generator_ema.state_dict(),
                "critic": self.critic.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_c": self.opt_c.state_dict(),
            },
            self.checkpoint_dir / f"wgan_gp_epoch_{epoch + 1}.pt",
        )

    def _system_specs(self):
        header = "\n" + "=" * 50 + "\n System Specs \n" + "=" * 50
        rows = [header]

        rows.append(f"OS: {platform.system()} {platform.release()}")
        rows.append(f"Machine: {platform.machine()}")
        rows.append(f"Python: {platform.python_version()}")
        rows.append(f"CPU: {platform.processor()}")
        rows.append(f"Cores: {os.cpu_count()}")
        rows.append(f"PyTorch: {torch.__version__}")
        rows.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            rows.append(f"CUDA version: {torch.version.cuda}")
            rows.append(f"GPU: {torch.cuda.get_device_name(0)}")
            rows.append(f"GPU capability: {torch.cuda.get_device_capability(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            rows.append(f"GPU memory: {mem} MB")
        rows.append("")
        self.logger.info("\n".join(rows))

    def __str__(self):
        header = "\n" + "=" * 50 + f"\n{self.__class__.__name__}\n" + "=" * 50
        rows = [header]

        def indent(text: str) -> str:
            pad = " " * 4
            return "\n".join(pad + line for line in text.splitlines())

        def block(label, obj):
            rows.append(f"  {label}:")
            rows.append(indent(str(obj)))
            rows.append("")

        # --- blocks ---
        block("generator", self.generator)
        block("critic", self.critic)
        block("dataloader", self.dataloader)

        # --- trainer config ---
        rows.append(f"  device:         {self.device}")
        rows.append(f"  z_dim:          {self.z_dim}")
        rows.append(f"  epochs:         {self.epochs}")
        rows.append(f"  n_critic:       {self.n_critic}")
        rows.append(f"  batches:        {len(self.dataloader)}")
        rows.append(f"  lambda_gp:      {self.lambda_gp}")
        rows.append(f"  ema_decay:      {self.ema_decay}")
        rows.append("")

        # --- paths ---
        rows.append(f"  checkpoint dir: {self.checkpoint_dir}")
        rows.append(f"  samples dir:    {self.sample_dir}")
        rows.append("=" * 50)
        rows.append("")
        return "\n".join(rows)


def gradient_penalty(critic, real, fake, device="cuda", lambda_gp=10.0):
    """
    Computes the gradient penalty for WGAN-GP.

    Parameters:
    - critic: the WGAN-GP critic network
    - real: batch of real images
    - fake: batch of generated images
    - device: computation device
    - lambda_gp: gradient penalty coefficient (default 10)

    Returns:
    - gradient penalty scalar
    """

    batch_size, C, H, W = real.shape

    # Random interpolation factor between real and fake
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    # Critic score for interpolated samples
    mixed_scores = critic(interpolated)

    # Compute gradients of scores w.r.t. interpolated images
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients per sample
    gradients = gradients.view(batch_size, -1)

    # L2 norm per sample
    gradient_norm = gradients.norm(2, dim=1)

    # Final gradient penalty
    gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return gp
