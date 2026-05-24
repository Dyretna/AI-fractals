"""Training configuration classes.

Simple configuration classes for managing training parameters.
No ABCs - just plain dataclasses.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class TrainingConfig:
    """Base configuration for training."""

    # Data
    data_dir: str
    image_size: Tuple[int, int] = (128, 128)
    batch_size: int = 32
    validation_split: float = 0.2

    # Training
    epochs: int = 100
    learning_rate: float = 0.001

    # Saving
    output_dir: str = "models/output"
    save_interval: int = 10

    # Hardware
    auto_batch_size: bool = True  # Auto-adjust batch size for available hardware

    def __post_init__(self):
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class GANConfig(TrainingConfig):
    """Configuration for GAN training."""

    latent_dim: int = 100
    discriminator_lr: float = 0.0002
    generator_lr: float = 0.0002
    beta_1: float = 0.5
    label_smoothing: float = 0.05
    n_discriminator_steps: int = 1

    output_dir: str = "models/gan"

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Return input shape for discriminator."""
        return (*self.image_size, 1)


@dataclass
class CNNConfig(TrainingConfig):
    """Configuration for CNN training."""

    patience: int = 10
    monitor: str = "val_loss"

    output_dir: str = "models/cnn"

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Return input shape for CNN."""
        return (*self.image_size, 1)
