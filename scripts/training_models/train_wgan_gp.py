"""
scripts/train_wgan_gp.py

Main script for training a WGAN-GP on RGB fractal images.

The trainer handles:
- critic updates with gradient penalty
- generator updates
- Wasserstein distance tracking
- logging and checkpointing
"""

import os
from pathlib import Path

import torchvision.transforms as T
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from ai_fractals.data import RGBDataset
from ai_fractals.logging_config import get_logger
from ai_fractals.models import WganGpCritic, WganGpGenerator
from ai_fractals.training import WganGpTrainer


def main(dataset_path: Path, models_path: Path):
    logger = get_logger("train_wgan_gp")
    logger.info("Starting WGAN-GP training script")

    transform = T.Compose(
        [
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
        ]
    )

    dataset = RGBDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # --- models ---
    generator = WganGpGenerator(z_dim=128)
    critic = WganGpCritic(feature_maps=32)

    # --- trainer ---
    trainer = WganGpTrainer(
        generator=generator,
        critic=critic,
        dataloader=loader,
        z_dim=128,
        lr_g=1e-4,
        lr_c=1e-4,
        epochs=200,
        n_critic=3,
        lambda_gp=2.5,
        ema_decay=0.999,
        device=None,  # trainer will set automatically
        # --- uncomment to save checkpoints each epoch - about 250mb per ---
        # checkpoint_dir=Path(models_path / "wgan_checkpoints"),
        sample_dir=Path(models_path / "wgan_samples"),
    )

    # --- training ---
    logger.info("Beginning training...")
    trainer.train()

    # --- save ---
    trainer.save(
        gen_path=models_path / "wgan_pg_generator.pth",
        critic_path=models_path / "wgan_pg_critic.pth",
        ema_path=models_path / "wgan_pg_generator_ema.pth",
    )
    logger.info(f"Training complete. Generator, Critic and Ema saved to {models_path}")


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    DATASET_DIR = PROJECT_ROOT / "dataset" / "rgb" / "mandelbrot"
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    print("\nDataset dir: ", DATASET_DIR)
    print("Output model dir: ", MODELS_DIR)
    print("Input dir exists: ", DATASET_DIR.exists(), "\n")

    main(dataset_path=DATASET_DIR, models_path=MODELS_DIR)
