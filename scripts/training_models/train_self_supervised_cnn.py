"""
scripts/train_self_supervised_cnn.py

Main script for training the self-supervised CNN (SimCLR-style)
on shoreline images. The model learns geometry-based embeddings
without labels, suitable for clustering and GAN conditioning.

Augmentation note:
    Self-supervised contrastive learning relies on generating two
    *different* augmented views of each input image. These stochastic
    augmentations (random crops, flips, rotations, noise) force the
    encoder to learn stable, geometry-aware features that remain
    consistent under perturbations.

    For fractal shorelines, augmentation is essential: it prevents the
    model from memorizing pixel-level details and instead encourages
    learning of global shape, curvature, and structural patterns.
"""

import os
from pathlib import Path

import torch
import torchvision.transforms as T
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from ai_fractals.data.shoreline_dataset import ShorelineDataset
from ai_fractals.logging_config import get_logger
from ai_fractals.models.self_supervised_cnn import SelfSupervisedCNN
from ai_fractals.training.self_supervised_trainer import SelfSupervisedTrainer


def main(dataset_path: Path, models_path: Path):
    # --- init logging ---
    logger = get_logger("train_self_supervised_cnn")
    logger.info("Starting self-supervised CNN training script")

    # --- dataset loading ---
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )
    dataset = ShorelineDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # --- model ---
    model = SelfSupervisedCNN(embedding_dim=256)

    augment = T.Compose(
        [
            T.RandomResizedCrop(256, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  # gaussian noise
        ]
    )

    # --- trainer ---
    trainer = SelfSupervisedTrainer(
        model=model,
        dataloader=loader,
        augment_fn=augment,
        lr=1e-3,
        epochs=120,
        temperature=0.5,
    )

    # --- log objects ---
    logger.info("\n" + str(dataset) + "\n")
    logger.info("\n" + str(augment) + "\n")
    logger.info("\n" + str(model) + "\n")
    logger.info("\n" + str(trainer) + "\n")

    # --- training ---
    logger.info("Beginning training...")
    trainer.train()

    # --- save ---
    out_path = models_path / "self_supervised_cnn.pth"
    trainer.save(out_path)
    logger.info(f"Training complete. Model saved to {out_path.name}")


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    SHORELINES_DIR = (
        PROJECT_ROOT / "dataset" / "shorelines" / "mandelbrot" / "256_256_iter256"
    )
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    print("\nDataset dir: ", SHORELINES_DIR)
    print("Output model dir: ", MODELS_DIR)
    print("Input dir exists: ", SHORELINES_DIR.exists(), "\n")

    main(dataset_path=SHORELINES_DIR, models_path=MODELS_DIR)
