"""
scripts/train_shoreline_autoencoder.py

Main script for training the shoreline cnn-autoencoder.
"""

import os
from pathlib import Path

import torchvision.transforms as T
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from ai_fractals.data.shoreline_dataset import ShorelineDataset
from ai_fractals.logging_config import get_logger
from ai_fractals.models.cnn_autoencoder import CNNAutoencoder
from ai_fractals.training import AutoencoderTrainer, EarlyStopping


def main(dataset_path: Path, models_path: Path):
    # --- init logging ---
    logger = get_logger("train_cnn_autoencoder")
    logger.info("Starting CNN autoencoder training script")

    # --- loading the dataset ---
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )
    dataset = ShorelineDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # --- model ---
    model = CNNAutoencoder()

    # --- early stopping
    early_stopping = EarlyStopping(
        monitor="train_loss",
        patience=20,
        mode="min",
    )

    # --- trainer ---
    trainer = AutoencoderTrainer(
        model,
        loader,
        lr=1e-3,
        epochs=120,
        early_stop=early_stopping,
    )

    # --- log objects via __str__ ---
    logger.info("\n" + str(dataset) + "\n")
    logger.info("\n" + str(model) + "\n")
    logger.info("\n" + str(trainer) + "\n")

    # --- training ---
    logger.info("Beginning training...")
    trainer.train()

    # --- save ---
    trainer.save(models_path / "shoreline_autoencoder.pth")
    logger.info("Training complete. Model saved to shoreline_autoencoder.pth")


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    SHORELINES_DIR = (
        PROJECT_ROOT / "dataset" / "shorelines" / "mandelbrot" / "256_256_iter256"
    )
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    print("\nDataset dir: ", SHORELINES_DIR)
    print("output Model dir: ", MODELS_DIR)
    print("input dir exists: ", SHORELINES_DIR.exists(), "\n")

    main(dataset_path=SHORELINES_DIR, models_path=MODELS_DIR)
