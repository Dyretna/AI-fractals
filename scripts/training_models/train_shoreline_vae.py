"""
scripts/train_shoreline_vae.py

Main script for training the conditional ShorelineVAE on shoreline images.
The model learns a continuous latent space over fractal geometry, conditioned
on the fractal region bounds [xmin, xmax, ymin, ymax].

Conditioning note:
    Shoreline masks contain only local geometric structure. Bounds provide
    the global context: the exact location of the region in the complex plane.
    Conditioning the VAE on bounds ensures that the latent space becomes
    position-aware, enabling stable interpolation and realistic synthetic
    shoreline generation.
"""

import os
from pathlib import Path

import torchvision.transforms as T
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from ai_fractals.data import ShorelineWithBoundsDataset
from ai_fractals.logging_config import get_logger
from ai_fractals.models import ShorelineVAE
from ai_fractals.training import VAETrainer


def main(region_root: Path, shoreline_root: Path, models_path: Path):
    # --- init logging ---
    logger = get_logger("train_shoreline_vae")
    logger.info("Starting ShorelineVAE training script")

    # --- dataset loading ---
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )
    dataset = ShorelineWithBoundsDataset(
        region_root=region_root,
        shoreline_root=shoreline_root,
        transform=transform,
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # --- model ---
    model = ShorelineVAE(latent_dim=64, cond_dim=4)

    # --- trainer ---
    trainer = VAETrainer(
        model=model,
        dataloader=loader,
        lr=1e-4,
        epochs=60,
    )

    # --- log objects ---
    logger.info("\n" + str(dataset) + "\n")
    logger.info("\n" + str(model) + "\n")
    logger.info("\n" + str(trainer) + "\n")

    # --- training ---
    logger.info("Beginning training...")
    trainer.train()

    # --- save ---
    out_path = models_path / "shoreline_vae.pth"
    trainer.save(out_path)
    logger.info(f"Training complete. Model saved to {out_path.name}")


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    REGIONS_DIR = PROJECT_ROOT / "dataset" / "region" / "evaluated"
    SHORELINES_DIR = PROJECT_ROOT / "dataset" / "shoreline" / "evaluated"
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    print("\nRegions dir: ", REGIONS_DIR)
    print("exists: ", REGIONS_DIR.exists())
    print("Shorelines dir: ", SHORELINES_DIR)
    print("exists: ", SHORELINES_DIR.exists())
    print("\nOutput model dir: ", MODELS_DIR)

    main(region_root=REGIONS_DIR, shoreline_root=SHORELINES_DIR, models_path=MODELS_DIR)
