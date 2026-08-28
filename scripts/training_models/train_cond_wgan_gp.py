"""
scripts/train_cond_wgan_gp.py

Main script for training a conditional WGAN-GP on RGB fractal images.

The trainer handles:
- critic updates with gradient penalty on (image, embedding)
- generator updates conditioned on embeddings
- Wasserstein distance tracking
- checkpointing
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import torchvision.transforms as T
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from ai_fractals.data import RGBWithEmbeddingDataset
from ai_fractals.models import WganGpCritic, WganGpGenerator
from ai_fractals.training import WganGpTrainer
from scripts.utils import get_system_specs_str, setup_logging

load_dotenv()
ROOT = Path(os.getenv("PROJECT_ROOT"))
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_logging(redirect_path=ROOT / "logs" / f"{ts}_wgan_train.log")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------


def main(dataset_path: Path, embed_path: Path, models_path: Path):
    # log system specs
    log.info(get_system_specs_str())
    log.info("Starting conditional WGAN-GP training script")

    transform = T.Compose(
        [
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
        ]
    )

    # dataset returns (rgb_image, cnn_embedding)
    dataset = RGBWithEmbeddingDataset(
        rgb_root=dataset_path,
        embedding_path=embed_path,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # --- models ---
    generator = WganGpGenerator(noise_dim=128, embed_dim=256)
    critic = WganGpCritic(feature_maps=32, embed_dim=256)

    # --- trainer ---
    trainer = WganGpTrainer(
        generator=generator,
        critic=critic,
        dataloader=loader,
        noise_dim=128,
        embed_dim=256,
        lr_g=1e-4,
        lr_c=1e-4,
        epochs=200,
        n_critic=3,
        lambda_gp=2.5,
        ema_decay=0.999,
        device=None,  # auto
        sample_dir=Path(models_path / "wgan_samples"),
        checkpoint_dir=Path(models_path / "wgan_checkpoints"),
    )

    log.info("Beginning training...")
    trainer.train()

    trainer.save(
        gen_path=models_path / "wgan_cond_generator.pth",
        critic_path=models_path / "wgan_cond_critic.pth",
        ema_path=models_path / "wgan_cond_generator_ema.pth",
    )
    log.info(f"Training complete. Models saved to {models_path}")


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

    # input
    DATASET_DIR = (
        PROJECT_ROOT
        / "dataset"
        / "rgb"
        / "mandelbrot"
        / "twilight_global_gradnorm_iter2048"
    )
    EMBED_PATH = PROJECT_ROOT / "models" / "cnn_embeddings.pt"

    # output
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    print("\nDataset dir: ", DATASET_DIR)
    print("Embedding path:", EMBED_PATH)
    print("Output model dir: ", MODELS_DIR)

    print("Dataset dir exist:", DATASET_DIR.exists())
    print("Embedding path is file:", EMBED_PATH.is_file())

    main(dataset_path=DATASET_DIR, embed_path=EMBED_PATH, models_path=MODELS_DIR)
