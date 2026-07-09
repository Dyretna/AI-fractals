"""
ai_fractals/data/shoreline_dataset.py

Dataset loaders for fractal shoreline and RGB images, including optional
metadata-based conditioning. These loaders provide the core data interface
for self-supervised CNN training, VAE training, and GAN conditioning.

The module defines three dataset classes:

1. RGBDataset
   Loads RGB fractal images stored as .png files. Used primarily for
   training GAN models that generate full-color fractal renderings.

2. ShorelineDataset
   Loads grayscale shoreline masks stored as .png files. These masks
   represent the geometric boundary of fractal regions and are used for
   self-supervised contrastive learning and geometry embedding.

3. ShorelineWithBoundsDataset
   Loads shoreline masks together with their associated fractal bounds.
   Shoreline PNGs and region JSON metadata may reside in different
   directories. Matching is performed via compact_id extracted from the
   PNG filename. JSON filenames may contain arbitrary suffixes, e.g.:

       260616180525.json
       260616180525_iter256_d05.json
       260616180525_bounds.json

   Only the compact_id prefix must match. The JSON file is expected to
   contain a "bounds" field:

       "bounds": [xmin, xmax, ymin, ymax]

   This dataset is used for training the conditional ShorelineVAE, where
   bounds provide global geometric context that complements the local
   shoreline mask. Conditioning the VAE on bounds enables position-aware
   latent representations and stable interpolation across fractal regions.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class RGBDataset(Dataset):
    """
    Dataset loader for RGB fractal images stored as .png files.
    Mirrors ShorelineDataset but loads 3-channel color images.
    """

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.paths = sorted(self.root.rglob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")  # 3‑channel

        if self.transform:
            img = self.transform(img)

        return img

    def __str__(self):
        rows = [
            "RGBDataset",
            f"  root:   {self.root}",
            f"  count:  {len(self.paths)}",
            f"  transform: {self.transform.__class__.__name__ if self.transform else None}",
        ]
        return "\n".join(rows)


class ShorelineDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.paths = sorted(self.root.rglob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("L")  # grayscale

        if self.transform:
            img = self.transform(img)

        return img

    def __str__(self):
        rows = [
            "ShorelineDataset",
            f"  root:   {self.root}",
            f"  count:  {len(self.paths)}",
            f"  transform: {self.transform.__class__.__name__ if self.transform else None}",
        ]
        return "\n".join(rows)


class ShorelineWithBoundsDataset(Dataset):
    """
    Loads shoreline images (grayscale) together with their fractal bounds.

    Args:
        shoreline_root: directory containing shoreline PNG files
        region_root: directory containing region JSON metadata files
        transform: optional torchvision transform

    Returns:
        (image_tensor, bounds_tensor)
    """

    def __init__(self, shoreline_root: Path, region_root: Path, transform=None):
        self.shoreline_root = Path(shoreline_root)
        self.region_root = Path(region_root)
        self.transform = transform

        # Collect shoreline PNGs
        self.paths = sorted(self.shoreline_root.rglob("*.png"))

    def __len__(self):
        return len(self.paths)

    def _find_json_for_id(self, compact_id: str) -> Path:
        """
        Finds the JSON file whose name *starts with* the compact_id.
        Example:
            compact_id = "260616180525"
            matches:
                260616180525_iter256_d05.json
                260616180525_bounds.json
        """
        candidates = list(self.region_root.glob(f"{compact_id}*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No JSON metadata found for compact_id={compact_id} "
                f"in {self.region_root}"
            )
        return candidates[0]

    def __getitem__(self, idx):
        png_path = self.paths[idx]

        # Extract compact_id from filename
        compact_id = png_path.stem.split("_")[0]

        # Load shoreline image
        img = Image.open(png_path).convert("L")
        if self.transform:
            img = self.transform(img)

        # Find matching JSON
        json_path = self._find_json_for_id(compact_id)

        # Load metadata
        with open(json_path, "r") as f:
            meta = json.load(f)

        bounds = torch.tensor(meta["bounds"], dtype=torch.float32)

        return img, bounds

    def __str__(self):
        rows = [
            "ShorelineWithBoundsDataset",
            f"  shoreline_root: {self.shoreline_root}",
            f"  region_root:    {self.region_root}",
            f"  count:          {len(self.paths)}",
            f"  transform:      {self.transform.__class__.__name__ if self.transform else None}",
            "  fields:         shoreline + bounds",
        ]
        return "\n".join(rows)


class RGBWithEmbeddingDataset(Dataset):
    """
    Dataset loader for RGB fractal images paired with precomputed CNN embeddings.

    This dataset assumes a single .pt file containing a tensor of shape (N, embed_dim),
    where each row corresponds to the embedding of the RGB image at the same index.

    Returns:
        (rgb_tensor, embedding_tensor)
    """

    def __init__(self, rgb_root: Path, embedding_path: Path, transform=None):
        self.rgb_root = Path(rgb_root)
        self.embedding_path = Path(embedding_path)
        self.transform = transform

        # Collect RGB PNGs
        self.rgb_paths = sorted(self.rgb_root.rglob("*.png"))

        # Load embedding tensor
        self.embeddings = torch.load(self.embedding_path, weights_only=True)

        if len(self.embeddings) != len(self.rgb_paths):
            raise ValueError(
                f"Embedding count ({len(self.embeddings)}) does not match "
                f"image count ({len(self.rgb_paths)})."
            )

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # Load RGB image
        png_path = self.rgb_paths[idx]
        img = Image.open(png_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Load embedding at same index
        emb = self.embeddings[idx]

        return img, emb

    def __str__(self):
        rows = [
            "RGBWithEmbeddingDataset",
            f"  rgb_root:       {self.rgb_root}",
            f"  embedding_path: {self.embedding_path}",
            f"  count:          {len(self.rgb_paths)}",
            f"  transform:      {self.transform.__class__.__name__ if self.transform else None}",
            "  fields:         rgb + embedding",
        ]
        return "\n".join(rows)
