"""
ai_fractals/data/shoreline_dataset.py

Dataset loader for shoreline images stored as .png files.
"""

from pathlib import Path

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
