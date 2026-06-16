# scripts/remove_images.py

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from ai_fractals.data import OUT_FILTERED

load_dotenv()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))


def extract_cmap(filename: str) -> str:
    """
    Extract the colormap name from a filename.
    Expected format: <depth>_<max_iter>_<cmap>.png

    Example: d15_iter256_twilight_r.png -> twilight_r
    """
    name = filename.split(".")[0]
    parts = name.split("_")

    # reversed colormap: ends with "_r"
    if parts[-1] == "r":
        return parts[-2] + "_r"

    # normal colormap
    return parts[-1]


def move_bad_cmaps(img_dir: Path, out_dir: Path, bad_cmaps: list) -> None:
    """
    Move all images whose colormap matches one of the entries in `bad_cmaps`.

    Filenames are expected to follow the format:
        <depth>_<max_iter>_<cmap>.png
    Example:
        d15_iter256_twilight.png  -> colormap = "twilight"

    Any file whose extracted colormap is found in `bad_cmaps` is moved from
    `img_dir` to `out_dir`. Files are removed from the source directory after
    moving. Non-files and unknown filename formats are ignored.
    """

    if not img_dir.is_dir():
        raise IsADirectoryError("image folder is not a directory!")

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    for file in img_dir.iterdir():
        if not file.is_file():
            continue

        cmap = extract_cmap(file.name)

        if cmap in bad_cmaps:
            new_file = out_dir / file.name
            print(f"moving {file.name} -> {new_file.name}")
            shutil.move(str(file), str(new_file))
            total += 1

    print(f"\nDone. Moved {total} files.")


if __name__ == "__main__":
    if not PROJECT_ROOT.is_dir():
        raise IsADirectoryError("check PROJECT_ROOT in .env")

    img_dir = PROJECT_ROOT / "dataset" / "mandelbrot" / "1024_1024_iter1024"
    output_dir = PROJECT_ROOT / "dataset" / "out" / "1024_1024_iter1024"
    bad_cmaps = OUT_FILTERED

    move_bad_cmaps(img_dir, output_dir, bad_cmaps)
