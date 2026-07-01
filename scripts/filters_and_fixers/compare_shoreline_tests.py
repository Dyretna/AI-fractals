from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


def load_images_by_id(folder: Path):
    """Return dict: {img_id: Path} for all PNGs in folder."""
    images = {}
    for path in folder.glob("*.png"):
        img_id = path.stem.split("_")[0]
        images[img_id] = path
    return images


def build_test_matrix(base_dir: Path, test_names):
    matrix = {}

    for test in test_names:
        eval_dir = base_dir / test / "evaluated"
        rej_dir = base_dir / test / "rejected"

        eval_imgs = load_images_by_id(eval_dir)
        rej_imgs = load_images_by_id(rej_dir)

        all_ids = set(eval_imgs.keys()) | set(rej_imgs.keys())

        for img_id in all_ids:
            if img_id not in matrix:
                matrix[img_id] = {}

            if img_id in eval_imgs:
                matrix[img_id][test] = {"path": eval_imgs[img_id], "status": "passed"}
            elif img_id in rej_imgs:
                matrix[img_id][test] = {"path": rej_imgs[img_id], "status": "failed"}
            else:
                matrix[img_id][test] = {"path": None, "status": "missing"}

    return matrix


def load_configs(cfg_paths):
    configs = {}
    for test_name, cfg_path in cfg_paths.items():
        with open(cfg_path, "r") as f:
            configs[test_name] = yaml.safe_load(f)
    return configs


def make_full_collage(base_dir: Path, test_names, out_dir: Path, configs):
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 33)
    except OSError as e:
        print(f"Font not found, using default: {e}")
        font = ImageFont.load_default()

    matrix = build_test_matrix(base_dir, test_names)
    img_ids = sorted(matrix.keys())

    # Find sample image size
    sample_path = None
    for img_id in img_ids:
        for test in test_names:
            entry = matrix[img_id][test]
            if entry["path"]:
                sample_path = entry["path"]
                break
        if sample_path:
            break

    sample_img = Image.open(sample_path)
    w, h = sample_img.size

    cols = 2
    rows = (len(test_names) + 1) // 2

    block_w = cols * w
    block_h = rows * h + 80  # extra for super-title

    for img_id in img_ids:
        block = Image.new("RGB", (block_w, block_h), (20, 20, 20))
        draw = ImageDraw.Draw(block)

        # Super title
        draw.text((10, 10), f"Image ID: {img_id}", fill=(255, 255, 255), font=font)

        for t_idx, test in enumerate(test_names):
            row = t_idx // 2
            col = t_idx % 2

            x = col * w
            y = 80 + row * h

            entry = matrix[img_id][test]
            status = entry["status"]

            if entry["path"]:
                img = Image.open(entry["path"]).convert("RGB")
            else:
                img = Image.new("RGB", (w, h), (50, 50, 50))

            block.paste(img, (x, y))

            # Border
            draw.rectangle(
                [x, y, x + w - 1, y + h - 1], outline=(200, 200, 200), width=5
            )

            # Title background
            draw.rectangle([x + 5, y + 5, x + w - 10, y + 40], fill=(0, 0, 0))

            # Title text
            color = (0, 255, 0) if status == "passed" else (255, 0, 0)
            draw.text((x + 10, y + 5), f"{test} - {status}", fill=color, font=font)

            # extract yaml config
            cfg = configs[test]
            det = cfg["detector"]
            ev = cfg["evaluator"]
            hi = cfg["hires_gen"]

            # DETECTOR LINE
            if det["apply_smoothing"]:
                det_line = (
                    f"canny edges: low={det['canny_low']} high={det['canny_high']} "
                    f"smooth={det['smoothing_method']}(k={det['smoothing_kernel']} sigma={det['smoothing_sigma']})"
                )
            else:
                det_line = f"canny edges:: low={det['canny_low']} high={det['canny_high']} smooth=off"

            # EVALUATOR LINE
            eval_iter_line = (
                f"eval: edge={ev['min_edge_ratio']}-{ev['max_edge_ratio']} "
                f"inside={ev['min_inside_ratio']}-{ev['max_inside_ratio']} | max_iter: {hi['max_iter']} "
            )

            # DRAW ALL THREE LINES
            draw.text((x + 150, y + 5), det_line, fill=(220, 220, 220), font=font)
            draw.text(
                (x + 150, y + 20), eval_iter_line, fill=(220, 220, 220), font=font
            )

        out_path = out_dir / f"{img_id}.png"
        block.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    base_dir = PROJECT_ROOT / "dataset" / "shoreline"
    out_dir = PROJECT_ROOT / "dataset" / "shoreline" / "test_collages"

    configs_dir = PROJECT_ROOT / "configs" / "shoreline_tests"

    cfg_paths = {
        "test_01": configs_dir / "shoreline_test_01.yaml",
        "test_02": configs_dir / "shoreline_test_02.yaml",
        "test_03": configs_dir / "shoreline_test_03.yaml",
        "test_04": configs_dir / "shoreline_test_04.yaml",
    }

    configs = load_configs(cfg_paths)

    make_full_collage(
        base_dir=base_dir,
        test_names=["test_01", "test_02", "test_03", "test_04"],
        out_dir=out_dir,
        configs=configs,
    )
