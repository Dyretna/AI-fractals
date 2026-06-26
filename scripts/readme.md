# Scripts Directory

This directory contains all standalone tools and batch scripts used for
dataset generation, shoreline extraction, dataset cleanup, and model
training. These scripts are **not** part of the core `ai_fractals` library.
They are operational utilities intended for running batch jobs, fixing
datasets, or performing one‑off tasks.

The reason is to keep the core library clean and modular while placing all
pipeline‑level tools here.

--------------------------------------------------------------------------

## Directory Structure
```
scripts/
├── batch_generation/
│   ├── batch_shoreline_generation.py
│   └── batch_RGB_generation.py
├── training_models/
├── filters_and_fixers/
├── cli_simple_fractal_generation.py
├── cli_batch_generation.py
└── shorelines_from_img.py
```

--------------------------------------------------------------------------

## CLI Tools

### cli_simple_fractal_generation.py
A lightweight command‑line tool for generating small sets of RGB fractals
without tile‑search. Useful for experimentation, debugging, and quick
visualization.

While useful for experimentation, building a full fractal-AI pipeline
requires large and diverse datasets. For this reason, dedicated batch
pipelines (dataset_batch_builder.py and shorelines_batch_builder.py) exist
to generate thousands of samples efficiently.

Example:
```bash
python cli_dataset_builder.py --type mandelbrot --n 50 --iter 512
```

---

### cli_batch_generation.py
The main entry point for running **one or multiple batch pipelines** using
YAML configuration files.

This CLI:
- loads one or more YAML configs
- reads the `job_type` field
- dispatches to the correct batch builder
- executes pipelines sequentially

Run a single config:
```bash
python cli_batch_generation.py --config configs/rgb/rgb_cfg.yaml
```

Run multipe configs in a sequence:
```bash
python cli_batch_generation.py --config cfg1.yaml cfg2.yaml cfg3.yaml
```

## Batch Builders

### batch_RGB_generation.py
Generates large batches of high‑resolution RGB fractal images. This is the
standard fractal dataset generator used for building reference datasets or
training GAN models.


### batch_shoreline_generation.py
Generates shoreline datasets from scratch, without saving intermediate
RGB images. This is the recommended method for producing training data for
the self‑supervised CNN that learns geometry‑based embeddings.


### shorelines_from_img.py
Builds shorelines **from existing RGB fractal images**.

This was an early conceptual tool used to understand the problem. It loads
RGB images, extracts shorelines, evaluates them, and applies augmentation.

Limitations:
- requires many large source images (often saved in 1024x1024 or larger)
- storing these source images consumes significant disk space
- not ideal for generating 5000-10000 shorelines needed for CNN training

The "from scratch" builder (shorelines_batch_builder.py) solves this by
generating fractal tiles directly in memory, without saving intermediate RGB
renders.

--------------------------------------------------------------------------

## Filters and Fixers

### fix_images_in_dataset.py
Repairs or normalizes datasets where colormap metadata is missing or
incorrect. Can rename files or adjust directory structure.

### remove_from_dataset.py
Filters out low-quality or invalid images based on heuristics or metadata.
Useful for cleaning large datasets before training.

--------------------------------------------------------------------------

## Training Scripts

### train_self_supervised_cnn.py
Trains the CNN that produces geometry‑based embeddings from shoreline
images.

### train_shoreline_autoencoder.py
Trains the VAE used to learn latent structure in shoreline geometry.

--------------------------------------------------------------------------

## Usage Notes

- All scripts assume that PROJECT_ROOT is defined in a .env file.
- GPU acceleration is recommended for high-resolution generation.
- The scripts directory is intended for batch operations and utilities, not
  for library imports.
- Each script can be run directly with Python.

--------------------------------------------------------------------------

## Typical Workflow

1. Explore fractals:

``` bash
python scripts/cli_dataset_builder.py --type mandelbrot --n 20
```

2. Generate RGB fractal datasets:
```bash
python scripts/cli_batch_generation.py --config configs/rgb/rgb_cfg.yaml
```

3. Generate shoreline datasets:
```bash
python scripts/cli_batch_generation.py --config configs/shoreline/shoreline_cfg.yaml
```

4. Run multiple pipelines in sequence:
```bash
python scripts/cli_batch_generation.py --config cfg1.yaml cfg2.yaml cfg3.yaml
```

5. Clean datasets:
```bash
python scripts/filters_and_fixers/remove_from_dataset.py
```

Train models:
```bash
python scripts/training_models/train_self_supervised_cnn.py
```
