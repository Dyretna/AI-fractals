# Scripts Directory

This directory contains all standalone tools and batch scripts used for
metadata generation, shoreline rendering, dataset cleanup, and model
training.

The core library (ai_fractals/) provides generators, evaluators, logging,
and processing utilities.

The scripts/ directory provides pipeline-level tools that orchestrate
large batch jobs.

--------------------------------------------------------------------------

## Directory Structure
```
scripts/
├── batch_generation/
│   ├── batch_region_params.py
│   └── batch_shoreline_from_meta.py
│
├── training_models/
│   ├── train_self_supervised_cnn.py
│   ├── train_shoreline_autoencoder.py
│   └── train_wgan_gp.py
│
├── filters_and_fixers/
│   ├── check_region_duplicates.py
│   └── compare_shoreline_tests.py
│
├── cli_batch_generation.py
└── readme.md
```

--------------------------------------------------------------------------


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

### batch_region_params.py

Generates region metadata (bounds, depth, compact_id, etc.) using the
fractal tile search.
This is the first step in the pipeline.
All downstream rendering (RGB or shoreline) is based on these metadata files.


### batch_shoreline_from_meta.py
Renders shorelines from metadata.
- For each region:
- loads bounds from JSON
- renders a high‑resolution fractal tile
- evaluates quality using FractalQualityEvaluator
- extracts shoreline using EdgeDetector
- saves the result into evaluated or rejected
- moves the region metadata accordingly (if enabled)


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

### check_region_duplicates.py
Detects and removes duplicate region metadata files (same bounds or same
fractal coordinates). Useful after large tile searches.

### compare_shoreline_tests.py
Compares shoreline extraction setups (smoothing, dilation, max_iter, etc.).
This was used during shoreline test analysis.
Only setup 3 is now used in production.

--------------------------------------------------------------------------

## Training Scripts

### train_self_supervised_cnn.py
Trains the CNN that produces geometry‑based embeddings from shoreline
images.

### train_shoreline_autoencoder.py
Trains the VAE used to learn latent structure in shoreline geometry.

### train_wgan_gp.py
Trains a WGAN‑GP model on RGB fractal images.

--------------------------------------------------------------------------

## Usage Notes

- All scripts assume that PROJECT_ROOT is defined in a .env file.
- GPU acceleration is recommended for high-resolution generation.
- The scripts directory is intended for batch operations and utilities, not
  for library imports.
- Each script can be run directly with Python.

--------------------------------------------------------------------------

## CLI
cli_batch_generation.py
Runs batch pipelines using YAML configuration files.
Reads the job_type field and dispatches to the correct batch script.

```
python scripts/cli_batch_generation.py --config configs/shoreline/shoreline_test_03.yaml
```
--------------------------------------------------------------------------




## Current Workflow

1. Generate region metadata

``` bash
python scripts/cli_batch_generation.py --config configs/region/region_cfg.yaml
```

2. Render shorelines from metadata (setup from test 3)
```bash
python scripts/cli_batch_generation.py --config configs/rgb/rgb_cfg.yaml
```

3. Render RGB fractals from evaluated metadata
```bash
python scripts/cli_batch_generation.py --config configs/rgb/rgb_cfg.yaml
```

4. Train models:
```bash
python scripts/training_models/train_self_supervised_cnn.py
```
