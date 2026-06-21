# Scripts Directory

This directory contains all standalone tools and batch scripts used for
dataset generation, shoreline extraction, dataset cleanup, and metadata
export. The structure is organized by purpose to keep the workflow clear
and modular.

The scripts are not part of the core library. They are utilities intended
for running large batch jobs, fixing datasets, or performing one-off tasks.

--------------------------------------------------------------------------

## Directory Structure
```
scripts/
├── batch_builders/
│   ├── dataset_batch_builder.py
│   └── shorelines_batch_builder.py
├── filters_and_fixers/
│   ├── fix_images_in_dataset.py
│   └── remove_from_dataset.py
├── cli_dataset_builder.py
├── register_to_csv.py
└── shorelines_from_img.py
```

--------------------------------------------------------------------------

## CLI Tools

### cli_dataset_builder.py
A simple command-line interface for generating fractal datasets without
writing Python code. This script is intended as an entry point for exploring
automatically generated fractals and understanding the dataset-building
workflow.

While useful for experimentation, building a full fractal-AI pipeline
requires large and diverse datasets. For this reason, dedicated batch
pipelines (dataset_batch_builder.py and shorelines_batch_builder.py) exist
to generate thousands of samples efficiently.

Example:
    python cli_dataset_builder.py --type mandelbrot --n 50 --iter 512

--------------------------------------------------------------------------

## Batch Builders

### dataset_batch_builder.py
Generates large batches of high-resolution RGB fractal images using the
FractalDatasetBuilder. Iterates over curated colormaps and iteration counts.
Used for building the main fractal dataset.

This is the conventional fractal generator: it produces full-color fractal
renders that can be inspected visually or used as a reference dataset.

### shorelines_batch_builder.py
Batch shoreline generation using ShorelineDatasetBuilder. This builder
creates shorelines **from scratch**, without loading any images from disk.

It performs:
- tile-search
- high-resolution fractal tile generation
- shoreline extraction
- quality evaluation
- augmentation
- saving

This is the *recommended* way to build large shoreline datasets for
self-supervised CNN training. It does not require storing thousands of
large RGB images; instead, it generates the final shoreline images directly
in the desired resolution.

**Important:** Shorelines are *not* part of the dataset used for training
GAN models, so paired RGB-and-shoreline samples are unnecessary. Shorelines
serve a different purpose: they are used to train a CNN that produces
geometry-based embeddings. These embeddings are later used by the GANs as
conditioning signals or structural guidance.

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

## Batch Fixers

### fix_images_in_dataset.py
Repairs or normalizes datasets where colormap metadata is missing or
incorrect. Can rename files or adjust directory structure.

### remove_from_dataset.py
Filters out low-quality or invalid images based on heuristics or metadata.
Useful for cleaning large datasets before training.

--------------------------------------------------------------------------

## Metadata Export

### register_to_csv.py
Scans a dataset directory and exports metadata (bounds, depth, colormap,
iteration count, timestamps, etc.) into a CSV file for analysis or training
pipelines.

--------------------------------------------------------------------------

## Usage Notes

- All scripts assume that PROJECT_ROOT is defined in a .env file.
- GPU acceleration is recommended for high-resolution generation.
- The scripts directory is intended for batch operations and utilities, not
  for library imports.
- Each script can be run directly with Python.

--------------------------------------------------------------------------

## Typical Workflow

1. Explore fractals using the CLI:
       python scripts/cli_dataset_builder.py --type mandelbrot --n 20

2. Generate RGB fractal datasets:
       python scripts/batch_builders/dataset_batch_builder.py

3. Generate shoreline datasets (recommended for CNN training):
       python scripts/batch_builders/shorelines_batch_builder.py

4. Convert existing images to shorelines (legacy method):
       python scripts/batch_builders/shorelines_from_img.py

5. Clean or filter datasets:
       python scripts/filter_and_fixers/remove_from_dataset.py

6. Export metadata:
       python scripts/register_to_csv.py

--------------------------------------------------------------------------

## Purpose

The goal of this directory is to keep all operational scripts in one place,
separate from the core fractal generation library. This makes it easy to
run batch jobs, maintain datasets, and automate workflows without mixing
utility code into the main package.

The batch builders form the backbone of the fractal-AI pipeline, enabling
the creation of large, diverse datasets required for training models that
learn fractal geometry and structure.
