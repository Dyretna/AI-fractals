# AI-Fractals

![image](example_images/conventional_fractals/YlGnBu_01.png)

## Overview

This project explores the intersection of artificial intelligence and fractal geometry, combining machine learning techniques with traditional fractal mathematics to generate and analyze fractal patterns.

The project is inspired by research from **Douglas C. Youvan** (doug@youvan.com), detailed in the paper [AI-Enhanced Fractal Geometry: Merging Machine Learning with Traditional Fractal Mathematics](docs/ai-enhanced-fractal-geometry.md).

AI-Fractals provides:

1. A **fractal generation library** (`ai_fractals/`)
2. A **dataset pipeline** for generating large-scale RGB and shoreline datasets
3. Tools for **CNN-based embedding learning** from fractal shorelines (in progress)
4. Infrastructure for **GAN-based pseudo-fractal generation** (in progress)

## project Status (June 2026)

Completed:
- **Automatic Fractal Generation**: Generate Mandelbrot and Julia set fractals with configurable parameters
- **Shoreline Extraction**:
    - Extract and analyze fractal shoreline patterns from generated fractals, or
    - create Shorelines from scratch in big batches
- **CUDA acceleration**: Both in conventional generation and in GANs training. Also see https://github.com/TomLemsky/pytorch-fractals?tab=readme-ov-file
- **Supersampling techniques**: upsampling -> downsampling -> gaussian blur, for smoother fractals

In Progress:
- **AI Training**: Train neural networks (CNNs, GANs) on fractal datasets to generate pseudo-fractals
- **Julia Set genereators**: (GPU/CPU backends, parameter handling, tile‑search integration).


## Directory Structure
```
.
├── docs/
├── dataset/                        # generated datasets (not stored in repo)
├── example_images/                 # Sampled fractals (no GANs, yet)
├── notebooks/                      # demos, prototyping
├── scripts/                        # CLI tools, batch pipelines, utilities
│   ├── cli_batch_generation.py
│   ├── cli_simple_fractal_generation.py
│   └── readme.md
└── src/
    └── ai_fractals/
        ├── analysis                # fractal evaluation and quality metrics
        ├── data                    # rgb dataset and shoreline builders
        ├── generators              # Mandelbrot, Julia, etc
        ├── models                  # CNN, VAEs and GANs architectures
        ├── processing              # edge-detection, filters, transforms
        ├── search                  # strategies in dataset generation
        └── training                # for training ai-models
```

## Typical Workflow

1. Explore conventional fractal generation using the simple CLI
2. Generate RGB datasets (batch pipeline)
3. Generate shoreline datasets (recommended for CNN training):
4. Train a CNN on shoreline images to obtain geometry-based embeddings. (future work)
5. Use these embeddings to condition or guide GAN models (future work).


## Installation

Requires Python 3.10 or higher.

Clone and install:

```bash
# Clone the repository
git clone https://github.com/Dyretna/AI-fractals.git
cd AI-fractals

# Install package
pip install -e .
```
Depending on your hardware, install the appropriate PyTorch build.


### CUDA (Nvidia GPU)
```bash
# install Pytorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
### CPU-only (NO GPU required)
```bash
# Install PyTorch - CPU‑only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```


### Set environment paths with python-dotenv
This repo makes use of load_dotenv() from python-dotenv.
Set your project directory in a `.env` file:

```
PROJECT_ROOT=/path/to/AI-fractals
```

Then all paths will load correctly.


## CLI Usage
The project includes two command‑line interfaces:

1. **A simple fractal generator CLI** — good for exploration and testing
2. **A batch CLI** — used for large‑scale dataset generation via YAML

### Simple CLI (quick fractal generation)

Basic usage (default: 50 Mandelbrot images):

```bash
    python scripts/cli_dataset_generation.py
```

High-resolution example:

    python scripts/cli_dataset_generation.py \
        --width 2048 \
        --height 2048 \
        --max_iter 1500

#### Arguments for simple CLI
    --n          Number of images to generate (default: 50)
    --type       Fractal type: mandelbrot or julia (default: mandelbrot)
    --width      Image width in pixels (default: 1024)
    --height     Image height in pixels (default: 1024)
    --max_iter   Maximum iterations for the fractal generator (default: 1024)
    --cmap       Matplotlib colormap name (default: twilight)
    --out        Output directory (default: PROJECT_ROOT/dataset/fractals/<type>)


### Batch CLI (YAML‑driven pipelines)
For large‑scale dataset generation (RGB or shoreline), use:

Generate RGB datasets (batch pipeline)
```bash
python scripts/cli_batch_generation.py --config configs/rgb/rgb_cfg.yaml
```

Generate shoreline datasets (recommended for CNN training):
```bash
python scripts/cli_batch_generation.py --config configs/shoreline/shoreline_cfg.yaml
```

You can also run multiple pipelines in sequence:
```bash
python scripts/cli_batch_generation.py --config cfg1.yaml cfg2.yaml cfg3.yaml
```

Each config file defines:

- job type (`rgb` or `shoreline`)
- fractal generator parameters
- fractal search strategy settings
- resolution, iteration counts, colormap, etc.

For full documentation, see:

```
scripts/README.md
```


## Research Background

This project implements concepts from AI-enhanced fractal geometry, where machine learning models are trained on traditional fractal patterns to:

1. **Learn fractal characteristics**: CNNs extract features from fractal shorelines
2. **Generate new patterns**: GANs create novel pseudo-fractals with learned properties
3. **Analyze complexity**: Study the fractal dimension and properties of generated patterns

These components are planned but not yet implemented.

## Author

**Jesper Anteryd** (jesper.anteryd@proton.me)
Inspired by the research of **Douglas C. Youvan**

## License

See [LICENSE](LICENSE) file for details.

## References

- Youvan, D.C. "AI-Enhanced Fractal Geometry: Merging Machine Learning with Traditional Fractal Mathematics"
