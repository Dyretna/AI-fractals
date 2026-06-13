# AI-Fractals

![image](example_images/d09_0.11673_-0.62989_twilight.png)

## Overview

This project explores the intersection of artificial intelligence and fractal geometry, combining machine learning techniques with traditional fractal mathematics to generate and analyze fractal patterns.

The project is inspired by research from **Douglas C. Youvan** (doug@youvan.com), detailed in the paper [AI-Enhanced Fractal Geometry: Merging Machine Learning with Traditional Fractal Mathematics](docs/ai-enhanced-fractal-geometry.md).

## project Status (June 2026)

Completed:
- **Automatic Fractal Generation**: Generate Mandelbrot and Julia set fractals with configurable parameters
- **Shoreline Extraction**: Extract and analyze fractal shoreline patterns from generated fractals
- **CUDA acceleration**: CPU-based fractal generation is slow for large datasets. Future versions will use PyTorch tensors on CUDA to parallelize escape-time iteration and dramatically speed up fractal generation. Also see https://github.com/TomLemsky/pytorch-fractals?tab=readme-ov-file
- **Supersampling techniques**: upsampling -> downsampling -> gaussian blur, for smoother fractals

In Progress:
- **AI Training**: Train neural networks (CNNs, GANs) on fractal datasets to generate pseudo-fractals
- **Julia Set**: Initial infrastructure for Julia set generation is implemented (GPU/CPU backends, parameter handling, tile‑search integration). However, Julia sets are extremely sensitive to the choice of the complex parameter c, and the current sampling strategy is only preliminary.


## Installation

Requires Python 3.10 or higher.

Clone and install:

```bash
# Clone the repository
git clone https://github.com/Dyretna/AI-fractals.git
cd AI-fractals

# Install dependencies
pip install -e .
```

### Set environment paths with python-dotenv
This repo makes use of load_dotenv() from python-dotenv.
Set your project directory in a `.env` file:

```
PROJECT_ROOT=/path/to/AI-fractals
```

Then all paths will load correctly.


### CUDA (GPU acceleration)
This project uses pytorch with CUDA-support for fast generation of fractals. Install the CUDA-version of PyTorch manually:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```
CPU works, but is much slower.


## Directory Structure
```
.
├── docs/
├── dataset/                        # training data - default output of dataset_builder (too big to store on GH)
├── example_images/                 # So far, only regularly generated (no CNN / GANs, yet)
├── notebooks/                      # demos, prototyping
├── scripts/
│   └── cli_dataset_builder.py      # for creating the training-dataset (usage, see below)
└── src/
    └── ai_fractals/
        ├── analysis                # module for evaluating fractals (used for both generating and CNN / Gans training)
        ├── data                    # dataset_builder and shoreline_extractor
        └── generators              # the fractal generators (Mandelbrot, Julia)
```

## CLI Usage
The project includes a command-line runner for generating fractal datasets:

Basic usage (default: 50 Mandelbrot images):

    python scripts/cli_dataset_builder.py

High-resolution output:

    python scripts/cli_dataset_builder.py \
        --width 2048 \
        --height 2048 \
        --max_iter 1500

### CLI Arguments
    --n          Number of images to generate (default: 50)
    --type       Fractal type: mandelbrot or julia (default: mandelbrot)
    --width      Image width in pixels (default: 1024)
    --height     Image height in pixels (default: 1024)
    --max_iter   Maximum iterations for the fractal generator (default: 1024)
    --cmap       Matplotlib colormap name (default: twilight)
    --out        Output directory (default: PROJECT_ROOT/dataset/fractals/<type>)

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
