# AI-Fractals

AI-Enhanced Fractal Geometry: Merging Machine Learning with Traditional Fractal Mathematics

## Overview

This project explores the intersection of artificial intelligence and fractal geometry, combining machine learning techniques with traditional fractal mathematics to generate and analyze fractal patterns. The work is inspired by research from **Douglas C. Youvan** (doug@youvan.com), detailed in the paper [AI-Enhanced Fractal Geometry: Merging Machine Learning with Traditional Fractal Mathematics](docs/AI-EnhancedFractalGeometry-MergingMachineLearningwithTraditionalFractalMathematics.pdf).

## Features

- **Fractal Generation**: Generate Mandelbrot and Julia set fractals with configurable parameters
- **Shoreline Extraction**: Extract and analyze fractal shoreline patterns from generated fractals
- **AI Training**: Train neural networks (CNNs, GANs) on fractal datasets to generate pseudo-fractals
- **Image Pipeline**: Automated saving and processing of fractal images

## Project Structure

```
AI-fractals/
├── src/
│   └── ai_fractals/               # Main package
│       ├── generators/            # Fractal generation
│       │   └── fractal_generators.py
│       ├── io/                    # Image I/O and pipelines
│       │   ├── img_saver.py
│       │   └── fractal_saver_pipe.py
│       ├── processing/            # Image processing
│       │   └── shoreline_extractor.py
│       ├── models/                # AI models
│       │   ├── architectures/     # Network architectures
│       │   │   ├── cnn.py
│       │   │   └── gan.py
│       │   ├── trainers/          # Training logic
│       │   │   ├── base.py
│       │   │   ├── gan_trainer.py
│       │   │   └── cnn_trainer.py
│       │   ├── configs/           # Configuration classes
│       │   │   └── training_config.py
│       │   └── data/              # Data utilities
│       │       └── data_generator.py
│       └── config.py              # Configuration
├── images/
│   ├── fractal_images/            # Generated fractal images
│   └── shoreline_images/          # Extracted shoreline images
├── notebooks/                      # Jupyter notebooks for exploration
└── docs/                           # Documentation and research papers
```

## Installation

Requires Python 3.10 or higher.

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/Dyretna/AI-fractals.git
cd AI-fractals

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### GPU/CUDA Support (Recommended for Training)

For faster training with NVIDIA GPUs:

```bash
# 1. Install TensorFlow with CUDA support
pip install tensorflow[and-cuda]

# 2. Configure environment (copy and edit .env file)
cp .env.example .env
# Edit .env and set NVIDIA_BASE path (see .env.example for instructions)

# 3. Activate environment before running Python
source activate_env.sh

# 4. Verify GPU setup
python scripts/check_hardware.py
```

**Important:** Always run `source activate_env.sh` before training to enable GPU support.

See [GPU-SETUP-SUCCESS.md](GPU-SETUP-SUCCESS.md) for detailed setup instructions.

See [docs/GPU-CUDA-Setup.md](docs/GPU-CUDA-Setup.md) for detailed GPU configuration instructions.

**The package automatically detects and configures GPU/CPU - no manual setup needed!**

## Usage

### Generating Fractals

```python
from ai_fractals.generators import MandelbrotGenerator, JuliaGenerator

# Generate Mandelbrot set
mandelbrot = MandelbrotGenerator(width=800, height=600, max_iter=256)
img = mandelbrot.generate(xmin=-2.5, xmax=1.0, ymin=-1.0, ymax=1.0)

# Generate Julia set
julia = JuliaGenerator(width=800, height=600, max_iter=256)
img = julia.generate(c=-0.7+0.27j, xmin=-1.5, xmax=1.5, ymin=-1.0, ymax=1.0)
```

### Training AI Models

Train GANs or CNNs on fractal datasets with automatic GPU/CPU configuration:

```python
from ai_fractals import setup_hardware, GANTrainer, GANConfig

# Setup hardware (automatically detects and configures GPU/CPU)
hw_config = setup_hardware()

# Configure training
config = GANConfig(
    data_dir="images/shoreline_images",
    epochs=100,
    batch_size=32,  # Auto-adjusted for your hardware
    latent_dim=100,
    learning_rate=0.0002
)

# Train (automatically uses GPU if available, falls back to CPU)
trainer = GANTrainer(config)
trainer.train()
```

See [scripts/train_gan_example.py](scripts/train_gan_example.py) for a complete example.

# Generate samples
samples = trainer.generate_samples(n=10)

# Save checkpoint
trainer.save_checkpoint()
```

Or for CNN training:

```python
from ai_fractals.models import CNNTrainer, CNNConfig

config = CNNConfig(
    data_dir="images/shoreline_images",
    epochs=50,
    batch_size=32,
    patience=10
)

trainer = CNNTrainer(config)
trainer.train()
```

## Research Background

This project implements concepts from AI-enhanced fractal geometry, where machine learning models are trained on traditional fractal patterns to:

1. **Learn fractal characteristics**: CNNs extract features from fractal shorelines
2. **Generate new patterns**: GANs create novel pseudo-fractals with learned properties
3. **Analyze complexity**: Study the fractal dimension and properties of generated patterns

The approach bridges classical fractal mathematics with modern deep learning, enabling exploration of new fractal-like structures.

## Configuration

Set your project directory in a `.env` file:

```
PROJECT_DIR=/path/to/AI-fractals
```

## License

See [LICENSE](LICENSE) file for details.

## Author

**Jesper Anteryd** (jesper.anteryd@proton.me)

Inspired by the research of **Douglas C. Youvan**

## References

- Youvan, D.C. "AI-Enhanced Fractal Geometry: Merging Machine Learning with Traditional Fractal Mathematics"
- [Project Repository](https://github.com/Dyretna/AI-fractals)
