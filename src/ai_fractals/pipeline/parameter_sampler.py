"""
Intelligent parameter sampling for fractal generation.
Implements adaptive exploration strategies.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    from .parameter_optimizer import AdaptiveParameterSearch

    _optimizer_available = True
except ImportError:
    _optimizer_available = False


@dataclass
class ParameterRange:
    """Parameter range specification."""

    min_val: float
    max_val: float
    log_scale: bool = False

    def sample(self) -> float:
        """Sample random value from range."""
        if self.log_scale:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            return 10 ** np.random.uniform(log_min, log_max)
        else:
            return np.random.uniform(self.min_val, self.max_val)


class ParameterSampler:
    """
    Intelligent parameter sampler for fractal generation.

    Strategies:
    1. Uniform random sampling (initial exploration)
    2. Adaptive sampling (focus on successful regions)
    3. Zoom-based sampling (explore interesting subregions)
    """

    def __init__(self, fractal_type: str = "mandelbrot", use_optimizer: bool = True):
        self.fractal_type = fractal_type
        self.success_history = []
        self.quality_heatmap = {}
        self.use_optimizer = use_optimizer and _optimizer_available

        # Define parameter spaces based on paper Section 4.1
        if fractal_type == "mandelbrot":
            self.param_ranges = {
                "x_min": ParameterRange(-2.5, 1.0),
                "x_max": ParameterRange(-2.5, 1.0),
                "y_min": ParameterRange(-1.5, 1.5),
                "y_max": ParameterRange(-1.5, 1.5),
                "size": ParameterRange(512, 1024),  # SQUARE images only
                "max_iter": ParameterRange(512, 2048),  # Higher iterations for detail
            }
            # Interesting regions for Mandelbrot
            self.interesting_regions = [
                {"center": (-0.7, 0.0), "range": 2.0},  # Main bulb
                {"center": (-0.1, 0.65), "range": 0.5},  # North antenna
                {"center": (0.285, 0.0), "range": 0.3},  # East valley
                {"center": (-0.75, 0.1), "range": 0.5},  # West bulb
            ]
        else:  # julia
            self.param_ranges = {
                "c_real": ParameterRange(-1.0, 1.0),
                "c_imag": ParameterRange(-1.0, 1.0),
                "x_min": ParameterRange(-2.0, 2.0),
                "x_max": ParameterRange(-2.0, 2.0),
                "y_min": ParameterRange(-2.0, 2.0),
                "y_max": ParameterRange(-2.0, 2.0),
                "size": ParameterRange(512, 1024),  # SQUARE images only
                "max_iter": ParameterRange(512, 2048),  # Higher iterations for detail
            }
            # Interesting Julia set parameters
            self.interesting_regions = [
                {"c": (-0.7, 0.27), "range": 3.0},
                {"c": (-0.4, 0.6), "range": 3.0},
                {"c": (0.285, 0.01), "range": 3.0},
                {"c": (-0.8, 0.156), "range": 3.0},
            ]

        # Initialize optimizer if available
        if self.use_optimizer:
            self.optimizer = AdaptiveParameterSearch(self.param_ranges)
        else:
            self.optimizer = None

    def sample_uniform(self) -> Dict[str, float]:
        """
        Sample parameters uniformly from defined ranges.

        Returns:
            Dictionary of parameter values
        """
        params = {}

        if self.fractal_type == "mandelbrot":
            # Sample center point and zoom
            cx = np.random.uniform(-2.0, 1.0)
            cy = np.random.uniform(-1.5, 1.5)
            zoom = 10 ** np.random.uniform(-0.5, 3.0)  # 0.3 to 1000

            size = 3.0 / zoom
            img_size = int(self.param_ranges["size"].sample())
            params = {
                "x_min": cx - size / 2,
                "x_max": cx + size / 2,
                "y_min": cy - size / 2,
                "y_max": cy + size / 2,
                "width": img_size,  # Square images
                "height": img_size,
                "max_iter": int(self.param_ranges["max_iter"].sample()),
            }
        else:  # julia
            img_size = int(self.param_ranges["size"].sample())
            params = {
                "c_real": self.param_ranges["c_real"].sample(),
                "c_imag": self.param_ranges["c_imag"].sample(),
                "x_min": -2.0,
                "x_max": 2.0,
                "y_min": -2.0,
                "y_max": 2.0,
                "width": img_size,  # Square images
                "height": img_size,
                "max_iter": int(self.param_ranges["max_iter"].sample()),
            }

        return params

    def sample_from_interesting_region(self) -> Dict[str, float]:
        """
        Sample parameters from known interesting regions.

        Returns:
            Dictionary of parameter values
        """
        # Choose random interesting region
        region = np.random.choice(self.interesting_regions)

        if self.fractal_type == "mandelbrot":
            # Add noise around interesting center
            cx = region["center"][0] + np.random.normal(0, region["range"] / 10)
            cy = region["center"][1] + np.random.normal(0, region["range"] / 10)
            zoom = 10 ** np.random.uniform(0.0, 2.5)

            size = region["range"] / zoom
            img_size = int(self.param_ranges["size"].sample())
            params = {
                "x_min": cx - size / 2,
                "x_max": cx + size / 2,
                "y_min": cy - size / 2,
                "y_max": cy + size / 2,
                "width": img_size,  # Square images
                "height": img_size,
                "max_iter": int(self.param_ranges["max_iter"].sample()),
            }
        else:  # julia
            # Sample around interesting c values
            c_real = region["c"][0] + np.random.normal(0, 0.1)
            c_imag = region["c"][1] + np.random.normal(0, 0.1)

            img_size = int(self.param_ranges["size"].sample())
            params = {
                "c_real": c_real,
                "c_imag": c_imag,
                "x_min": -2.0,
                "x_max": 2.0,
                "y_min": -2.0,
                "y_max": 2.0,
                "width": img_size,  # Square images
                "height": img_size,
                "max_iter": int(self.param_ranges["max_iter"].sample()),
            }

        return params

    def sample_zoom_from_successful(self) -> Dict[str, float]:
        """
        Generate zoomed version of successful parameters.

        Returns:
            Dictionary of parameter values
        """
        if not self.success_history:
            return self.sample_uniform()

        # Choose random successful parameter set
        base_params = np.random.choice(self.success_history)

        # Create zoomed version
        if self.fractal_type == "mandelbrot":
            cx = (base_params["xmin"] + base_params["xmax"]) / 2
            cy = (base_params["ymin"] + base_params["ymax"]) / 2
            current_size = base_params["xmax"] - base_params["xmin"]

            # Zoom in by factor of 2-10
            zoom_factor = np.random.uniform(2.0, 10.0)
            new_size = current_size / zoom_factor

            # Add small random offset
            cx += np.random.normal(0, new_size / 4)
            cy += np.random.normal(0, new_size / 4)

            params = {
                "xmin": cx - new_size / 2,
                "xmax": cx + new_size / 2,
                "ymin": cy - new_size / 2,
                "ymax": cy + new_size / 2,
                "width": base_params["width"],
                "height": base_params["height"],
                "max_iter": min(512, int(base_params["max_iter"] * 1.5)),
            }
        else:  # julia
            # For Julia, keep c similar but vary viewport
            params = base_params.copy()
            zoom_factor = np.random.uniform(0.5, 2.0)
            size = 4.0 / zoom_factor
            params["xmin"] = -size / 2
            params["xmax"] = size / 2
            params["ymin"] = -size / 2
            params["ymax"] = size / 2

        return params

    def sample(self, strategy: str = "mixed") -> Dict[str, float]:
        # Use optimizer if enabled and has enough data
        if self.optimizer and strategy in ["mixed", "adaptive"]:
            return self.optimizer.suggest_parameters()

        if strategy == "uniform":
            return self.sample_uniform()
        elif strategy == "interesting":
            return self.sample_from_interesting_region()
        elif strategy == "zoom":
            return self.sample_zoom_from_successful()
        else:  # mixed
            r = np.random.random()
            if r < 0.4:
                return self.sample_from_interesting_region()
            elif r < 0.7:
                return self.sample_uniform()
            else:
                return self.sample_zoom_from_successful()

    def record_success(
        self, params: Dict[str, float], quality: float, fractal_dimension: float = None
    ):
        self.success_history.append(params)

        # Update optimizer if enabled
        if self.optimizer:
            self.optimizer.update(params, quality, fractal_dimension)

        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]

    def get_optimizer_status(self) -> Dict:
        if self.optimizer:
            return self.optimizer.get_status()
        return {"optimizer": "disabled"}
