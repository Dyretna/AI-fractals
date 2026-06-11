"""
Analysis module for fractal quality evaluation.
Implements methods from Youvan (2024) "AI-Enhanced Fractal Geometry".
"""

from .complexity_measures import (
    calculate_entropy,
    calculate_kolmogorov_complexity_estimate,
    calculate_lacunarity,
    entropy_score,
    is_sufficient_entropy,
)
from .evaluator import FractalQualityEvaluator
from .fractal_dimension import (
    box_count,
    fractal_dimension,
    fractal_dimension_score,
    is_valid_fractal_dimension,
)
from .multiscale_analysis import (
    analyze_scale_invariance,
    check_multiscale_consistency,
    visualize_multiscale,
)
from .statistical_properties import (
    analyze_spatial_correlation,
    analyze_statistical_properties,
    is_sufficient_variance,
    variance_score,
)

__all__ = [
    "analyze_scale_invariance",
    "analyze_spatial_correlation",
    "analyze_statistical_properties",
    "box_count",
    "calculate_entropy",
    "calculate_lacunarity",
    "calculate_kolmogorov_complexity_estimate",
    "check_multiscale_consistency",
    "entropy_score",
    "fractal_dimension",
    "fractal_dimension_score",
    "FractalQualityEvaluator",
    "is_valid_fractal_dimension",
    "is_sufficient_entropy",
    "is_sufficient_variance",
    "variance_score",
    "visualize_multiscale",
]
