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
    "box_count",
    "fractal_dimension",
    "is_valid_fractal_dimension",
    "fractal_dimension_score",
    "analyze_statistical_properties",
    "is_sufficient_variance",
    "variance_score",
    "analyze_spatial_correlation",
    "calculate_entropy",
    "is_sufficient_entropy",
    "entropy_score",
    "calculate_lacunarity",
    "calculate_kolmogorov_complexity_estimate",
    "check_multiscale_consistency",
    "analyze_scale_invariance",
    "visualize_multiscale",
]
