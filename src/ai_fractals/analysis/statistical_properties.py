"""
Statistical property analysis for fractal images.
Implementation from Youvan (2024), Section 6.1.
"""

from typing import Dict

import numpy as np


def analyze_statistical_properties(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze statistical properties of fractal image.

    From paper Section 6.1:
    "Analyzing statistical properties involves examining distributions
    of features in the fractal images. Key properties to study include:
    - Mean and Variance: Basic statistical measures
    - Higher-Order Moments: Skewness and kurtosis"

    Args:
        img: Grayscale image (2D numpy array)

    Returns:
        Dictionary with mean, variance, std, skewness, kurtosis
    """
    img_flat = img.flatten().astype(float)

    mean = np.mean(img_flat)
    variance = np.var(img_flat)
    std = np.std(img_flat)

    # Skewness: measure of asymmetry
    if std > 0:
        skewness = np.mean((img_flat - mean) ** 3) / (std**3)
    else:
        skewness = 0.0

    # Kurtosis: measure of "tailedness"
    if variance > 0:
        kurtosis = np.mean((img_flat - mean) ** 4) / (variance**2) - 3
    else:
        kurtosis = 0.0

    return {
        "mean": float(mean),
        "variance": float(variance),
        "std": float(std),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }


def is_sufficient_variance(stats: Dict[str, float], threshold: float = 50.0) -> bool:
    """
    Check if image has sufficient variance to be interesting.

    From paper: Images with std > 50 (on 0-255 scale) are interesting.

    Args:
        stats: Statistical properties dict
        threshold: Minimum standard deviation threshold

    Returns:
        True if variance is sufficient
    """
    return stats["std"] > threshold


def variance_score(stats: Dict[str, float], max_std: float = 100.0) -> float:
    """
    Convert variance to normalized score [0, 1].

    Args:
        stats: Statistical properties dict
        max_std: Maximum expected std for normalization

    Returns:
        Score between 0 and 1
    """
    normalized = min(stats["std"] / max_std, 1.0)
    return normalized


def analyze_spatial_correlation(img: np.ndarray, lag: int = 1) -> float:
    """
    Analyze spatial correlation in image.

    From paper Section 6.1:
    "Spatial Correlation: Analyzing how pixel intensities are correlated
    across different regions of the image."

    Args:
        img: Grayscale image
        lag: Pixel distance for correlation calculation

    Returns:
        Correlation coefficient
    """
    if img.shape[0] <= lag or img.shape[1] <= lag:
        return 0.0

    # Calculate horizontal correlation
    h_corr = np.corrcoef(img[:, :-lag].flatten(), img[:, lag:].flatten())[0, 1]

    # Calculate vertical correlation
    v_corr = np.corrcoef(img[:-lag, :].flatten(), img[lag:, :].flatten())[0, 1]

    # Average correlation
    correlation = (h_corr + v_corr) / 2.0

    return float(correlation) if not np.isnan(correlation) else 0.0
