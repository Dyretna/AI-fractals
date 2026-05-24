"""
Multi-scale analysis for self-similarity validation.
Implementation from Youvan (2024), Section 6.1.
"""

from typing import List

import cv2
import numpy as np

from .complexity_measures import calculate_entropy


def check_multiscale_consistency(img: np.ndarray, scales: List[float] = None) -> float:
    """
    Check if complexity persists across scales (self-similarity test).

    From paper Section 6.1:
    "To analyze the self-similarity of AI-generated fractals, we can perform
    multi-scale analysis to observe if the patterns repeat at various levels
    of magnification."

    True fractals maintain similar complexity at different scales.

    Args:
        img: Grayscale image
        scales: List of scale factors (default: [1.0, 0.5, 0.25])

    Returns:
        Consistency score [0, 1] - higher means more self-similar
    """
    if scales is None:
        scales = [1.0, 0.5, 0.25]

    entropies = []

    for scale in scales:
        if scale < 1.0:
            # Downscale image
            new_size = (
                max(1, int(img.shape[1] * scale)),
                max(1, int(img.shape[0] * scale)),
            )
            scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            scaled = img

        entropy = calculate_entropy(scaled)
        entropies.append(entropy)

    if len(entropies) < 2:
        return 0.0

    # Self-similar images have consistent entropy across scales
    # Lower variance = better consistency
    entropy_variance = np.var(entropies)

    # Convert to score: lower variance = higher score
    # Using exponential decay for variance penalty
    consistency_score = np.exp(-entropy_variance / 0.5)

    return float(consistency_score)


def analyze_scale_invariance(img: np.ndarray, scales: List[float] = None) -> dict:
    """
    Comprehensive multi-scale analysis.

    From paper Section 6.1:
    "Visualizing AI-generated fractals at different scales can reveal
    the emergent properties and self-similarity inherent in the patterns."

    Args:
        img: Grayscale image
        scales: List of scale factors

    Returns:
        Dictionary with scale analysis results
    """
    if scales is None:
        scales = [1.0, 0.75, 0.5, 0.25]

    results = {
        "scales": scales,
        "entropies": [],
        "means": [],
        "stds": [],
        "consistency_score": 0.0,
    }

    for scale in scales:
        if scale < 1.0:
            new_size = (
                max(1, int(img.shape[1] * scale)),
                max(1, int(img.shape[0] * scale)),
            )
            scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            scaled = img

        results["entropies"].append(float(calculate_entropy(scaled)))
        results["means"].append(float(np.mean(scaled)))
        results["stds"].append(float(np.std(scaled)))

    # Calculate overall consistency
    results["consistency_score"] = check_multiscale_consistency(img, scales)

    return results


def visualize_multiscale(img: np.ndarray, scales: List[float] = None):
    """
    Create visualization of image at multiple scales.

    From paper Section 6.1 (Figure visualization).

    Args:
        img: Grayscale image
        scales: List of scale factors

    Returns:
        List of scaled images
    """
    if scales is None:
        scales = [1.0, 0.5, 0.25]

    scaled_images = []

    for scale in scales:
        if scale < 1.0:
            new_size = (
                max(1, int(img.shape[1] * scale)),
                max(1, int(img.shape[0] * scale)),
            )
            scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            scaled = img

        scaled_images.append(scaled)

    return scaled_images
