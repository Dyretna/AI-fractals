"""
Fractal dimension calculation using box-counting method.
Implementation from Youvan (2024), Section 6.1.
"""

from typing import List

import numpy as np


def box_count_cpu(img: np.ndarray, box_size: int) -> int:
    """CPU fallback for box counting."""

    count = 0
    for i in range(0, img.shape[0], box_size):
        for j in range(0, img.shape[1], box_size):
            if np.sum(img[i : i + box_size, j : j + box_size]) > 0:
                count += 1
    return count


def box_count(img: np.ndarray, box_size: int) -> int:
    """
    Count non-empty boxes of given size covering the image.

    From paper Section 6.1: Box-Counting Method

    Args:
        img: Binary or grayscale image (2D numpy array)
        box_size: Size of boxes to use for counting

    Returns:
        Number of non-empty boxes
    """

    return box_count_cpu(img, box_size)


def fractal_dimension(img: np.ndarray, box_sizes: List[int] = None) -> float:
    """
    Calculate fractal dimension using box-counting method.

    From paper Section 6.1:
    "The fractal dimension quantifies the complexity of a fractal by
    measuring how detail in the fractal changes with scale."

    Good fractal shorelines have dimension between 1.2 and 1.8.

    Args:
        img: Binary or grayscale image (2D numpy array)
        box_sizes: List of box sizes to use (default: [2, 4, 8, 16, 32, 64])

    Returns:
        Fractal dimension (float)
    """

    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32, 64]

    if img.max() > 1:
        img = (img > img.mean()).astype(int)

    counts = [box_count(img, size) for size in box_sizes]

    if all(c == 0 for c in counts):
        return 0.0

    valid_pairs = [(bs, c) for bs, c in zip(box_sizes, counts) if c > 0]
    if len(valid_pairs) < 2:
        return 0.0

    valid_box_sizes, valid_counts = zip(*valid_pairs)
    coeffs = np.polyfit(np.log(valid_box_sizes), np.log(valid_counts), 1)
    dimension = -coeffs[0]

    return float(dimension)


def is_valid_fractal_dimension(dimension: float) -> bool:
    """
    Check if fractal dimension is in valid range for good shorelines.

    From paper: Good shorelines have 1.2 < dimension < 1.8
    Adjusted: Accept slightly higher for flexibility: 1.0 < dimension < 2.0

    Args:
        dimension: Fractal dimension to check

    Returns:
        True if in valid range
    """
    if dimension is None or np.isnan(dimension) or np.isinf(dimension):
        return False
    return 1.0 < dimension < 2.0


def fractal_dimension_score(dimension: float) -> float:
    """Convert fractal dimension to normalized score [0, 1]."""
    if not is_valid_fractal_dimension(dimension):
        return 0.0

    # Optimal dimension is around 1.5 (middle of range)
    optimal = 1.5
    distance = abs(dimension - optimal)
    max_distance = 0.3  # Distance from 1.5 to boundaries

    score = 1.0 - (distance / max_distance)
    return max(0.0, min(1.0, score))
