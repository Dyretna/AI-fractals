"""
Complexity measures for fractal quality evaluation.
Implementation from Youvan (2024), Section 6.4.
"""

import numpy as np


def calculate_entropy(img: np.ndarray) -> float:
    """
    Calculate Shannon entropy of image.

    From paper Section 6.4:
    "Entropy: A measure of randomness or unpredictability in the pattern."

    Higher entropy indicates more complex/interesting patterns.
    Paper suggests entropy > 3.0 for interesting fractals.

    Args:
        img: Grayscale image (2D numpy array)

    Returns:
        Shannon entropy value
    """
    # Calculate histogram
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256), density=True)

    # Remove zero entries
    hist = hist[hist > 0]

    # Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def is_sufficient_entropy(entropy: float, threshold: float = 3.0) -> bool:
    """
    Check if entropy meets minimum threshold for interesting fractals.

    From paper: Entropy > 3.0 indicates sufficient complexity.

    Args:
        entropy: Calculated entropy value
        threshold: Minimum entropy threshold

    Returns:
        True if entropy is sufficient
    """
    return entropy > threshold


def entropy_score(entropy: float, max_entropy: float = 5.0) -> float:
    """Convert entropy to normalized score [0, 1]."""

    normalized = min(entropy / max_entropy, 1.0)
    return normalized


def calculate_lacunarity(img: np.ndarray, box_sizes: list = None) -> float:
    """
    Calculate lacunarity - measure of gaps/holes in fractal.

    From paper Section 6.4:
    "Lacunarity: A measure of the gaps or holes in the fractal,
    indicating the texture's heterogeneity."

    Lower lacunarity = more uniform texture
    Higher lacunarity = more varied texture with gaps

    Args:
        img: Binary or grayscale image
        box_sizes: List of box sizes for analysis

    Returns:
        Lacunarity value
    """
    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32]

    # Binarize image
    if img.max() > 1:
        img_binary = (img > img.mean()).astype(int)
    else:
        img_binary = img.astype(int)

    lacunarities = []

    for box_size in box_sizes:
        # Count pixels in each box
        box_masses = []
        for i in range(0, img_binary.shape[0] - box_size + 1, box_size):
            for j in range(0, img_binary.shape[1] - box_size + 1, box_size):
                mass = np.sum(img_binary[i : i + box_size, j : j + box_size])
                box_masses.append(mass)

        if len(box_masses) > 0:
            box_masses = np.array(box_masses)
            mean_mass = np.mean(box_masses)

            if mean_mass > 0:
                # Lacunarity = variance / mean^2
                lacunarity = np.var(box_masses) / (mean_mass**2)
                lacunarities.append(lacunarity)

    if len(lacunarities) > 0:
        # Average lacunarity across scales
        return float(np.mean(lacunarities))
    else:
        return 0.0


def calculate_kolmogorov_complexity_estimate(img: np.ndarray) -> float:
    """
    Estimate Kolmogorov complexity using compression.

    From paper Section 6.4:
    "Kolmogorov Complexity: The length of the shortest algorithm
    that can generate the fractal pattern."

    We estimate this using compression ratio.

    Args:
        img: Grayscale image

    Returns:
        Estimated complexity (compression ratio)
    """
    import zlib

    # Convert to bytes
    img_bytes = img.tobytes()
    original_size = len(img_bytes)

    # Compress
    compressed = zlib.compress(img_bytes, level=9)
    compressed_size = len(compressed)

    # Complexity = how incompressible it is
    # Higher ratio = more complex/random
    complexity = compressed_size / original_size

    return float(complexity)
