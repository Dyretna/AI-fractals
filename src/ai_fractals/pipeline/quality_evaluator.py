"""
Fractal quality evaluator using methods from Youvan (2024).
Implements comprehensive quality scoring for automatic filtering.
"""

from typing import Dict, Tuple

import cv2
import numpy as np

from ..analysis import (
    analyze_statistical_properties,
    calculate_entropy,
    check_multiscale_consistency,
    fractal_dimension,
    is_valid_fractal_dimension,
)


class FractalQualityEvaluator:
    """
    Evaluate fractal image quality using paper-based metrics.

    From Youvan (2024) "AI-Enhanced Fractal Geometry":
    - Fractal dimension (Section 6.1)
    - Shannon entropy (Section 6.4)
    - Statistical properties (Section 6.1)
    - Edge density (Section 4.2)
    - Multi-scale consistency (Section 6.1)
    """

    def __init__(
        self,
        quality_threshold: float = 0.65,
        min_entropy: float = 3.0,
        min_std: float = 50.0,
        min_edge_ratio: float = 0.05,
        max_edge_ratio: float = 0.40,
    ):
        """
        Initialize evaluator with thresholds from paper.

        Args:
            quality_threshold: Minimum composite score for acceptance
            min_entropy: Minimum Shannon entropy (paper: 3.0)
            min_std: Minimum standard deviation (paper: 50.0)
            min_edge_ratio: Minimum edge density ratio
            max_edge_ratio: Maximum edge density ratio
        """
        self.quality_threshold = quality_threshold
        self.min_entropy = min_entropy
        self.min_std = min_std
        self.min_edge_ratio = min_edge_ratio
        self.max_edge_ratio = max_edge_ratio

        # Weights from updated plan (paper-based)
        self.w_dimension = 0.35
        self.w_entropy = 0.25
        self.w_variance = 0.20
        self.w_edge = 0.15
        self.w_scale = 0.05

    def extract_shoreline_canny(self, image: np.ndarray) -> np.ndarray:
        """
        Extract shoreline using Canny edge detection.

        From paper Section 4.2:
        "The Canny edge detection algorithm is effective for extracting
        edges from images."

        Args:
            image: Input fractal image

        Returns:
            Binary edge image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur (paper: (5,5) kernel)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection (paper: thresholds 50, 150)
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def calculate_edge_density(self, image: np.ndarray) -> float:
        """
        Calculate edge density ratio.

        Args:
            image: Input fractal image

        Returns:
            Edge density (ratio of edge pixels to total pixels)
        """
        edges = self.extract_shoreline_canny(image)
        edge_density = np.sum(edges > 0) / edges.size
        return float(edge_density)

    def evaluate(self, image: np.ndarray) -> Tuple[float, bool, Dict]:
        """
        Comprehensive quality evaluation using paper's methods.

        Args:
            image: Fractal image to evaluate (grayscale or BGR)

        Returns:
            Tuple of (composite_score, accept, metrics_dict)
        """
        # Ensure grayscale for analysis
        if len(image.shape) == 3:
            # Handle both BGR (from OpenCV) and RGB (from matplotlib)
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Normalize to 0-255 if needed
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)

        # 1. Fractal Dimension (35% weight - most important)
        fractal_dim = fractal_dimension(gray)
        if is_valid_fractal_dimension(fractal_dim):
            # Optimal dimension ~1.5 (middle of 1.2-1.8 range)
            dim_distance = abs(fractal_dim - 1.5) / 0.3
            dim_score = max(0.0, 1.0 - dim_distance)
        else:
            dim_score = 0.0

        # 2. Shannon Entropy (25% weight)
        entropy = calculate_entropy(gray)
        entropy_score = min(entropy / 5.0, 1.0)  # Normalize, cap at 5

        # 3. Statistical Properties (20% weight)
        stats = analyze_statistical_properties(gray)
        variance_score = min(stats["std"] / 100.0, 1.0)  # Normalize

        # 4. Edge Density (15% weight)
        edge_ratio = self.calculate_edge_density(image)
        if self.min_edge_ratio < edge_ratio < self.max_edge_ratio:
            edge_score = 1.0
        else:
            edge_score = 0.3  # Partial credit

        # 5. Multi-scale Consistency (5% weight)
        scale_score = check_multiscale_consistency(gray)

        # Weighted composite score
        composite_score = (
            self.w_dimension * dim_score
            + self.w_entropy * entropy_score
            + self.w_variance * variance_score
            + self.w_edge * edge_score
            + self.w_scale * scale_score
        )

        # Paper-based acceptance criteria (ALL must be met)
        accept = (
            composite_score > self.quality_threshold
            and is_valid_fractal_dimension(fractal_dim)
            and entropy > self.min_entropy
            and stats["std"] > self.min_std
        )

        # Comprehensive metrics dictionary
        metrics = {
            "composite_score": float(composite_score),
            "fractal_dimension": float(fractal_dim),
            "fractal_dimension_valid": is_valid_fractal_dimension(fractal_dim),
            "entropy": float(entropy),
            "entropy_sufficient": entropy > self.min_entropy,
            "statistical_properties": stats,
            "variance_sufficient": stats["std"] > self.min_std,
            "edge_density": float(edge_ratio),
            "edge_density_valid": self.min_edge_ratio
            < edge_ratio
            < self.max_edge_ratio,
            "multiscale_consistency": float(scale_score),
            "acceptance_criteria_met": {
                "composite_score": composite_score > self.quality_threshold,
                "fractal_dimension_range": is_valid_fractal_dimension(fractal_dim),
                "min_entropy": entropy > self.min_entropy,
                "min_variance": stats["std"] > self.min_std,
            },
        }

        return composite_score, accept, metrics


def quality_score(image: np.ndarray) -> Tuple[float, bool, Dict]:
    """
    Convenience function for quality evaluation.

    Args:
        image: Fractal image to evaluate

    Returns:
        Tuple of (score, accept, metrics)
    """
    evaluator = FractalQualityEvaluator()
    return evaluator.evaluate(image)
