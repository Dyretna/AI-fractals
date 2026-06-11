# Composite quality scoring for fractal images.
# Weights based on Youvan (2024):
#   fractal dimension 35%  (Section 6.1)
#   Shannon entropy   25%  (Section 6.4)
#   pixel variance    20%  (Section 6.1)
#   edge density      20%  (Section 4.2)
#
# Pass generate_raw() output for analysis, generate() output for display.
#
# Gate: boundary_presence check runs first on raw grayscale.
# Pixels == 255 are inside the set (iteration == max_iter).
# If inside_ratio < 0.02 the tile is fully outside - no boundary.
# If inside_ratio > 0.98 the tile is fully inside - no boundary.
# Either case -> score = 0, skip all other metrics.

from typing import Dict, Tuple

import cv2
import numpy as np

from .complexity_measures import calculate_entropy, entropy_score
from .fractal_dimension import fractal_dimension, fractal_dimension_score
from .statistical_properties import analyze_statistical_properties, variance_score


class FractalQualityEvaluator:
    def __init__(
        self,
        quality_threshold: float = 0.3,
        min_edge_ratio: float = 0.03,
        max_edge_ratio: float = 0.45,
        min_inside_ratio: float = 0.0001,
        max_inside_ratio: float = 0.9999,
    ):
        self.quality_threshold = quality_threshold
        self.min_edge_ratio = min_edge_ratio
        self.max_edge_ratio = max_edge_ratio
        self.min_inside_ratio = min_inside_ratio
        self.max_inside_ratio = max_inside_ratio

    def _boundary_present(self, raw: np.ndarray) -> tuple[bool, float]:
        # raw is uint8 from generate_raw(): value 255 == inside set (iteration == max_iter)
        inside_ratio = float(np.sum(raw == 255) / raw.size)
        ok = self.min_inside_ratio <= inside_ratio <= self.max_inside_ratio
        return ok, inside_ratio

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def edge_density(self, image: np.ndarray) -> float:
        gray = self._to_gray(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return float(np.sum(edges > 0) / edges.size)

    def _edge_score(self, density: float) -> float:
        lo, hi = self.min_edge_ratio, self.max_edge_ratio
        if density < lo or density > hi:
            return 0.0
        mid = (lo + hi) / 2.0
        return 1.0 - abs(density - mid) / ((hi - lo) / 2.0)

    def evaluate(self, image: np.ndarray) -> Tuple[float, bool, Dict]:
        gray = self._to_gray(image)

        # Gate: boundary must be present before computing any other metric.
        boundary_ok, inside_ratio = self._boundary_present(gray)
        if not boundary_ok:
            return 0.0, False, {"inside_ratio": inside_ratio, "boundary_present": False}

        dim = fractal_dimension(gray)
        entropy = calculate_entropy(gray)
        stats = analyze_statistical_properties(gray)
        density = self.edge_density(image)

        dim_s = fractal_dimension_score(dim)
        ent_s = entropy_score(entropy)
        var_s = variance_score(stats)
        edge_s = self._edge_score(density)

        score = 0.35 * dim_s + 0.25 * ent_s + 0.20 * var_s + 0.20 * edge_s

        metrics = {
            "inside_ratio": inside_ratio,
            "boundary_present": True,
            "fractal_dimension": dim,
            "entropy": entropy,
            "std": stats["std"],
            "edge_density": density,
            "dim_score": dim_s,
            "entropy_score": ent_s,
            "variance_score": var_s,
            "edge_score": edge_s,
        }

        return float(score), score >= self.quality_threshold, metrics
