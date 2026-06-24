"""
Tile-search strategies for fractal exploration
----------------------------------------------

This module defines a small hierarchy of tile-search strategies used to
navigate the fractal parameter space. A tile-search strategy receives:

- a low-resolution fractal generator
- a quality evaluator
- the current bounds (xmin, xmax, ymin, ymax)

It subdivides the region into a grid, evaluates each tile, and selects one
tile to explore further. Different strategies implement different selection
policies (e.g., with or without jitter).

A tile-search step returns a dictionary:

    {
        "bounds": (xmin, xmax, ymin, ymax),
        "score": float,
        "accept": bool,
        "metrics": dict
    }

These strategies are used by ShorelineDatasetBuilder to recursively zoom into
interesting regions of the fractal.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.generators import BaseFractalGenerator

Bounds = Tuple[float, float, float, float]
TileResult = Dict[str, Any]


# ======================================================================
# Base class
# ======================================================================


class BaseTileSearch(ABC):
    """
    Abstract base class for tile-search strategies.

    A tile-search strategy performs:
    1. Grid subdivision of the current bounds.
    2. Raw tile generation using a low-resolution fractal generator.
    3. Scoring and acceptance using a FractalQualityEvaluator.
    4. Selection of one tile to explore further.

    Subclasses may override the selection logic (e.g., jittering).
    """

    def __init__(
        self,
        tile_gen: BaseFractalGenerator,
        evaluator: FractalQualityEvaluator,
        n_tiles: int = 5,
        top_k: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        tile_gen : BaseFractalGenerator
            Low-resolution generator used to compute raw iteration tiles.

        evaluator : FractalQualityEvaluator
            Evaluator that scores raw tiles and determines acceptance.

        n_tiles : int
            Grid resolution (n_tiles x n_tiles).

        top_k : int
            Number of fallback candidates when no tile is accepted.
        """
        self.tile_gen = tile_gen
        self.evaluator = evaluator
        self.n_tiles = int(n_tiles)
        self.top_k = int(top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, bounds: Bounds) -> TileResult:
        """Perform one tile-search step."""
        tiles = self._tile_and_score(*bounds)
        chosen = self._select_tile(tiles)
        return chosen

    # ------------------------------------------------------------------
    # Tile scoring
    # ------------------------------------------------------------------

    def _tile_and_score(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> List[TileResult]:
        """
        Subdivide bounds into an n_tiles x n_tiles grid and evaluate each tile.
        """
        xs = np.linspace(xmin, xmax, self.n_tiles + 1)
        ys = np.linspace(ymin, ymax, self.n_tiles + 1)

        tiles: List[TileResult] = []

        for row in range(self.n_tiles):
            for col in range(self.n_tiles):
                tile_bounds: Bounds = (xs[col], xs[col + 1], ys[row], ys[row + 1])

                raw = self.tile_gen.generate_raw(*tile_bounds)
                score, accept, metrics = self.evaluator.evaluate(raw)

                tiles.append(
                    {
                        "bounds": tile_bounds,
                        "score": float(score),
                        "accept": bool(accept),
                        "metrics": metrics,
                    }
                )

        return tiles

    # ------------------------------------------------------------------
    # Selection policy (to be overridden)
    # ------------------------------------------------------------------

    @abstractmethod
    def _select_tile(self, tiles: List[TileResult]) -> TileResult:
        """
        Select one tile from the list of evaluated tiles.

        Subclasses implement the selection policy.
        """
        ...

    def __str__(self):
        rows = [f"{self.__class__.__name__}:"]
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue

            if isinstance(v, (int, float, str, bool)):
                val = v
            else:
                val = type(v).__name__

            rows.append(f"  {k}: {val}")

        return "\n".join(rows)


# ======================================================================
# Basic strategy (no jitter)
# ======================================================================


class TileSearchBasic(BaseTileSearch):
    """
    Basic tile-search strategy without jitter.

    Selection policy:
    - If any tiles are accepted, choose randomly among accepted ones.
    - Otherwise, choose randomly among the top-k highest scoring tiles.
    """

    def _select_tile(self, tiles: List[TileResult]) -> TileResult:
        accepted = [t for t in tiles if t["accept"]]

        if accepted:
            pool = accepted
        else:
            pool = sorted(tiles, key=lambda t: t["score"], reverse=True)[: self.top_k]

        return random.choice(pool)


# ======================================================================
# Jittered strategy
# ======================================================================


class TileSearchJittered(BaseTileSearch):
    """
    Tile-search strategy with jitter applied to the chosen tile's bounds.

    Jitter increases exploration diversity by slightly perturbing the tile
    bounds. The jitter magnitude is proportional to the tile width and height,
    ensuring scale-invariant behavior.
    """

    def __init__(
        self,
        tile_gen: BaseFractalGenerator,
        evaluator: FractalQualityEvaluator,
        n_tiles: int = 5,
        top_k: int = 5,
        jitter_scale: float = 0.05,
    ) -> None:
        super().__init__(tile_gen, evaluator, n_tiles, top_k)
        self.jitter_scale = float(jitter_scale)

    def _select_tile(self, tiles: List[TileResult]) -> TileResult:
        # Use the basic selection logic
        accepted = [t for t in tiles if t["accept"]]

        if accepted:
            pool = accepted
        else:
            pool = sorted(tiles, key=lambda t: t["score"], reverse=True)[: self.top_k]

        chosen = random.choice(pool)
        chosen["bounds"] = self._jitter_bounds(chosen["bounds"], self.jitter_scale)
        return chosen

    # ------------------------------------------------------------------
    # Jitter helper
    # ------------------------------------------------------------------

    def _jitter_bounds(self, bounds: Bounds, scale: float) -> Bounds:
        """Apply a small random perturbation to tile bounds."""
        xmin, xmax, ymin, ymax = bounds
        w = xmax - xmin
        h = ymax - ymin

        jx = (np.random.rand() - 0.5) * w * scale
        jy = (np.random.rand() - 0.5) * h * scale

        return (xmin + jx, xmax + jx, ymin + jy, ymax + jy)
