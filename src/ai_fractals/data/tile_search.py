# src/ai_fractals/data/tile_search.py

import random

import numpy as np

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.generators import BaseFractalGenerator


class TileSearch:
    """
    Generic tile-search strategy.

    This class performs:
    - grid subdivision of the current bounds
    - raw tile generation using a low-res generator
    - scoring and acceptance using an evaluator
    - fallback logic (top-k selection)
    - random choice among accepted or fallback tiles

    It returns a dictionary describing the chosen tile:
        {
            "bounds": (xmin, xmax, ymin, ymax),
            "score": float,
            "accept": bool,
            "metrics": dict
        }
    """

    def __init__(self, tile_gen, evaluator, n_tiles=5, top_k=3):
        """
        Parameters
        ----------
        tile_gen : BaseFractalGenerator
            Low-resolution generator with .generate_raw(xmin, xmax, ymin, ymax)

        evaluator : FractalQualityEvaluator
            Evaluator with .evaluate(raw) -> (score, accept, metrics)

        n_tiles : int
            Grid size (n_tiles x n_tiles)

        top_k : int
            Number of fallback candidates when no tile is accepted
        """
        self.tile_gen: BaseFractalGenerator = tile_gen
        self.evaluator: FractalQualityEvaluator = evaluator
        self.n_tiles: int = n_tiles
        self.top_k: int = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, bounds):
        """
        Perform one tile-search step.

        Parameters
        ----------
        bounds : tuple
            (xmin, xmax, ymin, ymax)

        Returns
        -------
        dict
            Chosen tile dictionary.
        """
        tiles = self._tile_and_score(*bounds)

        # Accepted tiles
        candidates = [t for t in tiles if t["accept"]]

        # create tiled candidates for further depth search
        # Random choice, since best metrics is not necessarily most interesting
        # Fallback if no tile is over threshold: pick 1 random of top 3
        if candidates:
            pool = candidates
        else:
            pool = sorted(tiles, key=lambda t: t["score"], reverse=True)[: self.top_k]

        chosen = random.choice(pool)
        return chosen

    # ------------------------------------------------------------------
    # tile scoring
    # ------------------------------------------------------------------

    def _tile_and_score(self, xmin, xmax, ymin, ymax):
        """
        Subdivide bounds into n_tiles x n_tiles grid and evaluate each tile.
        """
        xs = np.linspace(xmin, xmax, self.n_tiles + 1)
        ys = np.linspace(ymin, ymax, self.n_tiles + 1)

        tiles = []

        for row in range(self.n_tiles):
            for col in range(self.n_tiles):
                tile_bounds = (xs[col], xs[col + 1], ys[row], ys[row + 1])

                raw = self.tile_gen.generate_raw(*tile_bounds)
                score, accept, metrics = self.evaluator.evaluate(raw)

                tiles.append(
                    {
                        "bounds": tile_bounds,
                        "score": score,
                        "accept": accept,
                        "metrics": metrics,
                    }
                )

        return tiles
