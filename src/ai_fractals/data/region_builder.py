# ai_fractals/data/region_builder.py

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from ai_fractals.analysis import FractalQualityEvaluator
from ai_fractals.search.tile_search import BaseTileSearch


class RegionBuilder:
    """
    RegionBuilder

    Minimal, clean builder that performs tile-search and saves ONLY metadata.
    No RGB renders, no shoreline extraction, no embeddings.

    The output is a directory of JSON files, each describing a fractal region.
    Later modules can regenerate RGB, shoreline, or embeddings from this metadata.

    Parameters
    ----------
    tile_search : BaseTileSearch
        The tile-search strategy responsible for exploring the fractal space.
        It generates low-resolution tiles, scores them, and selects the next
        region to zoom into.

    evaluator : FractalQualityEvaluator
        Evaluates raw tiles produced by the tile-search generator. Determines
        acceptance, scoring, and auxiliary metrics used during exploration.

    output_dir : Path
        Directory where generated images (and optional metadata) will be saved.

    save_min_depth : int, default=2
        Minimum zoom depth at which images are eligible to be saved. Depth is
        incremented each time a tile is accepted and exploration continues.

    save_max_depth : int, default=10
        Maximum zoom depth at which images are saved. When this depth is
        reached, the search is reset to the initial bounds.
    """

    def __init__(
        self,
        *,
        tile_search: BaseTileSearch,
        evaluator: FractalQualityEvaluator,
        output_dir: Path,
        save_min_depth: int = 2,
        save_max_depth: int = 10,
    ):
        # tools
        self.tile_search = tile_search
        self.evaluator = evaluator

        # fractal info
        self.fractal_type = self.tile_search.tile_gen.fractal_type
        self.max_iter = self.tile_search.tile_gen.max_iter

        # depth control
        self.save_min_depth = save_min_depth
        self.save_max_depth = save_max_depth

        # bounds + depth
        self.bounds = self.tile_search.tile_gen.default_bounds()
        self.depth = 0

        # fallback detection
        self.consecutive_fallbacks = 0
        self.max_fallbacks = 10

        # output
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # logging
        self.log = logging.getLogger(__name__)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def run(self, n_regions: int):
        remaining = n_regions
        pbar = tqdm(total=n_regions, desc=f"Generating {self.fractal_type}")

        while remaining > 0:
            saved = self.step()  # returns 1 if saved, else 0
            if saved:
                remaining -= 1
                pbar.update(1)

            if self.consecutive_fallbacks >= self.max_fallbacks:
                self._reset_search()

        pbar.close()

    def step(self) -> int:
        self.tile_search.depth = self.depth
        chosen = self.tile_search.run(self.bounds)

        if chosen["accept"]:
            self.consecutive_fallbacks = 0
        else:
            self.consecutive_fallbacks += 1

        if self.consecutive_fallbacks >= self.max_fallbacks:
            self._reset_search()
            return 0

        # update bounds + depth
        self.bounds = chosen["bounds"]
        self.depth += 1

        # only if we are in accepted depth
        if not (self.save_min_depth <= self.depth <= self.save_max_depth):
            return 0

        self._save_region(chosen)

        if self.depth >= self.save_max_depth:
            self._reset_search()

        return 1

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _reset_search(self):
        self.bounds = self.tile_search.tile_gen.default_bounds()
        self.depth = 0
        self.consecutive_fallbacks = 0

    def _save_region(self, chosen: dict):
        ts = datetime.now().isoformat()
        cid = datetime.fromisoformat(ts).strftime("%y%m%d%H%M%S")

        meta = {
            "timestamp": ts,
            "compact_id": cid,
            "fractal_type": self.fractal_type,
            "max_iter": self.max_iter,
            "depth": self.depth,
            "bounds": chosen.get("bounds", None),
            "score": chosen.get("score", None),
            "metrics": chosen.get("metrics", {}),
        }

        name = f"{cid}_iter{self.max_iter}_d{self.depth:02d}.json"
        path = self.output_dir / name

        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

        self.log.info(f"Saved region: {name}")

    def __str__(self):
        header = "\n" + "=" * 50 + f"\n{self.__class__.__name__}\n" + "=" * 50
        rows = [header]

        def indent(text: str) -> str:
            pad = " " * 4
            return "\n".join(pad + line for line in text.splitlines())

        def block(label, obj):
            rows.append(f"  {label}:")
            rows.append(indent(str(obj)))
            rows.append("")

        block("tile_search", self.tile_search)
        block("evaluator", self.evaluator)
        block("edge_detector", self.evaluator.detector)

        rows.append(f"  fractal_type:   {self.fractal_type}")
        rows.append(f"  max_iter:       {self.max_iter}")
        rows.append(f"  output_dir:     {self.output_dir}")
        rows.append("=" * 50)

        return "\n".join(rows)
