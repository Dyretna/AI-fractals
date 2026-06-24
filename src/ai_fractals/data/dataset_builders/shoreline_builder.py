"""
ShorelineDatasetBuilder
-----------------------

This module generates a dataset of shoreline images for training a
self-supervised CNN that learns geometry-based representations of fractals.

A shoreline is a grayscale edge map extracted from raw fractal iteration
data. It captures only the mathematical boundary structure of the fractal,
making it ideal for learning shape-invariant embeddings.

Why this dataset exists:
- A CNN is trained on shoreline images to learn geometry-focused features.
- The model outputs an embedding vector for each shoreline.
- Similar fractal structures produce similar embeddings.
- These embeddings are later clustered (e.g., UMAP + HDBSCAN) to form
  automatic fractal categories such as “spiral”, “bulb”, or “valley”.
- The resulting labels or embeddings can be used to condition generative
  models (GANs, VAEs) or explore the latent space.

Why we generate shorelines directly:
- RGB renders are unnecessary for geometry learning.
- Shorelines are lightweight, fast to compute, and contain only essential
  structural information.
- Quality evaluation removes uninformative tiles (low entropy, low edge
  density, fully inside/outside).

The builder performs:
1. Tile-search to explore the fractal space and locate interesting regions.
2. Raw fractal tile generation at a configurable resolution.
3. Shoreline extraction using edge detection.
4. Quality evaluation.
5. Saving via ShorelineSaver.

The output is a geometry-focused dataset suitable for training
self-supervised CNNs that produce fractal embeddings.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ai_fractals.data.savers import ShorelineSaver

from .base import BaseDatasetBuilder


class ShorelineDatasetBuilder(BaseDatasetBuilder):
    """
    Dataset builder for shoreline (edge-map) fractal datasets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.saver = ShorelineSaver(
            output_dir=self.output_dir,
            fractal_type=self.fractal_type,
            colormap=self.colormap,
            max_iter=self.hires_generator.max_iter,
        )

    # ------------------------------------------------------------------
    # Tile Processing
    # ------------------------------------------------------------------

    def _process_tile(self, chosen: dict):
        xmin, xmax, ymin, ymax = chosen["bounds"]

        # raw iteration data
        raw = self.hires_generator.generate_raw(xmin, xmax, ymin, ymax)

        # evaluate raw
        score, passed, metrics = self.evaluator.evaluate(raw)

        if not passed:
            return None, score, False, metrics

        # extract shoreline after passing
        shoreline = self.evaluator.detector.detect(raw)
        shoreline = shoreline.astype(np.uint8)

        return shoreline, score, True, metrics

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save(self, shoreline, chosen, score, metrics):
        ts = datetime.now().isoformat()
        cid = datetime.fromisoformat(ts).strftime("%y%m%d%H%M%S")

        root = f"{cid}_iter{self.hires_generator.max_iter}"
        name = self.output_dir / f"{root}_d{self.depth:02d}"

        self.saver.save_img(shoreline, name)

        if self.save_metadata:
            self.saver.save_metadata(
                name=name,
                resolution=f"({self.hires_generator.width}, {self.hires_generator.height})",
                bounds=chosen["bounds"],
                score=score,
                metrics=metrics,
                ts=ts,
                cid=cid,
            )
