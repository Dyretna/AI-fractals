# Shoreline Tests — Summary

The shoreline pipeline evaluates fractal regions using Youvan’s composite scoring: fractal dimension, entropy, variance, and edge density. Before scoring, the evaluator checks whether the tile contains a fractal boundary; tiles that are fully inside or outside are rejected immediately. Because all metrics depend on extracted edges, the choice of edge‑detector setup has a huge impact on what passes.

To understand this, five shoreline test configurations were compared. The main variables were:

Gaussian smoothing (on/off, different sigmas)

max_iter in the shoreline generator

optional dilation to thicken thin edges

Metadata was generated at different times with different iteration limits (256 -> 1024), and newer metadata used 7×7 tiles, which naturally produces deeper zoom levels. At high depth, higher max_iter reveals more structure; lower max_iter sometimes fails.

The evaluated/rejected counts:

| Test   | Evaluated | Rejected |
|--------|-----------|----------|
| test 1 | 3854      | 1514     |
| test 2 | 2835      | 2533     |
| test 3 | 3919      | 1449     |
| test 4 | 2894      | 2474     |
| test 5 | 4001      | 1367     |


Test 3 is the only configuration without Gaussian smoothing. It consistently preserves thin edges and captures the most structural detail. Smoothing often removes exactly the fine geometry the evaluator needs, even at low sigma. Dilation (test 5) thickens edges and can increase pass‑rate, but it distorts the geometry and is not ideal for GAN embeddings.

Across all comparisons, test 3 produced the most reliable shoreline structure for both evaluator scoring and downstream GAN training. It is simple, fast, and preserves the geometry needed for structural embeddings. If deeper tiles fail, that indicates max_depth should be lowered rather than smoothing or dilating the edges.

**Chosen setup: Test 3 — liberal Canny, no smoothing.**
