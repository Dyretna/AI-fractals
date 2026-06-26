# Devlog

## First Entry, **June 26, 2026**

I’ve been working on this project for a while now, and today feels like a good moment to stop and actually write down where things are, how they got here, and what the hell I’ve been doing. The codebase has changed a bit since i started out,  So here’s a recap...

---

### Early Days — Naive Dataset Generation

The project started out with finding the article that the repo builds upon,
then using the generators i found to generate a bunch of fractals and save them.

the author shows a `fractal_walk` method that could be used but my implementation was poor - I realized that I needed a better way to *find* interesting regions automatically. That’s when the first tile-search experiments began — messy, notebook-driven prototypes that barely held together but proved the concept.

---

### Shorelines, Augmenters, and Saving Too Many Images

Once I had tile-search working, I started extracting shorelines.
The first version used a standalone augmenter that saved every augmented image to disk. It did the job, but it also produced **massive** datasets — thousands of files per batch, many of them redundant.

---

### The First CNN and VAE — “Holy shit, it works”

This was a fun moment.
felt like gettin' learnt with Ricky:
https://www.youtube.com/watch?v=ewz2cFUUyck

I trained a self-supervised CNN on shoreline images, and the embeddings actually clustered in meaningful ways. Different fractal structures grouped together. The latent space wasn’t random noise — it had shape.

Then I built a shoreline autoencoder (a small VAE), and that worked too.
Not perfect, but enough to prove that the geometry of fractals *can* be learned.

This was the first time the project felt like more than a generator.
It felt like an AI system.


After buildning the CNN i realized that PyTorch can do augmentation on-the-fly, in the dataloader. So the old augmenter was removed.

---

### The Big Refactor — Builders, Strategies, and a Cleaner Design

After removing the augmentation step and simplifying the shoreline pipeline, I took a step back and realized that the RGB and shoreline builders were not as different as they looked. In many ways, they could follow the same overall structure, just with different steps and outputs. They are still separate for now, but a lot of work has gone into making them feel like two variations of the same pipeline rather than two unrelated systems.

Several simplifications came out of this:

- Tile-search was moved into its own module with pluggable strategies (basic, jittered, and more in the future).
- The shoreline builder was reorganized into clear, explicit steps.
- The RGB builder was cleaned up and aligned with the structure of the shoreline builder.
- Old scripts and leftover code were removed.
- The Julia generator had become difficult to maintain due to unstable parameter handling and awkward bounds logic. It created more complexity than value at this stage, so it was removed during the simplification process. The plan is to bring it back later, once the overall pipeline is stable and the parameter strategy is clearer.
- Naming conventions were cleaned up across the codebase.

This refactor made the entire project feel more coherent. The builders now share a common shape, the tile-search logic is isolated and easier to extend, and the codebase is much easier to navigate. Julia will return eventually, but only when the design is ready for her.


---

### Today — Unified Batch Pipelines and a Proper CLI

This brings me to today’s work.

I finally built a **config-driven batch CLI** that can run multiple pipelines in sequence.
Each YAML config now has a `job_type` field (`rgb` or `shoreline`), and the CLI dispatches to the correct builder automatically.

One thing that has become increasingly clear while working on the batch pipeline is how strongly `max_iter` influences the entire generation process. Higher iteration counts allow deeper zooms and more detailed fractal structures, but they also make every step more expensive. This directly affects what values for `min_depth` and `max_depth` make sense for a given profile.

At the same time, `n_tiles` controls how much exploration tile-search performs. It determines how many candidate regions are evaluated before selecting one to zoom into. Unlike `max_iter`, `n_tiles` is not tied to fractal detail, but the two interact in practice: when both are high, the total cost grows quickly. Each tile must be rendered at a reasonable resolution, and with a high iteration count this becomes slow. So increasing both `max_iter` and `n_tiles` at the same time leads to very long generation times. Since every tile becomes smaller with increase of `n_tiles`, this also increases the zoom.

This raises the question of whether the current tile-search strategy is the right one. It works, and it produces good results, but it may not be the most efficient approach when iteration counts are high. It is worth considering whether alternative strategies could reduce the cost without losing the exploratory behavior that tile-search provides.

For now, the important point is that iteration profiles need to reflect these relationships:

- `max_iter` controls detail level and feasible zoom depth.
- This determines what `min_depth` and `max_depth` should be.
- `n_tiles` controls exploration and is a separate heuristic.
- High `max_iter` combined with high `n_tiles` becomes very expensive.

Because of this, it makes sense to define multiple iteration profiles, each with its own parameters. A single config file becomes too limited when different iteration levels require different depth rules and tile-search settings. The cleanest solution is to have several config files, for example:

configs/rgb_256.yaml
configs/rgb_512.yaml
configs/rgb_1024.yaml

Each config defines its own iteration profile, and the batch CLI can run several of them in sequence. This makes it easy to test different combinations of iteration levels and zoom depths, and it keeps the pipeline flexible and easy to extend.

the quest for less expensive search strategies goes on...

---

### what's next
first and foremost:
- tweak on the configs for the batch pipelines, and think about better search strategies

in due time:
- Create the first GAN models (probably a simple DCGAN baseline to start)
- Experiment with conditioning GANs using the CNN embeddings
- Test whether the GAN can follow “directions” — e.g., generate fractals with certain geometric properties

peace out!
