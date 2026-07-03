# Dataset

This directory does not contain any prebuilt datasets.

The full dataset used in this project consists of thousands of generated fractal
images, and the total size is far too large to distribute through GitHub.

To work with this project, you need to generate your own dataset using the dataset
builder tools included in the repository. The builder automatically creates the
required directory structure, renders fractal images, extracts shorelines, and
prepares everything needed for training and embedding generation.

EDIT: I have refined the pipeline steps, and a ZIP file containing region JSONs
is now saved in this folder. These JSON files include the "bounds" required by
the shoreline and RGB batch generators in /scripts. You can use these to save
time, or generate/enrich them with your own regions.

The regions are mostly based on 7×7 tiles, with some generated using 6×6 and a
smaller number using 5×5 (this gives variety).
