# Dataset
This directory does not contain any datasets.
The full dataset used in this project consists of tens of thousands of generated fractal images, and the total size is far too large to distribute through GitHub.

To work with this project, you will need to generate your own dataset using the dataset builder tools included in the repository. The builder will automatically create the required directory structure, render fractal images, extract shorelines, and prepare everything needed for training and embedding generation.

A pretrained shoreline autoencoder (4.1 MB) is available in the /models directory.
This allows you to use embeddings immediately without training your own model.
