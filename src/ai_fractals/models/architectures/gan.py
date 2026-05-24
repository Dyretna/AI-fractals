"""GAN (Generative Adversarial Network) architectures for fractal generation."""

from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Sequential


def build_generator(latent_dim: int, output_shape=(28, 28, 1)):
    """Build the generator network.

    Args:
        latent_dim: Dimension of the latent input vector
        output_shape: Shape of generated images (height, width, channels)

    Returns:
        Generator model
    """
    model = Sequential(
        [
            Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim),
            Reshape((7, 7, 128)),
            Conv2DTranspose(
                128, (4, 4), strides=(2, 2), padding="same", activation="relu"
            ),
            Conv2DTranspose(
                64, (4, 4), strides=(2, 2), padding="same", activation="relu"
            ),
            Conv2DTranspose(
                output_shape[2], (7, 7), padding="same", activation="sigmoid"
            ),
        ],
        name="generator",
    )
    return model


def build_discriminator(input_shape=(128, 128, 1)):
    """Build the discriminator network.

    Args:
        input_shape: Shape of input images (height, width, channels)

    Returns:
        Discriminator model
    """
    model = Sequential(
        [
            Conv2D(64, (3, 3), strides=(2, 2), padding="same", input_shape=input_shape),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    return model
