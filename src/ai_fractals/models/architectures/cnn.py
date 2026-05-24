"""CNN (Convolutional Neural Network) architectures for fractal analysis."""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential


def build_cnn(input_shape=(128, 128, 1)):
    """Build a CNN autoencoder for fractal analysis.

    This architecture can be used for feature extraction, denoising,
    or pattern reconstruction of fractal images.

    Args:
        input_shape: Shape of input images (height, width, channels)

    Returns:
        CNN model
    """
    model = Sequential(
        [
            # Encoder
            Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            # Decoder
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(input_shape[2], (3, 3), activation="sigmoid", padding="same"),
        ],
        name="cnn_autoencoder",
    )
    return model
