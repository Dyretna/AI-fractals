"""Data loading and augmentation utilities."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generator(
    rescale=True, augment=True, validation_split=0.2
) -> ImageDataGenerator:
    """Create an image data generator with optional augmentation.

    Args:
        rescale: Whether to rescale pixel values to [0, 1]
        augment: Whether to apply data augmentation
        validation_split: Fraction of data to use for validation

    Returns:
        Configured ImageDataGenerator
    """
    params = {"validation_split": validation_split}

    if rescale:
        params["rescale"] = 1.0 / 255.0

    if augment:
        params.update(
            {
                "rotation_range": 20,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True,
                "fill_mode": "nearest",
            }
        )

    return ImageDataGenerator(**params)


# Pre-configured generators
datagen = create_data_generator(rescale=True, augment=True, validation_split=0.2)
datagen_no_aug = create_data_generator(
    rescale=True, augment=False, validation_split=0.2
)
