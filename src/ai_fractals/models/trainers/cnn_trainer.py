"""CNN Trainer for fractal analysis."""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from ai_fractals.hardware_config import get_hardware_config, get_optimal_batch_size
from ai_fractals.models.architectures import build_cnn
from ai_fractals.models.configs import CNNConfig
from ai_fractals.models.data import datagen

from .base import BaseTrainer


class CNNTrainer(BaseTrainer):
    """Trainer for Convolutional Neural Networks."""

    def __init__(self, config: CNNConfig):
        super().__init__(config, name="cnn")
        self.config: CNNConfig = config

        # Get hardware configuration
        self.hw_config = get_hardware_config()

        # Adjust batch size if needed for hardware
        if hasattr(config, "auto_batch_size") and config.auto_batch_size:
            optimal_batch = get_optimal_batch_size(
                base_size=config.batch_size, image_size=config.image_size
            )
            if optimal_batch != config.batch_size:
                print(
                    f"Adjusting batch size: {config.batch_size} → {optimal_batch} (for hardware)"
                )
                config.batch_size = optimal_batch

        # Build and compile model (within strategy for multi-GPU)
        strategy = self.hw_config.get_device_strategy()
        with strategy.scope():
            self.model = build_cnn(config.input_shape)
            self.model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

        # Data generators
        self.train_generator = datagen.flow_from_directory(
            config.data_dir,
            target_size=config.image_size,
            color_mode="grayscale",
            batch_size=config.batch_size,
            class_mode="input",
            subset="training",
        )

        self.val_generator = None
        if config.validation_split > 0:
            self.val_generator = datagen.flow_from_directory(
                config.data_dir,
                target_size=config.image_size,
                color_mode="grayscale",
                batch_size=config.batch_size,
                class_mode="input",
                subset="validation",
            )

    def train(self):
        """Train the CNN."""
        print(f"Starting CNN training for {self.config.epochs} epochs...")

        callbacks = [
            ModelCheckpoint(
                str(self.output_dir / "cnn_best.keras"),
                monitor=self.config.monitor,
                save_best_only=True,
            ),
            EarlyStopping(
                monitor=self.config.monitor,
                patience=self.config.patience,
                restore_best_weights=True,
            ),
        ]

        self.model.fit(
            self.train_generator,
            epochs=self.config.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
        )

        self.save_model(self.model, "cnn_final")
        print("\n✓ Training complete!")

    def generate_samples(self, n_samples: int = 10):
        """Generate reconstructions (for autoencoder)."""
        # Get some test images
        test_batch = next(self.train_generator)[:n_samples]
        return self.model.predict(test_batch, verbose=0)
