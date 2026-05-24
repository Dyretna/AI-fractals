"""GAN Trainer for fractal generation."""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from ai_fractals.hardware_config import get_hardware_config, get_optimal_batch_size
from ai_fractals.models.architectures import build_discriminator, build_generator
from ai_fractals.models.configs import GANConfig
from ai_fractals.models.data import datagen

from .base import BaseTrainer


class GANTrainer(BaseTrainer):
    """Trainer for Generative Adversarial Networks."""

    def __init__(self, config: GANConfig):
        super().__init__(config, name="gan")
        self.config: GANConfig = config

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

        # Build models (within strategy for multi-GPU)
        strategy = self.hw_config.get_device_strategy()
        with strategy.scope():
            self.generator = build_generator(
                config.latent_dim, output_shape=config.input_shape
            )
            self.discriminator = build_discriminator(config.input_shape)

            # Compile discriminator
            self.discriminator.compile(
                optimizer=Adam(
                    learning_rate=config.discriminator_lr, beta_1=config.beta_1
                ),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Build GAN
            self.discriminator.trainable = False
            self.gan = Sequential([self.generator, self.discriminator], name="gan")
            self.gan.compile(
                optimizer=Adam(learning_rate=config.generator_lr, beta_1=config.beta_1),
                loss="binary_crossentropy",
            )
            self.discriminator.trainable = True

        # Data generator
        self.image_generator = datagen.flow_from_directory(
            config.data_dir,
            target_size=config.image_size,
            color_mode="grayscale",
            class_mode=None,
            batch_size=config.batch_size,
        )

    def train(self):
        """Train the GAN."""
        print(f"Starting GAN training for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            # Get real images
            real_images = next(self.image_generator)
            if real_images.shape[-1] != self.config.input_shape[-1]:
                real_images = np.expand_dims(real_images[..., 0], axis=-1)

            # Generate fake images
            noise = np.random.normal(
                0, 1, (self.config.batch_size, self.config.latent_dim)
            )
            fake_images = self.generator.predict(noise, verbose=0)

            # Labels
            real_labels = np.ones((self.config.batch_size, 1))
            fake_labels = np.zeros((self.config.batch_size, 1))

            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(
                0, 1, (self.config.batch_size, self.config.latent_dim)
            )
            g_loss = self.gan.train_on_batch(noise, real_labels)

            # Log progress
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"D Loss: {d_loss[0]:.4f}, D Acc: {100 * d_loss[1]:.2f}%, "
                    f"G Loss: {g_loss:.4f}"
                )

            # Save at intervals
            if epoch % self.config.save_interval == 0 and epoch > 0:
                self.save_model(self.generator, f"generator_epoch_{epoch}")

        # Save final
        self.save_model(self.generator, "generator_final")
        self.save_model(self.discriminator, "discriminator_final")
        print("\n✓ Training complete!")

    def generate_samples(self, n_samples: int = 10):
        """Generate sample images."""
        noise = np.random.normal(0, 1, (n_samples, self.config.latent_dim))
        return self.generator.predict(noise, verbose=0)
