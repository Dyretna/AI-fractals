"""Base trainer class with ABC.

Simple abstract base for trainers.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTrainer(ABC):
    """Abstract base trainer.

    Subclasses must implement train() and generate_samples().
    """

    def __init__(self, config, name="model"):
        """Initialize trainer.

        Args:
            config: Training configuration object
            name: Name for this trainer instance
        """
        self.config = config
        self.name = name
        self.current_epoch = 0

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self):
        """Train the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def generate_samples(self, n_samples: int = 10):
        """Generate sample outputs. Must be implemented by subclasses."""
        pass

    def save_model(self, model, name: str):
        """Save a Keras model.

        Args:
            model: Keras model to save
            name: Name for the saved model file
        """
        model_path = self.output_dir / f"{name}.keras"
        model.save(model_path)
        print(f"Model saved: {model_path}")
