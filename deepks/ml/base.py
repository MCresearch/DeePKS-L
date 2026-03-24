"""Abstract interfaces for ML backends."""

from abc import ABC, abstractmethod


class ModelBackend(ABC):
    """Contract for ML model training/inference backends."""

    @abstractmethod
    def train(self, **kwargs):
        """Run model training with keyword-only configuration."""

    @abstractmethod
    def evaluate(self, **kwargs):
        """Run model evaluation and return backend-specific metrics."""

    @abstractmethod
    def predict(self, **kwargs):
        """Run model inference for a prepared input batch."""
