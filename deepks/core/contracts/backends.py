"""Abstract backend contracts used by orchestration and pipelines.

These interfaces provide a stable boundary between orchestration logic and
concrete ML/physics implementations during incremental refactoring.
"""

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


class PhysicsBackend(ABC):
    """Contract for physics/SCF backends."""

    @abstractmethod
    def run_scf(self, **kwargs):
        """Run SCF calculation and return backend-specific outputs."""

    @abstractmethod
    def collect_stats(self, **kwargs):
        """Collect summary statistics from generated SCF results."""
