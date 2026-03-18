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
        """Run SCF calculation and return backend-specific outputs.

        Args:
            systems: List of system paths
            model_file: Path to model file
            proj_basis: Projection basis
            dump_dir: Output directory
            dump_fields: Fields to output
            device: Computation device
            verbose: Verbosity level
            **backend_args: Backend-specific arguments

        Returns:
            None (results written to dump_dir)
        """

    @abstractmethod
    def collect_stats(self, **kwargs):
        """Collect summary statistics from generated SCF results.

        Args:
            systems: List of system paths (optional)
            dump_dir: Directory containing SCF results (optional)
            **kwargs: Additional statistics options

        Returns:
            Statistics dictionary or prints to stdout
        """

    @abstractmethod
    def validate_args(self, **kwargs):
        """Validate backend-specific arguments.

        Args:
            **kwargs: Backend-specific arguments to validate

        Raises:
            ValueError: If required arguments are missing or invalid
            TypeError: If arguments have incorrect types
        """
