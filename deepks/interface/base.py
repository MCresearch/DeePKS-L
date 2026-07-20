"""Interface-layer abstract contracts.

The interface layer owns task-semantic abstractions (objectives, recipe
hooks) because it is the only layer permitted to import both ``ml`` and
``physics``. Network and physics abstractions remain in their own layers
(``deepks.ml.base`` and ``deepks.physics.base`` respectively).
"""

from abc import ABC, abstractmethod


class ObjectiveAdapter(ABC):
    """Abstract objective contract consumed by the ML train/eval engines."""

    @abstractmethod
    def compute_losses(self, model, batch):
        """Return ordered loss terms with the last term being total loss."""

    @abstractmethod
    def compute_metrics(self, model, batch):
        """Return evaluation metrics for a batch."""

    @abstractmethod
    def print_head(self, name, data_keys, align_len=20):
        """Print aligned column headers for loss terms."""
