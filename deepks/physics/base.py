"""Abstract physics-side contracts for scheme execution and backend execution."""

from abc import ABC, abstractmethod


class RepresentationBuilder(ABC):
    """Build model-facing batches from raw physical data."""

    @abstractmethod
    def build_batch(self, raw_batch, runtime_config=None):
        """Return a structured batch with model inputs, targets, and context."""


class PropertyScheme(ABC):
    """A concrete model/output scheme for recovering physical quantities."""

    @abstractmethod
    def supported_properties(self):
        """Return the property names supported by this scheme."""

    @abstractmethod
    def required_model_outputs(self, requested_properties):
        """Describe which model-output keys are required."""

    @abstractmethod
    def required_model_derivatives(self, requested_properties):
        """Describe which model-input derivatives are required."""

    @abstractmethod
    def validate_context(self, requested_properties, context):
        """Validate that context contains what the requested properties need."""

    @abstractmethod
    def compute_property(self, name, *, model_outputs, model_derivatives, context, cache):
        """Compute one property under the current scheme."""


class BackendRunner(ABC):
    """Physical backend execution contract."""

    @property
    @abstractmethod
    def name(self):
        """Backend name."""

    @abstractmethod
    def prepare(self, systems, config, workdir):
        """Prepare backend inputs under workdir."""

    @abstractmethod
    def run(self, prepared, runtime_config):
        """Run backend execution."""

    @abstractmethod
    def collect(self, prepared, runtime_config):
        """Collect backend outputs."""
