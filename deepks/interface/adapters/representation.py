"""Representation builders owned by the interface layer."""

from typing import Any, Dict, Mapping, Optional

from deepks.interface.batch import TaskBatch
from deepks.physics.base import RepresentationBuilder

from .sample import sample_to_task_batch


class TaskBatchRepresentationBuilder(RepresentationBuilder):
    """Build the canonical TaskBatch from current reader/sample payloads.

    R3: ``input_field_mapping`` is the recipe-declared map from reader
    sample keys to TaskBatch ``model_inputs`` keys; passed through to
    :func:`sample_to_task_batch` so multi-input model recipes (e.g. a
    GNN consuming nodes + edges + coords) route the right fields into
    the right model-input slots without modifying the adapter source.
    """

    def __init__(self, *, input_field_mapping: Optional[Mapping[str, str]] = None):
        self.input_field_mapping = dict(input_field_mapping) if input_field_mapping else None

    def build_batch(self, raw_batch: Dict[str, Any], runtime_config=None):
        if isinstance(raw_batch, TaskBatch):
            return raw_batch
        return sample_to_task_batch(raw_batch, input_field_mapping=self.input_field_mapping)
