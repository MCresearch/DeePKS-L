"""Adapters between DeePKS data structures and the unified interface layer."""

from .sample import sample_to_task_batch, task_batch_to_sample
from .representation import TaskBatchRepresentationBuilder
from .model_config import (
    resolve_corrnet_model_args,
    resolve_hierarchical_model_levels,
    resolve_model_args,
)
from .preprocess import fit_elem_const, preprocess

__all__ = [
    "sample_to_task_batch",
    "task_batch_to_sample",
    "TaskBatchRepresentationBuilder",
    "resolve_corrnet_model_args",
    "resolve_hierarchical_model_levels",
    "resolve_model_args",
    "fit_elem_const",
    "preprocess",
]
