"""Compatibility re-export for TaskBatch adapters now owned by io."""

from deepks.io.task_batches import sample_to_task_batch, task_batch_to_sample

__all__ = ["sample_to_task_batch", "task_batch_to_sample"]
