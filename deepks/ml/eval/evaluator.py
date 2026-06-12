"""Reusable ML-side evaluator."""

import numpy as np


class Evaluator:
    """Generic evaluator for model/objective pairs."""

    def __init__(self, *, batch_adapter=None):
        self.batch_adapter = batch_adapter

    def _iter_batches(self, reader):
        if self.batch_adapter is not None:
            if hasattr(reader, "sample_all_batch"):
                for batch in reader.sample_all_batch():
                    yield self.batch_adapter(batch)
                return
            for batch in reader:
                yield self.batch_adapter(batch)
            return
        if hasattr(reader, "sample_all_task_batches"):
            yield from reader.sample_all_task_batches()
            return
        if hasattr(reader, "sample_all_batch"):
            yield from reader.sample_all_batch()
            return
        yield from reader

    def evaluate(self, model, reader, objective):
        """Return averaged per-batch metrics/losses."""

        model.eval()
        losses = []
        for batch in self._iter_batches(reader):
            metrics = objective.compute_metrics(model, batch)
            losses.append([metric.item() for metric in metrics])
        if not losses:
            return np.array([])
        return np.mean(losses, axis=0)
