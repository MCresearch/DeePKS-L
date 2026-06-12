"""Unified batch container exchanged across layers.

``TaskBatch`` lives in the io layer because it is a generic data
container (a structured ``model_inputs`` / ``targets`` / ``context``
record). The interface, ml, and physics layers all consume it but none
of them own it — keeping it in io preserves the
``io → ml/physics → interface`` dependency direction.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch

from deepks.ml.base import BatchProtocol


@dataclass
class TaskBatch(BatchProtocol):
    """Structured batch exchanged between interface, physics, and ML layers."""

    model_inputs: Dict[str, Any]
    targets: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def field_names(self) -> Tuple[str, ...]:
        return tuple(self.model_inputs) + tuple(self.targets) + tuple(self.context)

    def display_keys(self) -> Tuple[str, ...]:
        display = self.meta.get("display_keys")
        if display is not None:
            return tuple(display)
        return self.field_names()

    @property
    def metadata(self):
        return self.meta

    def to_device(self, device: str, *, complex_cpu_context_keys=()):
        def _move(value, *, keep_complex_cpu=False):
            if isinstance(value, list):
                return [_move(v, keep_complex_cpu=keep_complex_cpu) for v in value]
            if isinstance(value, dict):
                return {
                    key: _move(item, keep_complex_cpu=keep_complex_cpu)
                    for key, item in value.items()
                }
            if not torch.is_tensor(value):
                return value
            if torch.is_complex(value):
                if keep_complex_cpu:
                    return value.to("cpu", dtype=torch.complex128, non_blocking=True)
                return value.to(device, dtype=torch.complex128, non_blocking=True)
            return value.to(device, non_blocking=True)

        model_inputs = {key: _move(value) for key, value in self.model_inputs.items()}
        targets = {key: _move(value) for key, value in self.targets.items()}
        context = {
            key: _move(value, keep_complex_cpu=(key in complex_cpu_context_keys))
            for key, value in self.context.items()
        }
        return TaskBatch(
            model_inputs=model_inputs,
            targets=targets,
            context=context,
            meta=self.meta,
        )


__all__ = ["TaskBatch"]
