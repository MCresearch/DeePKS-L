"""Abstract ML-side contracts for model and batch handling.

The ML layer holds only physics-agnostic abstractions. Task-semantic
abstractions such as ``ObjectiveAdapter`` live in
``deepks.interface.base`` because they bridge model outputs with
physics-side property recovery, which requires importing both layers.

Model contract (R2 — dict-in / dict-out)
----------------------------------------

Trainable models talk to the rest of the framework through two methods
on :class:`ModelAdapter`:

``forward(model_inputs)``
    Accepts a ``Dict[str, Tensor | Any]`` of named model inputs and
    returns a ``Dict[str, Tensor]`` of named raw model outputs. The
    framework never assumes a specific input/output key set; it discovers
    them from the recipe's input-field mapping and the property scheme.

    For backward compatibility, concrete model classes MAY also accept a
    bare tensor positionally — typically those that are TorchScript-traced
    into SCF backends (where the SCF caller passes a single descriptor
    tensor). In that case ``forward(tensor)`` returns ``tensor``.

``forward_with_derivatives(model_inputs, derivative_spec)``
    The training-time entry point. ``model_inputs`` is always a dict.
    ``derivative_spec`` is a ``Dict[str, bool]`` keyed by input names —
    ``True`` requests gradient of the primary output with respect to
    that input. Returns ``(outputs, derivatives)`` where ``outputs`` is
    the same dict shape as ``forward`` and ``derivatives`` is a dict
    keyed by input name with tensor values (or ``None`` when not
    requested).

    Models with a single canonical input may treat both
    ``{"<canonical>": tensor}`` and the legacy ``{"input": tensor}`` as
    equivalent for compatibility.

Networks MUST NOT bake in physically-meaningful reductions
(summation over atoms, mean-over-nodes, etc.). Output reduction is the
responsibility of the interface layer (see ``deepks.interface.reducers``)
because the choice of reduction is a physical-task decision, not a
network-architecture decision.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable


class ModelAdapter(ABC):
    """Abstract trainable model contract consumed by the ML loop.

    Only ``forward`` is mandatory. Autograd of the (post-reduction)
    primary output w.r.t. selected named inputs is the objective
    adapter's responsibility (it calls ``forward`` and runs
    ``torch.autograd.grad`` itself), so models are not required to
    implement ``forward_with_derivatives``. Models MAY override it as
    an opt-in fast path; the framework only uses it when present.
    """

    @abstractmethod
    def forward(self, model_inputs):
        """Return raw model outputs.

        ``model_inputs`` is typically a ``Dict[str, Tensor | Any]`` of
        named inputs, though concrete models may also accept a bare
        tensor positionally for TorchScript / SCF compatibility.

        Returns either a tensor (single-output convenience) or a
        ``Dict[str, Tensor]`` of named raw model outputs.
        """

    @abstractmethod
    def parameters(self):
        """Return iterable parameters for optimizer construction."""

    @abstractmethod
    def state_dict(self):
        """Return serializable state."""

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load serialized state."""


@runtime_checkable
class BatchProtocol(Protocol):
    """Structured batch exchanged between interface and ML loop."""

    model_inputs: Dict[str, Any]
    targets: Dict[str, Any]
    context: Dict[str, Any]
    metadata: Dict[str, Any]
