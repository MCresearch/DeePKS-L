"""Output reducers for the interface layer.

A reducer turns a model's raw per-element output (per-atom, per-node,
per-edge ...) into the shape consumed downstream by the property scheme
or supervision term. Reduction is a **physics-task** decision (e.g.
"energy is extensive in atoms, so sum over atoms"), not a network-
architecture decision. Keeping it here — rather than inside the model
``forward`` — is what lets the ML layer stay physics-agnostic: a
network is just a function from inputs to outputs of whatever shape it
naturally produces.

The minimal contract is a callable taking a torch tensor and returning
a torch tensor. ``model_meta`` is optional and lets a reducer pull
calibration values (e.g. an additive ``output_bias``) directly off the
model object when one exists.

Reducers MUST be torch-differentiable so the objective adapter can keep
autograd alive through them and recover ``∂E/∂input`` style chain-rule
derivatives.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


class OutputReducer:
    """Abstract base for output reducers."""

    name: str = "identity"

    def __call__(self, tensor: torch.Tensor, *, model_meta: Optional[Any] = None) -> torch.Tensor:
        return self.reduce(tensor, model_meta=model_meta)

    def reduce(self, tensor: torch.Tensor, *, model_meta: Optional[Any] = None) -> torch.Tensor:
        raise NotImplementedError


class Identity(OutputReducer):
    """Pass-through: the model output is already in the supervision shape."""

    name = "identity"

    def reduce(self, tensor, *, model_meta=None):
        return tensor


class SumOverAtoms(OutputReducer):
    """Sum over the per-atom axis (``dim=-2``).

    Optionally adds an ``output_bias`` (read from ``model_meta.output_bias``
    when available) — this is the standard "extensive scalar" reduction used
    by descriptor-energy models such as CorrNet / LinearModel /
    HierarchicalRegressionNet. The bias lives on the model as a calibration
    parameter; the reducer just reads and applies it so the model body itself
    stays free of the physical-extensivity assumption.
    """

    name = "sum_over_atoms"

    def reduce(self, tensor, *, model_meta=None):
        reduced = tensor.sum(-2)
        bias = _read_attribute(model_meta, "output_bias")
        if bias is not None:
            reduced = reduced + bias
        return reduced


class MeanOverAtoms(OutputReducer):
    """Mean over the per-atom axis (``dim=-2``)."""

    name = "mean_over_atoms"

    def reduce(self, tensor, *, model_meta=None):
        return tensor.mean(-2)


_REDUCER_REGISTRY = {
    Identity.name: Identity,
    SumOverAtoms.name: SumOverAtoms,
    "sum-over-atoms": SumOverAtoms,
    MeanOverAtoms.name: MeanOverAtoms,
    "mean-over-atoms": MeanOverAtoms,
}


def build_reducer(spec) -> OutputReducer:
    """Construct an ``OutputReducer`` from a string name or instance."""

    if isinstance(spec, OutputReducer):
        return spec
    if spec is None:
        return Identity()
    if isinstance(spec, type) and issubclass(spec, OutputReducer):
        return spec()
    if isinstance(spec, str):
        try:
            cls = _REDUCER_REGISTRY[spec.strip().lower()]
        except KeyError as exc:
            raise KeyError(f"Unknown output reducer: {spec!r}") from exc
        return cls()
    raise TypeError(f"Unsupported reducer spec: {type(spec)!r}")


def _read_attribute(obj, name):
    if obj is None:
        return None
    return getattr(obj, name, None)


__all__ = [
    "OutputReducer",
    "Identity",
    "SumOverAtoms",
    "MeanOverAtoms",
    "build_reducer",
]
