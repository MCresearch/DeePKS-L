"""Additive stacked residual energy regressor.

Each model is an additive stack of independent per-level sub-networks:

    E(d) = sum_l level_net_l(d)

where every ``level_net_l`` is an independent MLP producing a per-atom
scalar contribution; the per-atom contributions are summed to obtain a
per-system energy of shape ``(batch, 1)``.

Staged training freezes lower levels and trains the current one, so
adding higher levels yields a "correction" on top of the locked-in
contribution of the lower levels. This matches the design where
higher-order LCAO basis data is supervised as a residual on top of the
contribution already captured by the lower-order basis data.

The network deliberately exposes a single scalar primary output so it can
participate in the existing ``EnergyDescriptorScheme`` chain-rule
recoveries (force, V_delta, V_delta(R) and so on) without any new physics
layer code.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from deepks.io.checkpoints import load_compiled_or_checkpoint, save_model_checkpoint
from deepks.ml.base import ModelAdapter
from deepks.ml.models.corrnet import mygelu

SCALE_EPS = 1e-8


class _MyGELU(nn.Module):
    def forward(self, x):
        return mygelu(x)


def _make_activation(actv_fn):
    if callable(actv_fn) and not isinstance(actv_fn, str):
        class _CallableActivation(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        return _CallableActivation(actv_fn)

    code = str(actv_fn).strip().lower()
    if code == "relu":
        return nn.ReLU()
    if code == "gelu":
        return nn.GELU()
    if code == "mygelu":
        return _MyGELU()
    if code == "tanh":
        return nn.Tanh()
    if code == "sigmoid":
        return nn.Sigmoid()
    if code == "softplus":
        return nn.Softplus()
    if code == "silu":
        return nn.SiLU()
    raise ValueError(f"{actv_fn} is not a valid activation function")


def _make_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, actv_fn="gelu") -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(in_dim, hidden_dim).double())
        layers.append(_make_activation(actv_fn))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim).double())
    return nn.Sequential(*layers)


def _coerce_int_sequence(seq):
    return [int(v) for v in seq]


def _normalize_level_hidden_sizes(
    level_hidden_sizes,
    *,
    n_levels: int,
    default_hidden_sizes: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """Return one hidden-size list per level."""

    if level_hidden_sizes is None:
        if default_hidden_sizes is None:
            default_hidden_sizes = (100, 100, 100)
        return [_coerce_int_sequence(default_hidden_sizes) for _ in range(n_levels)]
    if not isinstance(level_hidden_sizes, (list, tuple)):
        raise TypeError(
            f"level_hidden_sizes must be a list, got {type(level_hidden_sizes)!r}"
        )
    if level_hidden_sizes and all(isinstance(v, (int, float)) for v in level_hidden_sizes):
        return [_coerce_int_sequence(level_hidden_sizes) for _ in range(n_levels)]
    if len(level_hidden_sizes) != n_levels:
        raise ValueError(
            f"level_hidden_sizes expected {n_levels} entries, got {len(level_hidden_sizes)}"
        )
    return [_coerce_int_sequence(entry) for entry in level_hidden_sizes]


class HierarchicalRegressionNet(nn.Module, ModelAdapter):
    """Additive stacked-MLP regressor producing a single scalar primary output."""

    MODEL_FAMILY = "hierarchical_regression"

    def __init__(
        self,
        input_dim,
        n_levels=1,
        level_hidden_sizes=None,
        hidden_sizes=None,
        actv_fn="gelu",
        input_shift=0,
        input_scale=1,
        output_scale=1,
    ):
        super().__init__()
        n_levels = int(n_levels)
        if n_levels <= 0:
            raise ValueError("HierarchicalRegressionNet requires n_levels > 0")
        self.input_dim = int(input_dim)
        self.n_levels = n_levels
        self.actv_fn = actv_fn

        resolved_levels = _normalize_level_hidden_sizes(
            level_hidden_sizes,
            n_levels=n_levels,
            default_hidden_sizes=hidden_sizes,
        )

        self.level_hidden_sizes = [list(level) for level in resolved_levels]

        self._init_args = {
            "input_dim": self.input_dim,
            "n_levels": n_levels,
            "level_hidden_sizes": deepcopy(self.level_hidden_sizes),
            "actv_fn": actv_fn,
            "input_shift": input_shift,
            "input_scale": input_scale,
            "output_scale": output_scale,
        }

        self.level_nets = nn.ModuleList(
            [
                _make_mlp(self.input_dim, level_hidden, 1, actv_fn=self.actv_fn)
                for level_hidden in self.level_hidden_sizes
            ]
        )

        self.input_shift = nn.Parameter(
            torch.tensor(input_shift, dtype=torch.float64).expand(self.input_dim).clone(),
            requires_grad=False,
        )
        self.input_scale = nn.Parameter(
            torch.tensor(input_scale, dtype=torch.float64).expand(self.input_dim).clone(),
            requires_grad=False,
        )
        self.output_scale = nn.Parameter(
            torch.tensor(output_scale, dtype=torch.float64),
            requires_grad=False,
        )
        self.output_bias = nn.Parameter(
            torch.tensor(0, dtype=torch.float64),
            requires_grad=False,
        )

    def _normalize_input(self, x):
        return (x - self.input_shift) / (self.input_scale + SCALE_EPS)

    def forward(self, model_inputs):
        """Return the unreduced per-atom additive-stack contribution."""

        from deepks.ml.models.corrnet import _extract_descriptor_tensor

        x = _extract_descriptor_tensor(model_inputs)
        x_norm = self._normalize_input(x)
        per_atom = self.level_nets[0](x_norm)
        for net in self.level_nets[1:]:
            per_atom = per_atom + net(x_norm)
        return per_atom / self.output_scale

    def configure_stage_trainability(
        self,
        active_level: int,
        *,
        freeze_lower: bool = True,
        freeze_trunk: bool = True,  # accepted for backward compatibility; ignored in the new design
    ):
        """Freeze lower-level subnetworks and unfreeze the active one.

        The ``freeze_trunk`` flag is preserved in the signature for callers that
        still pass it from older configuration files; it has no effect in the
        new additive-stacking design (there is no shared trunk).
        """

        del freeze_trunk  # explicit no-op for compatibility
        if not 0 <= active_level < self.n_levels:
            raise IndexError(f"Active hierarchy level out of range: {active_level}")
        for index, net in enumerate(self.level_nets):
            if index == active_level:
                net.requires_grad_(True)
            elif freeze_lower and index < active_level:
                net.requires_grad_(False)
            elif index > active_level:
                # higher levels stay inert until their stage runs
                net.requires_grad_(False)
            else:
                net.requires_grad_(True)

    def set_normalization(self, shift=None, scale=None):
        dtype = self.input_scale.dtype
        if shift is not None:
            self.input_shift.data[:] = torch.tensor(shift, dtype=dtype)
        if scale is not None:
            self.input_scale.data[:] = torch.tensor(scale, dtype=dtype)

    def set_output_bias(self, const):
        dtype = self.output_bias.dtype
        self.output_bias.data = torch.tensor(const, dtype=dtype).reshape([])

    def save_dict(self, **extra_info):
        return {
            "model_family": self.MODEL_FAMILY,
            "state_dict": self.state_dict(),
            "init_args": self._init_args,
            "extra_info": extra_info,
        }

    def save(self, filename, **extra_info):
        save_model_checkpoint(filename, self.save_dict(**extra_info))

    def compile(self, set_eval=True, **kwargs):
        old_mode = self.training
        if set_eval:
            self.eval()
        # See CorrNet.compile docstring: tracing a nn.Module wrapper rather
        # than a closure prevents the "requires_grad as constant" trace error
        # that would otherwise hit because ``configure_stage_trainability``
        # intentionally leaves the active level's parameters trainable.
        from deepks.ml.models.corrnet import _DescriptorEnergyTraceWrapper

        wrapper = _DescriptorEnergyTraceWrapper(self).eval()
        smodel = torch.jit.trace(
            wrapper,
            torch.empty((2, 2, self.input_dim), dtype=torch.float64),
            **kwargs,
        )
        self.train(old_mode)
        return smodel

    def compile_save(self, filename, **kwargs):
        torch.jit.save(self.compile(**kwargs), filename)

    @staticmethod
    def load_dict(checkpoint, strict=False):
        init_args = dict(checkpoint["init_args"])
        # Legacy compatibility for checkpoints produced by the per-level
        # Hamiltonian-tensor design that lived in this file previously.
        legacy_aliases = (
            "max_depth",
            "max_output_dim",
            "trunk_hidden_sizes",
            "head_hidden_sizes",
            "level_output_shapes",
            "shared_trunk",
        )
        legacy_present = {key: init_args.pop(key) for key in legacy_aliases if key in init_args}
        if "n_levels" not in init_args and "max_depth" in legacy_present:
            init_args["n_levels"] = int(legacy_present["max_depth"])
        if "level_hidden_sizes" not in init_args:
            head_hidden = legacy_present.get("head_hidden_sizes")
            trunk_hidden = legacy_present.get("trunk_hidden_sizes")
            n_levels = int(init_args.get("n_levels", 1))
            if head_hidden is not None or trunk_hidden is not None:
                merged = list(trunk_hidden or []) + list(head_hidden or [])
                init_args["level_hidden_sizes"] = [list(merged) for _ in range(n_levels)]
        model = HierarchicalRegressionNet(**init_args)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model

    @staticmethod
    def load(filename, strict=False):
        checkpoint = load_compiled_or_checkpoint(filename)
        if not isinstance(checkpoint, dict):
            return checkpoint
        return HierarchicalRegressionNet.load_dict(checkpoint, strict=strict)
