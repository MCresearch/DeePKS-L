"""Model-config normalization helpers for interface recipes."""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from deepks.ml.model_io import normalize_model_family


def resolve_corrnet_model_args(model_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize CorrNet model args at the interface boundary.

    The main purpose of this adapter is to resolve physics-specific basis inputs
    into the lighter-weight shell grouping expected by the model embedding path.
    """

    normalized = dict(model_args or {})
    for unused_key in (
        "max_depth",
        "max_output_dim",
        "trunk_hidden_sizes",
        "head_hidden_sizes",
        "level_output_shapes",
        "shared_trunk",
    ):
        normalized.pop(unused_key, None)
    if normalized.get("input_partition") is not None:
        normalized.pop("shell_sec", None)
        return normalized
    if normalized.get("shell_sec") is not None:
        normalized["input_partition"] = list(normalized.pop("shell_sec"))
        return normalized

    proj_basis = normalized.pop("proj_basis", None)
    embedding = normalized.get("embedding")
    if proj_basis is None or embedding is None:
        return normalized

    from deepks.physics.backends.pyscf.basis import get_shell_sec, load_basis

    proj_basis = load_basis(proj_basis)
    normalized["input_partition"] = get_shell_sec(proj_basis)
    return normalized


def resolve_model_args(
    model_family: Optional[str],
    model_args: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize model args at the interface boundary.

    `ml` should only receive finalized model-layout arguments. Family-specific
    compatibility and physics-to-layout translation stay in `interface`.
    """

    family = normalize_model_family(model_family)
    normalized = dict(model_args or {})
    if family == "corrnet":
        return resolve_corrnet_model_args(normalized)
    if family == "hierarchical_regression":
        return resolve_hierarchical_regression_model_args(normalized)
    return normalized


def resolve_hierarchical_regression_model_args(
    model_args: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize hierarchical-regression model args for the additive-stack design.

    The model is an additive stack of independent per-level MLPs that all share
    the same descriptor input and produce a single scalar primary output. The
    only architecture knobs the model needs are ``input_dim``, ``n_levels``, and
    either a shared ``hidden_sizes`` MLP topology or per-level overrides via
    ``level_hidden_sizes``.

    The "levels" field is treated as metadata for the workflow / recipe (each
    entry may carry per-stage data hints such as ``target_shape``), and is not
    forwarded to the model itself. When ``n_levels`` is not given explicitly,
    it is inferred from ``len(levels)``.
    """

    normalized = dict(model_args or {})

    # Legacy alias: previous design used trunk_hidden_sizes + head_hidden_sizes.
    # Concatenate them into a single MLP topology used by every level.
    trunk_hidden_sizes = normalized.pop("trunk_hidden_sizes", None)
    head_hidden_sizes = normalized.pop("head_hidden_sizes", None)
    if normalized.get("hidden_sizes") is None and (trunk_hidden_sizes or head_hidden_sizes):
        merged = list(trunk_hidden_sizes or []) + list(head_hidden_sizes or [])
        if merged:
            normalized["hidden_sizes"] = merged

    if normalized.get("n_levels") is None:
        # Prefer explicit max_depth (legacy alias), then derive from `levels` metadata.
        max_depth = normalized.get("max_depth")
        if max_depth is not None:
            normalized["n_levels"] = int(max_depth)
        else:
            levels = normalized.get("levels")
            if isinstance(levels, list) and levels:
                normalized["n_levels"] = len(levels)

    # Strip fields that exist only for workflow / data orchestration, not
    # for the model body.
    for unused_key in (
        "embedding",
        "proj_basis",
        "shell_sec",
        "input_partition",
        "elem_table",
        "use_resnet",
        "layer_norm",
        "levels",
        "output_heads",
        "primary_output_name",
        "max_depth",
        "max_output_dim",
        "level_output_shapes",
        "shared_trunk",
    ):
        normalized.pop(unused_key, None)

    return normalized


def resolve_hierarchical_model_levels(config_or_model_args: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return normalized hierarchical level metadata from model args.

    Canonical source:
    - ``ml.model.args.levels`` in full task configs
    - ``model_args["levels"]`` in already-extracted model args
    """

    source = dict(config_or_model_args or {})
    if isinstance(source.get("ml"), dict):
        model_cfg = source["ml"].get("model") if isinstance(source["ml"].get("model"), dict) else {}
        model_args = model_cfg.get("args") if isinstance(model_cfg.get("args"), dict) else {}
    else:
        model_args = source
    raw_levels = model_args.get("levels") if isinstance(model_args.get("levels"), list) else []
    levels: List[Dict[str, Any]] = []
    for index, raw_level in enumerate(raw_levels):
        if not isinstance(raw_level, dict):
            raise TypeError(f"ml.model.args.levels[{index}] must be a dict, got {type(raw_level)!r}")
        level = deepcopy(raw_level)
        level.setdefault("level", index)
        if "name" not in level:
            level["name"] = f"level_{index}"
        levels.append(level)
    return levels
