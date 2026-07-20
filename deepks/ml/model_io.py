"""ML-side model construction and loading helpers."""

from typing import Any, Dict, Optional

from deepks.io.checkpoints import load_model_checkpoint
from deepks.ml.models.corrnet import CorrNet
from deepks.ml.models.hierarchical_regression import HierarchicalRegressionNet
from deepks.ml.models.linear import LinearModel


_MODEL_REGISTRY = {
    "corrnet": CorrNet,
    "linear": LinearModel,
    "hierarchical_regression": HierarchicalRegressionNet,
}


def normalize_model_family(model_family: Optional[str]) -> str:
    """Normalize user-facing model family names."""

    family = (model_family or "corrnet").strip().lower()
    aliases = {
        "corrnet": "corrnet",
        "corr-net": "corrnet",
        "linear": "linear",
        "hierarchical_regression": "hierarchical_regression",
        "hierarchical-regression": "hierarchical_regression",
    }
    try:
        return aliases[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported model family: {model_family}") from exc


def _infer_family_from_init_args(init_args: Dict[str, Any]) -> Optional[str]:
    """Best-effort family guess from a legacy checkpoint's ``init_args``.

    Pre-2026-05-30 checkpoints don't record ``model_family``. The three
    built-in model families have disjoint signatures, so we can recover the
    family from the init-args key set:

      - HierarchicalRegressionNet → ``n_levels`` or ``level_hidden_sizes``
      - CorrNet → ``hidden_sizes`` / ``embedding`` / ``input_partition``
        / ``layer_sizes`` (legacy alias)
      - LinearModel → anything else with just ``input_dim`` + scaling

    Returns ``None`` if no signature matches (caller falls back to its
    explicit default).
    """

    keys = set(init_args or ())
    if {"n_levels", "level_hidden_sizes", "max_depth"} & keys:
        return "hierarchical_regression"
    if {"hidden_sizes", "embedding", "input_partition", "layer_sizes"} & keys:
        return "corrnet"
    if "input_dim" in keys:
        return "linear"
    return None


def peek_model_family(model_file: Optional[str]) -> Optional[str]:
    """Return the ``model_family`` recorded inside a saved checkpoint.

    Models record their own ``MODEL_FAMILY`` class attribute in the checkpoint
    dict produced by ``save_dict``. This helper lets loaders pick the right
    model class without the caller having to thread the recipe's family
    through every layer (the ABACUS iterate ``convert_data`` path being the
    motivating case — it only has a model-file path, no recipe context).

    For legacy checkpoints that predate the ``model_family`` field, the
    family is inferred from the saved ``init_args`` signature (see
    :func:`_infer_family_from_init_args`).

    Returns ``None`` for:
      - missing / null file paths,
      - TorchScript-compiled (``.ptg``) files (which have no checkpoint dict),
      - checkpoints whose init_args don't match any known family.
    """

    if model_file is None or str(model_file).upper() == "NONE":
        return None
    try:
        checkpoint = load_model_checkpoint(model_file)
    except Exception:
        # File may be a TorchScript-traced module (e.g. model.ptg used by
        # SCF backends) which doesn't carry a checkpoint dict, or simply
        # unreadable in this context. Leave detection to the caller.
        return None
    if not isinstance(checkpoint, dict):
        return None
    family = checkpoint.get("model_family")
    if isinstance(family, str) and family:
        return family
    # Legacy fallback: infer from init_args structure.
    return _infer_family_from_init_args(checkpoint.get("init_args", {}))


def build_model(
    model_family: Optional[str],
    *,
    model_args: Optional[Dict[str, Any]] = None,
    restart: Optional[str] = None,
    strict: bool = False,
):
    """Create or load a model object for the requested family.

    When ``restart`` is given but ``model_family`` is ``None``, the
    family is auto-detected from the checkpoint's ``model_family`` field
    (see :func:`peek_model_family`). Explicit ``model_family`` always wins.
    """

    if restart is not None and model_family is None:
        detected = peek_model_family(restart)
        if detected is not None:
            model_family = detected

    family = normalize_model_family(model_family)
    model_args = dict(model_args or {})
    model_cls = _MODEL_REGISTRY[family]

    if restart is not None:
        return model_cls.load(restart, strict=strict)

    return model_cls(**model_args).double()


def load_runtime_model(
    model_file: Optional[str],
    *,
    model_family: Optional[str] = None,
    strict: bool = False,
):
    """Load a model object for runtime inference usage.

    The previous default of ``model_family="corrnet"`` silently mismatched
    when callers (the ABACUS iterate ``convert_data`` workflow in
    particular) handed in a non-CorrNet checkpoint. ``model_family`` now
    defaults to ``None`` and the family is auto-detected from the
    checkpoint's recorded ``MODEL_FAMILY`` (falling back to ``"corrnet"``
    only for legacy checkpoints that predate the field).
    """

    if model_file is None or str(model_file).upper() == "NONE":
        return None

    if model_family is None:
        model_family = peek_model_family(model_file) or "corrnet"

    model = build_model(model_family, restart=model_file, strict=strict)
    if hasattr(model, "double"):
        model = model.double()
    return model
