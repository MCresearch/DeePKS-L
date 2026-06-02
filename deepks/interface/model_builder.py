"""Interface-side model construction helpers.

This module translates recipe-level model arguments into the normalized ML-
family arguments expected by ``deepks.ml.model_io``. The actual model registry
and checkpoint loading remain in the ML layer.
"""

from typing import Any, Dict, Optional

from deepks.interface.adapters import resolve_model_args
from deepks.ml.model_io import build_model as build_ml_model
from deepks.ml.model_io import load_runtime_model as load_ml_runtime_model
from deepks.ml.model_io import normalize_model_family


def build_model(
    model_family: Optional[str],
    *,
    model_args: Optional[Dict[str, Any]] = None,
    restart: Optional[str] = None,
    strict: bool = False,
):
    """Create or load a model object for the requested family."""

    family = normalize_model_family(model_family)
    model_args = resolve_model_args(family, model_args)
    return build_ml_model(family, model_args=model_args, restart=restart, strict=strict)


def load_runtime_model(
    model_file: Optional[str],
    *,
    model_family: Optional[str] = "corrnet",
    strict: bool = False,
):
    """Load a model object for runtime inference / SCF usage."""

    if model_file is None or str(model_file).upper() == "NONE":
        return None

    model = load_ml_runtime_model(model_file, model_family=model_family, strict=strict)
    if hasattr(model, "double"):
        model = model.double()
    return model
