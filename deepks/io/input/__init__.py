"""Unified input configuration module for DeePKS."""

from .config import normalize_config
from .docs import render_input_parameter_doc
from .defaults import get_default_config
from .loader import load_config
from .merger import merge_configs
from .packager import (
    INTERNAL_PACKED_MARKER,
    get_payload_key,
    is_packed_config,
    package_config,
)
from .validator import validate_config
from .dispatcher import dispatch_command
from .validator import VALID_TASK_TYPES


def load_runtime_config(config_path):
    """Load and package the single runtime config used by the CLI."""
    raw_config = load_config(config_path)
    if is_packed_config(raw_config):
        task_type = raw_config.get("type")
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Unknown packed type: {task_type}. Valid types: {', '.join(sorted(VALID_TASK_TYPES))}"
            )
        payload_key = get_payload_key(task_type)
        if not isinstance(raw_config.get(payload_key), dict):
            raise ValueError(f"Packed config missing '{payload_key}'")
        return raw_config

    task_type = raw_config.get('type')
    if task_type is None:
        raise ValueError("'type' field is required in configuration file")
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Unknown type: {task_type}. Valid types: {', '.join(sorted(VALID_TASK_TYPES))}"
        )

    normalized = normalize_config(raw_config)
    validate_config(normalized, task_type)
    physics = normalized.get("physics") if isinstance(normalized.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    scf_soft = backend.get("name")
    defaults = get_default_config(task_type, scf_soft)
    config = merge_configs(defaults, normalized)
    return package_config(config)


__all__ = [
    'VALID_TASK_TYPES',
    'load_config',
    'merge_configs',
    'package_config',
    'validate_config',
    'get_default_config',
    'dispatch_command',
    'load_runtime_config',
    'normalize_config',
    'render_input_parameter_doc',
    'INTERNAL_PACKED_MARKER',
]
