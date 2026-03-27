"""Unified input configuration module for DeePKS."""

from .config import (
    VALID_TASK_TYPES,
    apply_backend_compatibility,
    get_default_config,
    infer_scf_backend,
    normalize_config,
    render_input_parameter_doc,
)
from .loader import load_config
from .merger import merge_configs, apply_parameter_inheritance, package_config
from .validator import validate_config
from .dispatcher import dispatch_command


def build_runtime_config_from_raw(raw_config):
    """Build packaged runtime config from an in-memory raw config dictionary."""
    task_type = raw_config.get('type')
    if task_type is None:
        raise ValueError("'type' field is required in configuration file")
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Unknown type: {task_type}. Valid types: {', '.join(sorted(VALID_TASK_TYPES))}"
        )

    normalized = normalize_config(raw_config, task_type)
    scf_soft = infer_scf_backend(normalized)
    defaults = get_default_config(task_type, scf_soft)
    config = merge_configs(defaults, normalized)

    if task_type == 'iterate':
        config = apply_parameter_inheritance(config)

    config = apply_backend_compatibility(config, task_type)
    validate_config(config, task_type)
    return package_config(config)


def build_runtime_config(config_path):
    """Build packaged runtime config from a user config path."""
    raw_config = load_config(config_path)
    return build_runtime_config_from_raw(raw_config)


__all__ = [
    'VALID_TASK_TYPES',
    'load_config',
    'merge_configs',
    'apply_parameter_inheritance',
    'package_config',
    'validate_config',
    'get_default_config',
    'dispatch_command',
    'build_runtime_config',
    'build_runtime_config_from_raw',
    'normalize_config',
    'apply_backend_compatibility',
    'infer_scf_backend',
    'render_input_parameter_doc',
]
