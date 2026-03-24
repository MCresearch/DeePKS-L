"""Configuration loader for DeePKS."""

import os
from deepks.io.utils import load_yaml


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        dict: Loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        config = load_yaml(config_path)
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(config)}")

    return config


def load_multiple_configs(config_paths):
    """Load and merge multiple configuration files.

    Later configs override earlier ones.

    Args:
        config_paths: List of configuration file paths

    Returns:
        dict: Merged configuration
    """
    from .merger import merge_configs

    configs = []
    for path in config_paths:
        config = load_config(path)
        configs.append(config)

    if not configs:
        return {}

    merged = configs[0]
    for config in configs[1:]:
        merged = merge_configs(merged, config)

    return merged
