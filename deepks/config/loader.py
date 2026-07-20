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
