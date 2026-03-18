"""Unified input configuration module for DeePKS."""

from .loader import load_config
from .merger import merge_configs
from .validator import validate_config
from .defaults import get_default_config
from .dispatcher import dispatch_command

__all__ = [
    'load_config',
    'merge_configs',
    'validate_config',
    'get_default_config',
    'dispatch_command',
]
