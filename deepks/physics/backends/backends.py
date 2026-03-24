"""Compatibility layer for backend imports.

This module re-exports backend functionality for backward compatibility.
"""

from .base import PhysicsBackend, SCFBackend
from .factory import get_backend, get_scf_backend, get_physics_backend

__all__ = [
    'PhysicsBackend',
    'SCFBackend',
    'get_backend',
    'get_scf_backend',
    'get_physics_backend',
]
