"""Physics backends package.

This package contains implementations of different physics calculation backends.

Available backends:
- ABACUS: First-principles calculation software (primary backend)
- PySCF: Python-based quantum chemistry library (secondary backend)
"""

from .base import PhysicsBackend, SCFBackend
from .factory import get_backend, get_scf_backend

__all__ = [
    'PhysicsBackend',
    'SCFBackend',
    'get_backend',
    'get_scf_backend',
]
