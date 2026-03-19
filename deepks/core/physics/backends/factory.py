"""Physics backend factory.

This module provides factory functions to create backend instances.
"""

from typing import Optional, Dict, Any
from .base import PhysicsBackend, SCFBackend


def get_backend(backend_name: str, config: Optional[Dict[str, Any]] = None) -> PhysicsBackend:
    """Get physics backend instance by name.

    Args:
        backend_name: Backend name ('pyscf', 'abacus')
        config: Backend-specific configuration

    Returns:
        PhysicsBackend: Backend instance

    Raises:
        ValueError: If backend name is not recognized
    """
    backend_name = backend_name.lower()

    if backend_name == 'pyscf':
        from .pyscf.backend import PySCFBackend
        return PySCFBackend(config)
    elif backend_name == 'abacus':
        from .abacus.backend import AbacusBackend
        return AbacusBackend(config)
    else:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available backends: pyscf, abacus"
        )


def get_scf_backend(backend_name: str, config: Optional[Dict[str, Any]] = None) -> SCFBackend:
    """Get SCF backend instance by name.

    Args:
        backend_name: Backend name ('pyscf', 'abacus')
        config: Backend-specific configuration

    Returns:
        SCFBackend: SCF backend instance

    Raises:
        ValueError: If backend name is not recognized or not an SCF backend
    """
    backend = get_backend(backend_name, config)

    if not isinstance(backend, SCFBackend):
        raise ValueError(f"Backend {backend_name} is not an SCF backend")

    return backend


# Backward compatibility with old code
def get_physics_backend(scf_soft: str = 'pyscf') -> SCFBackend:
    """Get physics backend (backward compatibility).

    This function maintains compatibility with the old CLI code.

    Args:
        scf_soft: SCF software name ('pyscf' or 'abacus')

    Returns:
        SCFBackend: Backend instance
    """
    return get_scf_backend(scf_soft)
