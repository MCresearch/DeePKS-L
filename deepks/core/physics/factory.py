"""Physics backend factory for selecting SCF implementations."""

from deepks.io.adapters.physics_backend import PySCFPhysicsBackend, ABACUSPhysicsBackend


def get_scf_backend(scf_soft):
    """Get SCF backend instance based on software choice.

    Args:
        scf_soft: SCF software name ('pyscf' or 'abacus')

    Returns:
        PhysicsBackend instance

    Raises:
        ValueError: If scf_soft is not recognized
    """
    backends = {
        'pyscf': PySCFPhysicsBackend,
        'abacus': ABACUSPhysicsBackend,
    }

    scf_soft_lower = scf_soft.lower()
    if scf_soft_lower not in backends:
        raise ValueError(
            f"Unknown SCF backend: {scf_soft}. "
            f"Available backends: {list(backends.keys())}"
        )

    return backends[scf_soft_lower]()

