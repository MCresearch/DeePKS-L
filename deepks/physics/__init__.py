"""Physics calculation layer for DeePKS.

The main public structure is:

- `properties/`: one file per physical quantity
- `engine.py`: property orchestration facade
- `backends/`: concrete SCF/backend execution
"""

from .backends import PhysicsBackend, SCFBackend, get_backend, get_scf_backend

__all__ = [
    'PhysicsBackend',
    'SCFBackend',
    'get_backend',
    'get_scf_backend',
]
