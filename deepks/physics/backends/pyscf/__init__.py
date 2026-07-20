"""PySCF backend package.

Keep package import lightweight so environments without PySCF can still import
unrelated helpers such as basis metadata or interface adapters.
"""

from .backend import PySCFBackend

__all__ = ["PySCFBackend"]
