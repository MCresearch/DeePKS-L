"""PySCF backend package.

This package implements the PySCF backend for physics calculations.
Note: PySCF is not available in test_env, so tests are skipped.
"""

from .backend import PySCFBackend

__all__ = ['PySCFBackend']
