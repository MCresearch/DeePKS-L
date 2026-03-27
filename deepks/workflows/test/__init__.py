"""Test workflow package.

This package implements the model evaluation workflow for DeePKS.
It keeps the legacy evaluation behavior but exposes the same
run_*_workflow entrypoint shape used by other tasks.
"""

from .workflow import run_test_workflow

__all__ = ['run_test_workflow']
