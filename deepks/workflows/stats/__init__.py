"""Stats workflow package.

This package implements the SCF statistics workflow for DeePKS.
It exposes the same run_*_workflow entrypoint shape used by other tasks.
"""

from .workflow import run_stats_workflow

__all__ = ['run_stats_workflow']
