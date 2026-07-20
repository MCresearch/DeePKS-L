"""SCF workflow package.

This package implements the SCF (Self-Consistent Field) workflow,
which is backend-agnostic and follows the three-stage pattern:
1. Prepare: Create directories and input files
2. Execute: Run calculations via scheduler
3. Collect: Parse and aggregate results
"""

from .workflow import run_scf_workflow

__all__ = ['run_scf_workflow']
