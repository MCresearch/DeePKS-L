"""SCF workflow - main orchestration.

This module implements the SCF (Self-Consistent Field) workflow,
which is backend-agnostic and follows the three-stage pattern.
"""

from .prepare import prepare_scf_tasks
from .execute import execute_scf_tasks
from .collect import collect_scf_results


def run_scf_workflow(config):
    """Run SCF workflow.

    This is the main entry point for SCF calculations. It orchestrates
    the three stages: prepare, execute, and collect.

    Physical Process:
    1. Prepare: Create working directories and generate input files
    2. Execute: Run SCF calculations via scheduler
    3. Collect: Parse results and aggregate data

    Args:
        config: Configuration dictionary containing:
            - type: 'scf'
            - scf_soft: Backend software ('pyscf' or 'abacus')
            - systems: List of system paths
            - dump_dir: Output directory for results
            - dump_fields: Fields to dump (e.g., ['e_tot', 'dm_eig'])
            - Backend-specific parameters (scf_abacus, mol_args, etc.)

    Returns:
        dict: Results dictionary with statistics and paths
    """
    # Stage 1: Prepare - Create directories and input files
    tasks = prepare_scf_tasks(config)

    # Stage 2: Execute - Run calculations
    execute_scf_tasks(tasks, config)

    # Stage 3: Collect - Parse and aggregate results
    results = collect_scf_results(tasks, config)

    return results
