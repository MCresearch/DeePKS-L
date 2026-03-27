"""Iterate workflow - main orchestration.

This module implements the iterative training workflow for DeePKS.
It combines SCF calculations and model training in an iterative loop.
"""

import os
from typing import Dict, Any

from deepks.io.input import build_runtime_config_from_raw
from .prepare import prepare_iterate


def run_iterate_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run iterative training workflow.

    This is the main entry point for iterative training. It orchestrates
    multiple iterations of SCF + Train cycles.

    Physical Process:
    For each iteration:
    1. SCF: Run SCF calculations with current model
    2. Train: Train model on SCF results
    3. Repeat until convergence or max iterations

    Args:
        config: Configuration dictionary containing:
            - type: 'iterate'
            - systems_train: Training system paths
            - systems_test: Test system paths (optional)
            - n_iter: Number of iterations
            - scf_soft: SCF backend ('pyscf' or 'abacus')
            - scf_input: SCF parameters
            - scf_machine: SCF machine settings
            - train_input: Training parameters
            - train_machine: Training machine settings
            - init_model: Initial model path (optional)
            - init_scf: Initial SCF parameters
            - init_train: Initial training parameters
            - proj_basis: Projection basis file
            - workdir: Working directory
            - share_folder: Shared files folder
            - cleanup: Whether to cleanup intermediate files

    Returns:
        dict: Iteration results with final model and statistics
    """
    runtime_config = build_runtime_config_from_raw(config)
    iterate_config = runtime_config['raw_config']

    # Prepare iteration workflow
    iteration_workflow, workdir, record_file = prepare_iterate(iterate_config)

    # Check if we should restart
    if os.path.exists(record_file):
        # print(f'# restarting from {record_file}')
        iteration_workflow.restart()
    else:
        # print(f'# starting new iteration in {workdir}')
        iteration_workflow.run()

    # Collect results
    n_iter = iterate_config.get('n_iter', 0)
    final_model = os.path.join(workdir, f'iter.{n_iter:02d}', '01.train', 'model.pth')

    results = {
        'final_model': final_model,
        'n_iterations': n_iter,
        'workdir': workdir
    }

    return results
