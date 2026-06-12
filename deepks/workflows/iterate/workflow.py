"""Iterate workflow - main orchestration.

This workflow now expects configuration to be normalized by ``deepks.config``
before dispatch. It only builds and executes the iterate workflow tree.
"""

import os
from typing import Dict, Any

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
        config: Iteration configuration already normalized by ``deepks.config``.

    Returns:
        dict: Iteration results with final model and statistics
    """
    iterate_config = config

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
    iterate_cfg = iterate_config.get("iterate") if isinstance(iterate_config.get("iterate"), dict) else {}
    n_iter = iterate_cfg.get('n_iter', 0)
    final_model = os.path.join(workdir, f'iter.{n_iter:02d}', '01.train', 'model.pth')

    results = {
        'final_model': final_model,
        'n_iterations': n_iter,
        'workdir': workdir
    }

    return results
