"""Unified command dispatcher for DeePKS."""

import os
import sys


def dispatch_command(config):
    """Dispatch to appropriate command handler based on config.

    Args:
        config: Unified configuration dictionary

    Raises:
        ValueError: If type is not recognized
    """
    task_type = config.get('type')

    if task_type == 'train':
        # Use new train workflow
        from deepks.workflows.train import run_train_workflow
        run_train_workflow(config)
    elif task_type == 'test':
        from deepks.pipelines.test.test import main as test_main
        test_main(**config)
    elif task_type == 'scf':
        # Use new SCF workflow
        from deepks.workflows.scf import run_scf_workflow
        run_scf_workflow(config)
    elif task_type == 'stats':
        from deepks.cli.main import _get_physics_backend
        scf_soft = config.get('scf_soft', 'pyscf')
        backend = _get_physics_backend(scf_soft)
        backend.collect_stats(**config)
    elif task_type == 'iterate':
        # Use new iterate workflow
        from deepks.workflows.iterate import run_iterate_workflow
        run_iterate_workflow(config)
    else:
        raise ValueError(f"Unknown type: {task_type}")
