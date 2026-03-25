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
        from deepks.ml.eval.test import main as test_main
        test_main(**config)
    elif task_type == 'scf':
        # Use new SCF workflow
        from deepks.workflows.scf import run_scf_workflow
        run_scf_workflow(config)
    elif task_type == 'stats':
        from deepks.physics.backends import get_scf_backend
        scf_soft = config.get('scf_soft', 'pyscf')
        backend = get_scf_backend(scf_soft)
        backend.collect_stats(**config)
    elif task_type == 'scf_task':
        # Low-level per-task SCF runner (called by BatchTask in iterate workflow)
        from deepks.physics.backends.pyscf.run import main as scf_run_main
        # Strip orchestration-only keys that pyscf.run.main() doesn't understand
        scf_config = {k: v for k, v in config.items()
                      if k not in ('type', 'scf_soft')}
        scf_run_main(**scf_config)
    elif task_type == 'train_task':
        # Low-level training runner (called by BatchTask in iterate workflow)
        from deepks.ml.train.train import main as train_main
        train_config = {k: v for k, v in config.items()
                        if k not in ('type',)}
        train_main(**train_config)
    elif task_type == 'iterate':
        # Use new iterate workflow
        from deepks.workflows.iterate import run_iterate_workflow
        run_iterate_workflow(config)
    else:
        raise ValueError(f"Unknown type: {task_type}")
