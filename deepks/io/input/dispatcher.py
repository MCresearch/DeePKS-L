"""Unified command dispatcher for DeePKS."""

import os
import sys


def dispatch_command(config):
    """Dispatch to appropriate command handler based on config.

    Args:
        config: Unified configuration dictionary

    Raises:
        ValueError: If command is not recognized
    """
    command = config.get('command')

    if command == 'train':
        from deepks.pipelines.train.train import main as train_main
        train_main(**config)
    elif command == 'test':
        from deepks.pipelines.test.test import main as test_main
        test_main(**config)
    elif command == 'scf':
        from deepks.cli.main import _get_physics_backend
        scf_soft = config.pop('scf_soft', 'pyscf')
        backend = _get_physics_backend(scf_soft)
        backend.run_scf(**config)
    elif command == 'stats':
        from deepks.cli.main import _get_physics_backend
        scf_soft = config.get('scf_soft', 'pyscf')
        backend = _get_physics_backend(scf_soft)
        backend.collect_stats(**config)
    elif command == 'iterate':
        from deepks.pipelines.iterate.iterate import main as iterate_main
        iterate_main(**config)
    else:
        raise ValueError(f"Unknown command: {command}")
