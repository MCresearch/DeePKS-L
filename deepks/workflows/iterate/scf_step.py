"""Iterate workflow - SCF step.

This module creates the SCF step for iteration workflow.
"""

from typing import Dict, Any, Optional

from deepks.pipelines.iterate.template import make_scf
from deepks.pipelines.iterate.template_abacus import make_scf_abacus


# Constants
DATA_TRAIN = "data_train"
DATA_TEST = "data_test"
MODEL_FILE = "model.pth"
SCF_STEP_DIR = "00.scf"


def create_scf_step(systems_train: Any,
                    systems_test: Optional[Any],
                    scf_soft: str,
                    scf_args: Optional[Dict[str, Any]],
                    scf_machine: Dict[str, Any],
                    proj_basis: Optional[str],
                    share_folder: str,
                    cleanup: bool,
                    no_model: bool = False):
    """Create SCF step for iteration.

    Args:
        systems_train: Training systems
        systems_test: Test systems
        scf_soft: SCF backend ('pyscf' or 'abacus')
        scf_args: SCF arguments (for ABACUS)
        scf_machine: Machine settings
        proj_basis: Projection basis file
        share_folder: Share folder path
        cleanup: Whether to cleanup
        no_model: Whether to run without model (for init iteration)

    Returns:
        Task: SCF step task
    """
    if scf_soft.lower() == 'abacus':
        # Merge scf_args with scf_machine
        scf_config = dict(scf_args, **scf_machine) if scf_args else scf_machine

        scf_step = make_scf_abacus(
            systems_train=systems_train,
            systems_test=systems_test,
            train_dump=DATA_TRAIN,
            test_dump=DATA_TEST,
            no_model=no_model,
            model_file=MODEL_FILE if not no_model else None,
            workdir=SCF_STEP_DIR,
            share_folder=share_folder,
            cleanup=cleanup,
            **scf_config
        )
    elif scf_soft.lower() == 'pyscf':
        scf_step = make_scf(
            systems_train=systems_train,
            systems_test=systems_test,
            train_dump=DATA_TRAIN,
            test_dump=DATA_TEST,
            no_model=no_model,
            workdir=SCF_STEP_DIR,
            share_folder=share_folder,
            source_arg="scf_input.yaml",
            source_model=MODEL_FILE if not no_model else None,
            source_pbasis=proj_basis,
            cleanup=cleanup,
            **scf_machine
        )
    else:
        raise ValueError(
            f"Unknown SCF backend: {scf_soft}. "
            f"Available backends: ['pyscf', 'abacus']"
        )

    return scf_step
