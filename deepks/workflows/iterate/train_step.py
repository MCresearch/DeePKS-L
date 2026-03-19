"""Iterate workflow - Train step.

This module creates the training step for iteration workflow.
"""

from typing import Dict, Any, Optional

from deepks.pipelines.iterate.template import make_train


# Constants
DATA_TRAIN = "data_train"
DATA_TEST = "data_test"
MODEL_FILE = "model.pth"
TRN_STEP_DIR = "01.train"


def create_train_step(train_args_name: Optional[str],
                      train_machine: Dict[str, Any],
                      proj_basis: Optional[str],
                      share_folder: str,
                      cleanup: bool):
    """Create training step for iteration.

    Args:
        train_args_name: Training arguments file name
        train_machine: Machine settings
        proj_basis: Projection basis file
        share_folder: Share folder path
        cleanup: Whether to cleanup

    Returns:
        Task: Training step task
    """
    train_step = make_train(
        source_train=DATA_TRAIN,
        source_test=DATA_TEST,
        restart=True,
        source_model=MODEL_FILE,
        save_model=MODEL_FILE,
        source_pbasis=proj_basis,
        source_arg=train_args_name,
        workdir=TRN_STEP_DIR,
        share_folder=share_folder,
        cleanup=cleanup,
        **train_machine
    )

    return train_step
