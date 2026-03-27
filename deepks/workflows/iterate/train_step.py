"""Iterate workflow - Train step.

This module creates the training step for iteration workflow.
"""

from typing import Dict, Any, Optional

from .template import make_train


# Constants
DATA_TRAIN = "data_train"
DATA_TEST = "data_test"
MODEL_FILE = "model.pth"
TRN_STEP_DIR = "01.train"


def create_train_step(train_config: Optional[Dict[str, Any]],
                      train_machine: Dict[str, Any],
                      proj_basis: Optional[str],
                      share_folder: str,
                      cleanup: bool,
                      restart: bool = True):
    """Create training step for iteration.

    Args:
        train_config: Finalized train child-task config snapshot
        train_machine: Machine settings
        proj_basis: Shared projection-basis file name, if materialized in share/
        share_folder: Share folder path
        cleanup: Whether to cleanup
        restart: Whether to load model from previous iteration.
                 Set False for the very first iteration when no prior model exists.

    Returns:
        Task: Training step task
    """
    train_step = make_train(
        source_train=DATA_TRAIN,
        source_test=DATA_TEST,
        restart=restart,
        source_model=MODEL_FILE,
        save_model=MODEL_FILE,
        task_config=train_config,
        source_pbasis=proj_basis,
        workdir=TRN_STEP_DIR,
        share_folder=share_folder,
        cleanup=cleanup,
        **train_machine
    )

    return train_step
