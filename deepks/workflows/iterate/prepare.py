"""Iterate workflow - Prepare stage.

This module handles the preparation stage of iterate workflow:
- Set up working directory structure
- Prepare shared files
- Create iteration workflow
"""

import os
from typing import Dict, Any, Tuple

from deepks.orchestration.workflow.workflow import Iteration
from deepks.utils import copy_file, save_yaml, load_yaml, load_basis, save_basis
from deepks.default import DEFAULT_SCF_ARGS_ABACUS
from .scf_step import create_scf_step
from .train_step import create_train_step


# Default machine configurations
DEFAULT_SCF_MACHINE = {
    "sub_size": 1,
    "sub_res": None,
    "group_size": 1,
    "ingroup_parallel": 1,
    "dispatcher": None,
    "resources": None,
    "python": "python",
    "dpdispatcher_machine": None,
    "dpdispatcher_resources": None
}

DEFAULT_TRN_MACHINE = {
    "dispatcher": None,
    "resources": None,
    "python": "python",
    "dpdispatcher_machine": None,
    "dpdispatcher_resources": None
}

# File names
SCF_ARGS_NAME = "scf_input.yaml"
SCF_ARGS_NAME_ABACUS = "scf_abacus.yaml"
TRN_ARGS_NAME = "train_input.yaml"
INIT_SCF_NAME = "init_scf.yaml"
INIT_SCF_NAME_ABACUS = "init_scf_abacus.yaml"
INIT_TRN_NAME = "init_train.yaml"

DATA_TRAIN = "data_train"
DATA_TEST = "data_test"
MODEL_FILE = "model.pth"
PROJ_BASIS = "proj_basis.npz"

SCF_STEP_DIR = "00.scf"
TRN_STEP_DIR = "01.train"

RECORD = "RECORD"


def check_share_folder(data: Any, name: str, share_folder: str = "share") -> str:
    """Save data to share_folder/name.

    Args:
        data: Data to save (True/str/dict)
        name: Target file name
        share_folder: Share folder path

    Returns:
        str: Relative path to saved file, or None
    """
    if not data:
        return None

    dst_name = os.path.join(share_folder, name)

    if data is True:
        # Check existence
        if not os.path.exists(dst_name):
            raise FileNotFoundError(f"No required file: {dst_name}")
        return name
    elif isinstance(data, str) and os.path.exists(data):
        # Copy file
        copy_file(data, dst_name)
        return name
    elif isinstance(data, dict):
        # Save as YAML
        save_yaml(data, dst_name)
        return name
    else:
        raise ValueError(f"Invalid argument: {data}")


def check_arg_dict(data: Any, default: Dict, strict: bool = True) -> Dict:
    """Check and merge arguments with defaults.

    Args:
        data: Input arguments (dict/str/None)
        default: Default arguments
        strict: If True, only keep keys in default

    Returns:
        dict: Merged arguments
    """
    if data is None:
        data = {}
    if isinstance(data, str):
        data = load_yaml(data)

    allowed = {k: v for k, v in data.items() if k in default}
    unknown = {k: v for k, v in data.items() if k not in default}

    if unknown and strict:
        print(f"Warning: Unknown arguments will be ignored: {list(unknown.keys())}")

    result = {**default, **allowed}
    return result


def prepare_iterate(config: Dict[str, Any]) -> Tuple[Iteration, str, str]:
    """Prepare iteration workflow (Stage 1).

    This function sets up the working directory, prepares shared files,
    and creates the iteration workflow.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (iteration_workflow, workdir, record_file)
    """
    # Extract configuration
    systems_train = config.get('systems_train')
    systems_test = config.get('systems_test')
    n_iter = config.get('n_iter', 0)
    workdir = config.get('workdir', '.')
    share_folder = config.get('share_folder', 'share')
    cleanup = config.get('cleanup', False)
    strict = config.get('strict', True)
    scf_soft = config.get('scf_soft', 'pyscf')

    # Create directories
    os.makedirs(workdir, exist_ok=True)
    share_path = os.path.join(workdir, share_folder)
    os.makedirs(share_path, exist_ok=True)

    # Handle SCF arguments
    scf_input = config.get('scf_input', True)
    scf_machine = config.get('scf_machine')
    scf_abacus = config.get('scf_abacus')

    if scf_soft.lower() == 'abacus':
        scf_args_name = check_share_folder(scf_abacus, SCF_ARGS_NAME_ABACUS, share_path)
        scf_abacus = check_arg_dict(scf_abacus, DEFAULT_SCF_ARGS_ABACUS, strict)
    else:
        scf_args_name = check_share_folder(scf_input, SCF_ARGS_NAME, share_path)

    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)

    # Handle training arguments
    train_input = config.get('train_input', True)
    train_machine = config.get('train_machine')

    train_args_name = check_share_folder(train_input, TRN_ARGS_NAME, share_path)
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)

    # Handle projection basis
    proj_basis = config.get('proj_basis')
    if proj_basis is not None and scf_soft.lower() == 'pyscf':
        save_basis(os.path.join(share_path, PROJ_BASIS), load_basis(proj_basis))
        proj_basis = PROJ_BASIS

    # Create SCF step
    scf_step = create_scf_step(
        systems_train=systems_train,
        systems_test=systems_test,
        scf_soft=scf_soft,
        scf_args=scf_abacus if scf_soft.lower() == 'abacus' else None,
        scf_machine=scf_machine,
        proj_basis=proj_basis,
        share_folder=share_folder,
        cleanup=cleanup
    )

    # Create training step
    train_step = create_train_step(
        train_args_name=train_args_name,
        train_machine=train_machine,
        proj_basis=proj_basis,
        share_folder=share_folder,
        cleanup=cleanup
    )

    # Create iteration workflow
    iteration_workflow = Iteration(
        [scf_step, train_step],
        n_iter=n_iter,
        workdir=workdir,
        record_file=RECORD
    )

    record_file = os.path.join(workdir, RECORD)

    return iteration_workflow, workdir, record_file
