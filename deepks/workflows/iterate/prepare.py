"""Iterate workflow - Prepare stage.

This module handles the preparation stage of iterate workflow:
- Set up working directory structure
- Prepare shared files
- Create iteration workflow with optional init iteration
"""

import os
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional

from deepks.orchestration.workflow.workflow import Iteration, Sequence
from deepks.io.utils import copy_file, save_yaml, load_yaml
from deepks.io.input.config import PYSCF_BACKEND_KEYS, ABACUS_BACKEND_KEYS
from deepks.io.input.merger import merge_configs
from deepks.physics.backends.pyscf.basis import load_basis, save_basis
from deepks.physics.defaults import DEFAULT_SCF_ARGS_ABACUS
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

TRAIN_TASK_CONFIG_KEYS = (
    "model_args",
    "data_args",
    "preprocess_args",
    "train_args",
    "proj_basis",
    "fit_elem",
    "seed",
    "device",
    "model_file",
    "e_name",
    "d_name",
    "group",
    "output_prefix",
    "batch_size",
)

SCF_TASK_COMMON_KEYS = (
    "dump_fields",
    "group",
    "device",
    "model_file",
    "proj_basis",
)


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


def _load_task_config_source(data: Any, share_path: str, name: str) -> Dict[str, Any]:
    """Resolve a legacy iterate child-config source into an in-memory dict."""
    if data in (None, False):
        return {}

    if data is True:
        return {}
    elif isinstance(data, str):
        data = load_yaml(data)
    elif isinstance(data, dict):
        data = deepcopy(data)
    else:
        raise ValueError(f"Invalid argument: {data}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected task config dict for {name}, got {type(data)}")
    return data


def _pick_config(config: Dict[str, Any], keys) -> Dict[str, Any]:
    return {key: deepcopy(config[key]) for key in keys if key in config}


def _save_task_snapshot(config: Dict[str, Any], share_path: str, name: str) -> str:
    """Persist a finalized child-task config snapshot into share/."""
    save_yaml(config, os.path.join(share_path, name))
    return name


def _build_train_task_config(config: Dict[str, Any], source: Any, share_path: str, name: str) -> Dict[str, Any]:
    base_config = {"type": "train"}
    base_config.update(_pick_config(config, TRAIN_TASK_CONFIG_KEYS))
    return merge_configs(base_config, _load_task_config_source(source, share_path, name))


def _build_pyscf_scf_task_config(config: Dict[str, Any], source: Any, share_path: str, name: str) -> Dict[str, Any]:
    base_config = {"type": "scf", "scf_soft": "pyscf"}
    base_config.update(_pick_config(config, SCF_TASK_COMMON_KEYS))
    base_config.update(_pick_config(config, PYSCF_BACKEND_KEYS))
    return merge_configs(base_config, _load_task_config_source(source, share_path, name))


def _build_abacus_scf_task_config(config: Dict[str, Any], block_key: str) -> Dict[str, Any]:
    base_config = {"type": "scf", "scf_soft": "abacus"}
    block = config.get(block_key, {})
    if isinstance(block, dict):
        base_config.update(_pick_config(block, ABACUS_BACKEND_KEYS))
    return base_config


def prepare_iterate(config: Dict[str, Any]) -> Tuple[Sequence, str, str]:
    """Prepare iteration workflow (Stage 1).

    This function sets up the working directory, prepares shared files,
    and creates the iteration workflow. If init_scf or init_train is specified,
    it creates an initial iteration (iter.init) with energy-only training.

    Behaviour of init_model and iter.00/00.scf:
    - init_model=False (default): no prior model exists before iter.00.
      iter.00/00.scf runs without a model (no_model=True, pure DFT) and
      iter.00/01.train trains from scratch (restart=False).
    - init_model=<path>: the given model is copied into share/ and
      iter.00/00.scf links it (no_model=False); iter.00/01.train restarts
      from it (restart=True).
    - init_scf / init_train set: an iter.init is prepended; its 01.train
      produces a bootstrap model that iter.00/00.scf can link; subsequent
      iterations all use restart=True.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (iteration_workflow, workdir, record_file)
    """
    config = dict(config)

    # Extract configuration
    systems_train = config.get('systems_train')
    systems_test = config.get('systems_test')
    n_iter = config.get('n_iter', 0)
    workdir = config.get('workdir', '.')
    share_folder = config.get('share_folder', 'share')
    cleanup = config.get('cleanup', False)
    strict = config.get('strict', True)
    scf_soft = config.get('scf_soft', 'pyscf')
    # init_scf / init_train: trigger an iter.init bootstrap iteration
    init_scf = config.get('init_scf', None)
    init_train = config.get('init_train', None)

    # init_model: False  -> train from scratch, no model before iter.00
    #             True   -> share/model.pth already exists, use it
    #             <path> -> copy that file to share/ and use it
    init_model = config.get('init_model', False)

    # Create directories
    os.makedirs(workdir, exist_ok=True)
    share_path = os.path.join(workdir, share_folder)
    os.makedirs(share_path, exist_ok=True)

    # Handle SCF arguments
    scf_input = config.get('scf_input', True)
    scf_machine = config.get('scf_machine')

    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)

    # Handle training arguments
    train_input = config.get('train_input', True)
    train_machine = config.get('train_machine')
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)

    # Handle projection basis
    proj_basis = config.get('proj_basis')
    if proj_basis is not None and scf_soft.lower() == 'pyscf':
        save_basis(os.path.join(share_path, PROJ_BASIS), load_basis(proj_basis))
        proj_basis = PROJ_BASIS
        config['proj_basis'] = proj_basis

    if scf_soft.lower() == 'abacus':
        scf_task_config = _build_abacus_scf_task_config(config, 'scf_abacus')
        scf_task_config = check_arg_dict(scf_task_config, {"type": "scf", "scf_soft": "abacus", **DEFAULT_SCF_ARGS_ABACUS}, strict)
        _save_task_snapshot(scf_task_config, share_path, SCF_ARGS_NAME_ABACUS)
    else:
        scf_task_config = _build_pyscf_scf_task_config(config, scf_input, share_path, SCF_ARGS_NAME)
        _save_task_snapshot(scf_task_config, share_path, SCF_ARGS_NAME)

    train_task_config = _build_train_task_config(config, train_input, share_path, TRN_ARGS_NAME)
    _save_task_snapshot(train_task_config, share_path, TRN_ARGS_NAME)

    # Copy explicit init_model path into share/ so iter.00 can link it.
    if isinstance(init_model, str):
        dst_model = os.path.join(share_path, MODEL_FILE)
        copy_file(init_model, dst_model)
        init_model = True

    # Whether the very first main iteration (iter.00) will have a model
    # available from a previous folder.
    # True when: an iter.init bootstrap is requested, OR an init_model was given.
    first_iter_has_model = bool(init_scf or init_train or init_model)

    # ------------------------------------------------------------------
    # Build the per-iteration SCF and train step templates.
    #
    # For main iterations (iter.00 .. iter.N-1):
    #   - If first_iter_has_model is False, iter.00/00.scf must run without
    #     a model (no_model=True) and iter.00/01.train must start from
    #     scratch (restart=False).  All subsequent iterations chain normally.
    #   - If first_iter_has_model is True, every iteration links the model
    #     from its previous folder (no_model=False) and restarts training.
    #
    # The Iteration class handles chaining iter.N-1/01.train -> iter.N/00.scf
    # for N >= 1.  For iter.00 the init_folder argument provides the source.
    # ------------------------------------------------------------------

    # Main-loop SCF step: needs a model from prev folder (normal case)
    scf_step = create_scf_step(
        systems_train=systems_train,
        systems_test=systems_test,
        scf_soft=scf_soft,
        scf_config=scf_task_config,
        scf_machine=scf_machine,
        proj_basis=proj_basis,
        share_folder=share_folder,
        cleanup=cleanup,
        no_model=not first_iter_has_model  # no model for iter.00 when starting from scratch
    )

    # Main-loop train step
    train_step = create_train_step(
        train_config=train_task_config,
        train_machine=train_machine,
        proj_basis=proj_basis,
        share_folder=share_folder,
        cleanup=cleanup,
        restart=first_iter_has_model  # restart only when a prior model exists
    )

    # Build the main Iteration workflow
    iteration_workflow = Iteration(
        [scf_step, train_step],
        iternum=n_iter,
        workdir=workdir,
        record_file=os.path.join(workdir, RECORD)
    )

    if init_scf or init_train:
        # Prepare init SCF step (always no_model=True: pure DFT bootstrap)
        if scf_soft.lower() == 'abacus':
            init_scf_config = _build_abacus_scf_task_config(config, 'init_scf_abacus')
            init_scf_config = check_arg_dict(
                init_scf_config,
                {"type": "scf", "scf_soft": "abacus", **DEFAULT_SCF_ARGS_ABACUS},
                strict,
            )
            _save_task_snapshot(init_scf_config, share_path, INIT_SCF_NAME_ABACUS)

            scf_init = create_scf_step(
                systems_train=systems_train,
                systems_test=systems_test,
                scf_soft=scf_soft,
                scf_config=init_scf_config,
                scf_machine=scf_machine,
                proj_basis=proj_basis,
                share_folder=share_folder,
                cleanup=cleanup,
                no_model=True  # No model for init SCF
            )
        else:  # pyscf
            init_scf_config = _build_pyscf_scf_task_config(config, init_scf, share_path, INIT_SCF_NAME)
            _save_task_snapshot(init_scf_config, share_path, INIT_SCF_NAME)
            scf_init = create_scf_step(
                systems_train=systems_train,
                systems_test=systems_test,
                scf_soft=scf_soft,
                scf_config=init_scf_config,
                scf_machine=scf_machine,
                proj_basis=proj_basis,
                share_folder=share_folder,
                cleanup=cleanup,
                no_model=True  # No model for init SCF
            )

        # Prepare init train step (energy-only, no restart)
        init_train_config = _build_train_task_config(config, init_train, share_path, INIT_TRN_NAME)
        _save_task_snapshot(init_train_config, share_path, INIT_TRN_NAME)
        train_init = create_train_step(
            train_config=init_train_config,
            train_machine=train_machine,
            proj_basis=proj_basis,
            share_folder=share_folder,
            cleanup=cleanup,
            restart=False  # bootstrap: no prior model
        )

        # Create init iteration sequence
        init_iter = Sequence([scf_init, train_init], workdir="iter.init")

        # Prepend init iteration to main workflow
        iteration_workflow.prepend(init_iter)

        # The first main iteration links its prev folder from iter.init/01.train
        iteration_workflow.set_init_folder(
            os.path.join(workdir, "iter.init", TRN_STEP_DIR)
        )

    elif init_model:
        # An explicit model was copied to share/; point iter.00/00.scf at it.
        iteration_workflow.set_init_folder(share_path)

    record_file = os.path.join(workdir, RECORD)

    return iteration_workflow, workdir, record_file
