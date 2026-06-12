"""Iterate workflow child-task snapshot helpers."""

import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np

from deepks.io.utils import check_share_folder, copy_file, save_yaml
from deepks.physics.backends.pyscf.basis import load_basis, save_basis

SCF_ARGS_NAME = "scf_input.yaml"
SCF_ARGS_NAME_ABACUS = "scf_abacus.yaml"
TRN_ARGS_NAME = "train_input.yaml"
INIT_SCF_NAME = "init_scf.yaml"
INIT_SCF_NAME_ABACUS = "init_scf_abacus.yaml"
INIT_TRN_NAME = "init_train.yaml"

MODEL_FILE = "model.pth"
PROJ_BASIS = "proj_basis.npz"
TRN_STEP_DIR = "01.train"
RECORD = "RECORD"


def _yaml_safe(value: Any):
    if isinstance(value, dict):
        return {key: _yaml_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_task_snapshot(config: Dict[str, Any], share_path: str, name: str) -> str:
    save_yaml(_yaml_safe(config), os.path.join(share_path, name))
    return name


def prepare_iterate_snapshots(iterate_param: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize iterate share files and return prepared child-task payloads."""
    runtime = iterate_param.get("runtime") if isinstance(iterate_param.get("runtime"), dict) else {}
    data = iterate_param.get("data") if isinstance(iterate_param.get("data"), dict) else {}
    physics = iterate_param.get("physics") if isinstance(iterate_param.get("physics"), dict) else {}
    ml = iterate_param.get("ml") if isinstance(iterate_param.get("ml"), dict) else {}
    iterate_cfg = iterate_param.get("iterate") if isinstance(iterate_param.get("iterate"), dict) else {}
    tasks = iterate_cfg.get("tasks") if isinstance(iterate_cfg.get("tasks"), dict) else {}
    main_tasks = tasks.get("main") if isinstance(tasks.get("main"), dict) else {}
    init_tasks = tasks.get("init") if isinstance(tasks.get("init"), dict) else {}

    workdir = runtime.get("workdir", ".")
    share_folder = runtime.get("share_folder", "share")
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    scf_soft = str(backend.get("name", "pyscf")).lower()
    checkpoint = ml.get("checkpoint") if isinstance(ml.get("checkpoint"), dict) else {}
    initial_model = checkpoint.get("file")
    representation = physics.get("representation") if isinstance(physics.get("representation"), dict) else {}
    rep_params = representation.get("params") if isinstance(representation.get("params"), dict) else {}
    proj_basis = rep_params.get("proj_basis")

    share_path = os.path.join(workdir, share_folder)
    os.makedirs(share_path, exist_ok=True)

    if proj_basis is not None and scf_soft == "pyscf":
        save_basis(os.path.join(share_path, PROJ_BASIS), load_basis(proj_basis))
        proj_basis = PROJ_BASIS

    scf_task_config = deepcopy(main_tasks.get("scf"))
    train_task_config = deepcopy(main_tasks.get("train"))
    if not isinstance(scf_task_config, dict) or not isinstance(train_task_config, dict):
        raise ValueError("Iterate packed config missing main child task snapshots")

    _save_task_snapshot(
        scf_task_config,
        share_path,
        SCF_ARGS_NAME_ABACUS if scf_soft == "abacus" else SCF_ARGS_NAME,
    )
    _save_task_snapshot(train_task_config, share_path, TRN_ARGS_NAME)

    initial_model_exists = False
    if isinstance(initial_model, str):
        copy_file(initial_model, os.path.join(share_path, MODEL_FILE))
        initial_model_exists = True
    elif initial_model:
        initial_model_exists = True

    init_scf_config = deepcopy(init_tasks.get("scf"))
    init_train_config = deepcopy(init_tasks.get("train"))
    if iterate_cfg.get("use_init", False):
        if not isinstance(init_scf_config, dict) or not isinstance(init_train_config, dict):
            raise ValueError("Iterate packed config missing init child task snapshots")
        _save_task_snapshot(
            init_scf_config,
            share_path,
            INIT_SCF_NAME_ABACUS if scf_soft == "abacus" else INIT_SCF_NAME,
        )
        _save_task_snapshot(init_train_config, share_path, INIT_TRN_NAME)

    return {
        "workdir": workdir,
        "share_path": share_path,
        "share_folder": share_folder,
        "systems_train": [] if data.get("train") is None else data.get("train", []),
        "systems_test": data.get("test"),
        "scf_soft": scf_soft,
        "proj_basis": proj_basis,
        "initial_model_exists": initial_model_exists,
        "scf_task_config": scf_task_config,
        "train_task_config": train_task_config,
        "init_scf_config": init_scf_config,
        "init_train_config": init_train_config,
    }
