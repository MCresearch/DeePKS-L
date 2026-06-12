"""Iterate workflow - Prepare stage."""

import os
from copy import deepcopy
from typing import Any, Dict, Tuple

from deepks.workflows.iterate.support import (
    MODEL_FILE,
    RECORD,
    TRN_STEP_DIR,
    build_abacus_iterate_scf_kwargs,
    check_share_folder,
    materialize_hierarchical_level_scf_config,
    make_scf,
    make_train,
    prepare_iterate_snapshots,
    resolve_hierarchical_iterate_levels,
    resolve_scf_profile_levels,
)
from deepks.orchestration.workflow.workflow import Iteration, Sequence
from deepks.workflows.iterate.abacus import make_scf_abacus


DATA_TRAIN = "data_train"
DATA_TEST = "data_test"
SCF_STEP_DIR = "00.scf"


def _create_scf_step(
    systems_train: Any,
    systems_test: Any,
    scf_soft: str,
    scf_config: Dict[str, Any],
    scf_machine: Dict[str, Any],
    proj_basis: Any,
    share_folder: str,
    cleanup: bool,
    *,
    no_model: bool = False,
    workdir: str = SCF_STEP_DIR,
):
    if scf_soft.lower() == "abacus":
        abacus_kwargs = build_abacus_iterate_scf_kwargs(scf_config)
        return make_scf_abacus(
            systems_train=systems_train,
            systems_test=systems_test,
            train_dump=DATA_TRAIN,
            test_dump=DATA_TEST,
            dispatcher=scf_machine.get("dispatcher"),
            resources=scf_machine.get("resources"),
            dpdispatcher_machine=scf_machine.get("dpdispatcher_machine"),
            dpdispatcher_resources=scf_machine.get("dpdispatcher_resources"),
            group_size=scf_machine.get("group_size", 1),
            no_model=no_model,
            model_file=MODEL_FILE if not no_model else None,
            workdir=workdir,
            share_folder=share_folder,
            cleanup=cleanup,
            orb_files=abacus_kwargs["orb_files"],
            pp_files=abacus_kwargs["pp_files"],
            proj_file=abacus_kwargs["proj_file"],
            run_cmd=abacus_kwargs["run_cmd"],
            abacus_path=abacus_kwargs["abacus_path"],
            **abacus_kwargs["backend_kwargs"],
        )
    if scf_soft.lower() == "pyscf":
        return make_scf(
            systems_train=systems_train,
            systems_test=systems_test,
            train_dump=DATA_TRAIN,
            test_dump=DATA_TEST,
            no_model=no_model,
            task_config=scf_config,
            workdir=workdir,
            share_folder=share_folder,
            source_model=MODEL_FILE if not no_model else None,
            source_pbasis=proj_basis,
            cleanup=cleanup,
            **scf_machine,
        )
    raise ValueError(f"Unknown SCF backend: {scf_soft}. Available backends: ['pyscf', 'abacus']")


def _create_hierarchical_scf_step(
    level_metas,
    *,
    scf_soft: str,
    base_scf_config: Dict[str, Any],
    scf_machine: Dict[str, Any],
    proj_basis: Any,
    share_folder: str,
    cleanup: bool,
    no_model: bool = False,
):
    level_steps = []
    for level_meta in level_metas:
        systems_cfg = level_meta["systems"]
        level_index = int(level_meta["level"])
        level_train = systems_cfg.get("train_paths", [])
        level_test = systems_cfg.get("test_paths")
        level_scf_config = materialize_hierarchical_level_scf_config(base_scf_config, level_meta)
        level_steps.append(
            _create_scf_step(
                systems_train=level_train,
                systems_test=level_test,
                scf_soft=scf_soft,
                scf_config=level_scf_config,
                scf_machine=scf_machine,
                proj_basis=proj_basis,
                share_folder=share_folder,
                cleanup=cleanup,
                no_model=no_model,
                workdir=f"level.{level_index:02d}",
            )
        )
    return Sequence(level_steps, workdir=SCF_STEP_DIR)


def _create_train_step(
    train_config: Dict[str, Any],
    train_machine: Dict[str, Any],
    proj_basis: Any,
    share_folder: str,
    cleanup: bool,
    *,
    restart: bool = True,
    link_default_data: bool = True,
):
    return make_train(
        source_train=DATA_TRAIN if link_default_data else None,
        source_test=DATA_TEST if link_default_data else None,
        restart=restart,
        source_model=MODEL_FILE,
        save_model=MODEL_FILE,
        task_config=train_config,
        source_pbasis=proj_basis,
        data_train=DATA_TRAIN if link_default_data else None,
        data_test=DATA_TEST if link_default_data else None,
        workdir=TRN_STEP_DIR,
        share_folder=share_folder,
        cleanup=cleanup,
        **train_machine,
    )


def prepare_iterate(config: Dict[str, Any]) -> Tuple[Sequence, str, str]:
    """Prepare iteration workflow from packed iterate parameters."""
    iterate_param = dict(config)
    runtime = iterate_param.get("runtime") if isinstance(iterate_param.get("runtime"), dict) else {}
    iterate_cfg = iterate_param.get("iterate") if isinstance(iterate_param.get("iterate"), dict) else {}
    n_iter = iterate_cfg.get("n_iter", 0)
    cleanup = iterate_cfg.get("cleanup", False)
    use_init = bool(iterate_cfg.get("use_init", False))
    snapshot = prepare_iterate_snapshots(iterate_param)
    systems_train = snapshot["systems_train"]
    systems_test = snapshot["systems_test"]
    workdir = snapshot["workdir"]
    share_folder = snapshot["share_folder"]
    scf_soft = snapshot["scf_soft"]
    proj_basis = snapshot["proj_basis"]
    scf_task_config = snapshot["scf_task_config"]
    train_task_config = snapshot["train_task_config"]
    init_scf_config = snapshot["init_scf_config"]
    init_train_config = snapshot["init_train_config"]
    first_iter_has_model = bool(use_init or snapshot["initial_model_exists"])
    hierarchical_levels = resolve_hierarchical_iterate_levels(iterate_param)
    profile_levels = [] if hierarchical_levels else resolve_scf_profile_levels(iterate_param)
    scf_levels = hierarchical_levels or profile_levels

    if scf_levels:
        scf_step = _create_hierarchical_scf_step(
            scf_levels,
            scf_soft=scf_soft,
            base_scf_config=scf_task_config,
            scf_machine=deepcopy(runtime.get("scf", {}).get("execute", {})) if isinstance(runtime.get("scf"), dict) else {},
            proj_basis=proj_basis,
            share_folder=share_folder,
            cleanup=cleanup,
            no_model=not first_iter_has_model,
        )
    else:
        scf_step = _create_scf_step(
            systems_train=systems_train,
            systems_test=systems_test,
            scf_soft=scf_soft,
            scf_config=scf_task_config,
            scf_machine=deepcopy(runtime.get("scf", {}).get("execute", {})) if isinstance(runtime.get("scf"), dict) else {},
            proj_basis=proj_basis,
            share_folder=share_folder,
            cleanup=cleanup,
            no_model=not first_iter_has_model,
        )
    train_step = _create_train_step(
        train_config=train_task_config,
        train_machine=deepcopy(runtime.get("train", {}).get("execute", {})) if isinstance(runtime.get("train"), dict) else {},
        proj_basis=proj_basis,
        share_folder=share_folder,
        cleanup=cleanup,
        restart=first_iter_has_model,
        link_default_data=not scf_levels,
    )

    iteration_workflow = Iteration(
        [scf_step, train_step],
        iternum=n_iter,
        workdir=workdir,
        record_file=os.path.join(workdir, RECORD),
    )

    if use_init:
        if scf_levels:
            scf_init = _create_hierarchical_scf_step(
                scf_levels,
                scf_soft=scf_soft,
                base_scf_config=init_scf_config,
                scf_machine=deepcopy(runtime.get("scf", {}).get("execute", {})) if isinstance(runtime.get("scf"), dict) else {},
                proj_basis=proj_basis,
                share_folder=share_folder,
                cleanup=cleanup,
                no_model=not snapshot["initial_model_exists"],
            )
        else:
            scf_init = _create_scf_step(
                systems_train=systems_train,
                systems_test=systems_test,
                scf_soft=scf_soft,
                scf_config=init_scf_config,
                scf_machine=deepcopy(runtime.get("scf", {}).get("execute", {})) if isinstance(runtime.get("scf"), dict) else {},
                proj_basis=proj_basis,
                share_folder=share_folder,
                cleanup=cleanup,
                no_model=not snapshot["initial_model_exists"],
            )
        train_init = _create_train_step(
            train_config=init_train_config,
            train_machine=deepcopy(runtime.get("train", {}).get("execute", {})) if isinstance(runtime.get("train"), dict) else {},
            proj_basis=proj_basis,
            share_folder=share_folder,
            cleanup=cleanup,
            restart=bool(snapshot["initial_model_exists"]),
            link_default_data=not scf_levels,
        )

        init_folder = snapshot["share_path"] if snapshot["initial_model_exists"] else None
        init_iter = Sequence([scf_init, train_init], workdir="iter.init", init_folder=init_folder)
        iteration_workflow.prepend(init_iter)
        iteration_workflow.set_init_folder(os.path.join(workdir, "iter.init", TRN_STEP_DIR))
    elif snapshot["initial_model_exists"]:
        iteration_workflow.set_init_folder(snapshot["share_path"])

    return iteration_workflow, workdir, os.path.join(workdir, RECORD)
