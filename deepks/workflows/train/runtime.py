"""Train workflow runtime helpers.

This module owns train-task assembly and execution glue used by the train
workflow. It translates packed task configs into readers/model runtime config,
then delegates actual optimization to the selected recipe and ML loop.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from deepks.interface.adapters import fit_elem_const, resolve_hierarchical_model_levels, resolve_model_args
from deepks.interface.objectives import build_descriptor_property_objective_args
from deepks.interface.recipes._hierarchical_helpers import (
    build_hierarchical_descriptor_objective_args as _build_hierarchical_descriptor_objective_args,
    build_stage_data_specs as _build_stage_data_specs,
    normalize_hierarchical_primary_output as _normalize_hierarchical_primary_output,
    normalize_hierarchical_terms as _normalize_hierarchical_terms,
)
from deepks.interface.registry import (
    HIERARCHICAL_REGRESSION_RECIPE_NAME,
    get_recipe,
    get_recipe_name,
)
from deepks.io.model_artifacts import load_elem_table_sidecar, save_elem_table_sidecar
from deepks.io.readers import GroupReader
from deepks.io import utils as _io_utils
from deepks.io.utils import load_elem_table


def prepare_train_runtime(config: Dict[str, Any]) -> Tuple[GroupReader, Optional[GroupReader], Dict[str, Any]]:
    """Build readers and the model runtime config for a packed train task."""
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    runtime_io = runtime.get("io") if isinstance(runtime.get("io"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    checkpoint = ml.get("checkpoint") if isinstance(ml.get("checkpoint"), dict) else {}
    train_cfg = ml.get("train") if isinstance(ml.get("train"), dict) else {}
    loader_cfg = data.get("loader") if isinstance(data.get("loader"), dict) else {}
    stage_cfgs = data.get("stages") if isinstance(data.get("stages"), list) else []
    targets_cfg = data.get("targets") if isinstance(data.get("targets"), dict) else {}
    representation = physics.get("representation") if isinstance(physics.get("representation"), dict) else {}
    rep_params = representation.get("params") if isinstance(representation.get("params"), dict) else {}
    model_cfg = ml.get("model") if isinstance(ml.get("model"), dict) else {}
    hierarchy_levels = resolve_hierarchical_model_levels(config)
    objective = ml.get("objective") if isinstance(ml.get("objective"), dict) else {}

    seed = runtime.get("seed")
    if seed is None:
        seed = np.random.randint(0, 2**32)
    print(f"# using seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_paths = [] if data.get("train") is None else data.get("train", [])
    test_paths = data.get("test")
    data_args = deepcopy(loader_cfg)
    if "batch_size" in train_cfg:
        data_args["batch_size"] = train_cfg["batch_size"]
    if "group_batch" in train_cfg:
        data_args["group_batch"] = train_cfg["group_batch"]

    target_name_map = {
        "energy": "e_name",
        "force": "f_name",
        "stress": "s_name",
        "orbital": "o_name",
        "v_delta": "h_name",
        "vdr": "hr_name",
        "phi": "h_base_name",
        "band": "h_ref_name",
    }
    for src_key, dst_key in target_name_map.items():
        if src_key in targets_cfg:
            data_args[dst_key] = deepcopy(targets_cfg[src_key])
    rep_name = representation.get("name")
    if rep_name:
        data_args["d_name"] = rep_name
    if targets_cfg.get("hamiltonian_levels") is not None:
        data_args["hamiltonian_level_names"] = deepcopy(targets_cfg.get("hamiltonian_levels"))

    model_args = deepcopy(model_cfg.get("args", {}))
    recipe_name = get_recipe_name(config=config)
    recipe = get_recipe(recipe=recipe_name)
    model_family = model_cfg.get("family") or recipe.schema.model_family
    proj_basis = rep_params.get("proj_basis")
    fit_elem = ml.get("fit_elem", False)
    restart = checkpoint.get("restart")
    if proj_basis is not None:
        model_args = {**model_args, "proj_basis": proj_basis}
    objective_terms = []
    if recipe_name == HIERARCHICAL_REGRESSION_RECIPE_NAME:
        objective_terms = _normalize_hierarchical_terms(objective)
    primary_output = _normalize_hierarchical_primary_output(objective)
    if recipe_name == HIERARCHICAL_REGRESSION_RECIPE_NAME and hierarchy_levels:
        if model_args.get("n_levels") is None and model_args.get("max_depth") is None:
            model_args["n_levels"] = len(hierarchy_levels)

    train_paths = _io_utils.load_dirs(train_paths)
    stage_data_specs = None
    if recipe_name == HIERARCHICAL_REGRESSION_RECIPE_NAME and stage_cfgs:
        stage_data_specs = _build_stage_data_specs(
            stage_cfgs,
            loader_cfg,
            rep_name,
            objective_terms,
            primary_output=primary_output,
        )
        train_reader = _build_stage_group_reader(stage_data_specs[0]["train_paths"], stage_data_specs[0]["loader_args"])
        test_reader = None
        for spec in reversed(stage_data_specs):
            if spec["test_paths"] is not None:
                test_reader = _build_stage_group_reader(spec["test_paths"], spec["loader_args"])
                break
    else:
        print(f"# training with {len(train_paths)} system(s)")
        train_reader = GroupReader(train_paths, **data_args)

        if test_paths is not None:
            test_paths = _io_utils.load_dirs(test_paths)
            print(f"# testing with {len(test_paths)} system(s)")
            test_reader = GroupReader(test_paths, **data_args)
        else:
            print("# testing with training set")
            test_reader = None

    input_dim = train_reader.ndesc
    if model_args.get("input_dim") is not None and model_args["input_dim"] != input_dim:
        print(f"# `input_dim` in `model_args` does not match data ({input_dim}). Use the one in data.")
    model_args["input_dim"] = input_dim
    model_args = resolve_model_args(model_family, model_args)

    reference_elem_table = None
    if fit_elem and restart is None:
        elem_table = model_args.get("elem_table", None)
        if isinstance(elem_table, str):
            elem_table = load_elem_table(elem_table)
        reference_elem_table = fit_elem_const(train_reader, test_reader, elem_table)
    model_args.pop("elem_table", None)

    if recipe_name == HIERARCHICAL_REGRESSION_RECIPE_NAME:
        objective_args = {
            "hierarchy_levels": deepcopy(hierarchy_levels),
            "terms": deepcopy(objective_terms),
            "descriptor_objective_args": _build_hierarchical_descriptor_objective_args(
                objective, objective_terms
            ),
            "property_scheme": recipe.schema.property_scheme,
            "primary_output": primary_output,
        }
    else:
        objective_args = build_descriptor_property_objective_args(objective)

    model_config = {
        "recipe_name": recipe_name,
        "model_args": model_args,
        "restart": restart,
        "reference_elem_table": reference_elem_table,
        "ckpt_file": runtime_io.get("ckpt_file", "model.pth"),
        "graph_file": runtime_io.get("graph_file"),
        "device": runtime.get("device", "cpu"),
        "preprocess_args": deepcopy(ml.get("preprocess", {})),
        "objective_args": objective_args,
        "train_args": {
            **{
                name: value
                for name, value in {
                    "n_epoch": train_cfg.get("epochs"),
                    "display_epoch": train_cfg.get("display_epoch"),
                    "display_detail_test": train_cfg.get("display_detail_test"),
                    "display_grouped_loss": train_cfg.get("display_natom_loss"),
                }.items()
                if value is not None
            },
            **(
                {
                    key: value
                    for key, value in {
                        "start_lr": train_cfg.get("optimizer", {}).get("lr"),
                        "weight_decay": train_cfg.get("optimizer", {}).get("weight_decay"),
                    }.items()
                    if value is not None
                }
                if isinstance(train_cfg.get("optimizer"), dict)
                else {}
            ),
            **(
                {
                    key: value
                    for key, value in {
                        "decay_steps": train_cfg.get("scheduler", {}).get("decay_steps"),
                        "decay_rate": train_cfg.get("scheduler", {}).get("decay_rate"),
                        "stop_lr": train_cfg.get("scheduler", {}).get("stop_lr"),
                    }.items()
                    if value is not None
                }
                if isinstance(train_cfg.get("scheduler"), dict)
                else {}
            ),
            **(
                {"stage_schedule": deepcopy(train_cfg.get("stage_schedule", []))}
                if train_cfg.get("stage_schedule") is not None
                else {}
            ),
            **({"stage_data_specs": stage_data_specs} if stage_data_specs is not None else {}),
        },
        "fit_elem": fit_elem,
    }

    return train_reader, test_reader, model_config


def run_training_stage(
    train_reader: GroupReader,
    test_reader: Optional[GroupReader],
    model_config: Dict[str, Any],
) -> Tuple[object, Dict[str, Any]]:
    """Run the task-specific training stage through the selected recipe."""
    model_args = model_config["model_args"]
    restart = model_config.get("restart")
    ckpt_file = model_config.get("ckpt_file", "model.pth")
    graph_file = model_config.get("graph_file")
    device = model_config.get("device", "cpu")
    preprocess_args = model_config.get("preprocess_args", {})
    train_args = model_config.get("train_args", {})
    objective_args = model_config.get("objective_args", {})
    reference_elem_table = model_config.get("reference_elem_table")
    recipe = get_recipe(recipe=model_config.get("recipe_name"))

    train_args = {
        **train_args,
        "ckpt_file": ckpt_file,
        "device": device,
    }
    if graph_file is not None:
        train_args["graph_file"] = graph_file

    if restart is not None:
        print(f"# loading model from {restart}")
        model = recipe.create_or_load_model(model_args=model_args, restart=restart)
        if reference_elem_table is None:
            reference_elem_table = load_elem_table_sidecar(restart)
        recipe.fit_restart_elem_const(train_reader, test_reader, reference_elem_table)
    else:
        print("# creating new model")
        model = recipe.create_or_load_model(model_args=model_args, restart=None)

    print("# preprocessing data")
    recipe.preprocess_training_data(model, train_reader, preprocess_args=preprocess_args)

    print("# starting training")
    recipe.train_model(
        model,
        train_reader,
        test_reader=test_reader,
        train_args=train_args,
        objective_args=objective_args,
    )
    if reference_elem_table is not None:
        save_elem_table_sidecar(ckpt_file, reference_elem_table)

    train_stats = {
        "n_epochs": train_args.get("n_epoch", 1000),
        "final_lr": train_args.get("start_lr", 0.001),
        "model_saved": ckpt_file,
    }
    return model, train_stats


def _build_stage_group_reader(paths, loader_args):
    return GroupReader(paths, **loader_args)


