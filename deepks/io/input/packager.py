"""Package merged standard config into task-specific execution parameters."""

from copy import deepcopy

from deepks.interface.registry import HIERARCHICAL_REGRESSION_RECIPE_NAME, get_recipe_name
from deepks.interface.adapters import resolve_hierarchical_model_levels


INTERNAL_PACKED_MARKER = "__internal_packed__"
PAYLOAD_KEYS = {
    "train": "train_param",
    "test": "test_param",
    "scf": "scf_param",
    "stats": "stats_param",
    "iterate": "iterate_param",
}

_OBJECTIVE_SHARED_KEYS = {
    "losses",
    "energy_per_atom",
    "grad_penalty",
    "vd_divide_by_nlocal",
    "vd_masked_loss",
    "vd_masked_S_threshold",
    "vd_masked_H_threshold",
    "vd_masked_width",
    "use_safe_eigh",
    # Used by hierarchical-regression to declare the model's primary output
    # name; carried through so train subtasks can build the correct
    # ObjectiveAdapter.
    "primary_output",
}

_OBJECTIVE_LEGACY_MAP = {
    "energy": ("energy_factor", "energy_loss", None),
    "force": ("force_factor", "force_loss", None),
    "stress": ("stress_factor", "stress_loss", None),
    "orbital": ("orbital_factor", "orbital_loss", None),
    "v_delta": ("v_delta_factor", "v_delta_loss", None),
    "v_delta_r": ("v_delta_r_factor", "v_delta_r_loss", None),
    "phi": ("phi_factor", "phi_loss", "phi_occ"),
    "band": ("band_factor", "band_loss", "band_occ"),
    "bandgap": ("bandgap_factor", "bandgap_loss", "bandgap_occ"),
    "density_m": ("density_m_factor", "density_m_loss", "density_m_occ"),
    "phi_align": ("phi_align_factor", "phi_align_loss", "phi_align_occ"),
    "density": ("density_factor", None, None),
}


def is_packed_config(config):
    return isinstance(config, dict) and bool(config.get(INTERNAL_PACKED_MARKER))


def get_payload_key(task_type):
    if task_type not in PAYLOAD_KEYS:
        raise ValueError(f"Unknown type: {task_type}")
    return PAYLOAD_KEYS[task_type]


def get_packed_payload(config):
    task_type = config.get("type")
    payload_key = get_payload_key(task_type)
    payload = config.get(payload_key)
    if not isinstance(payload, dict):
        raise ValueError(f"Packed config missing '{payload_key}'")
    return deepcopy(payload)


def _normalize_objective_for_packed(objective):
    """Convert legacy objective fields into structured losses for packed configs."""

    if not isinstance(objective, dict):
        return objective

    normalized = {
        key: deepcopy(value)
        for key, value in objective.items()
        if key in _OBJECTIVE_SHARED_KEYS
    }

    losses = objective.get("losses")
    if isinstance(losses, (list, dict)):
        normalized["losses"] = deepcopy(losses)
    terms = objective.get("terms")
    if isinstance(terms, list):
        normalized["terms"] = deepcopy(terms)
    level_losses = objective.get("level_losses")
    if isinstance(level_losses, list):
        normalized["level_losses"] = deepcopy(level_losses)
    if "losses" in normalized or "terms" in normalized or "level_losses" in normalized:
        return normalized

    generated_losses = []
    for loss_name, (factor_key, loss_key, occ_key) in _OBJECTIVE_LEGACY_MAP.items():
        factor = objective.get(factor_key)
        if factor in (None, 0, 0.0, False):
            continue
        item = {"name": loss_name, "weight": deepcopy(factor)}
        if loss_key and objective.get(loss_key) is not None:
            item["loss"] = deepcopy(objective[loss_key])
        if occ_key and objective.get(occ_key) not in (None, 0):
            item["occ"] = deepcopy(objective[occ_key])
        generated_losses.append(item)
    normalized["losses"] = generated_losses
    return normalized


def _resolve_hierarchical_terms(ml):
    objective = ml.get("objective") if isinstance(ml.get("objective"), dict) else {}
    terms = objective.get("terms") if isinstance(objective.get("terms"), list) else []
    if not terms:
        level_losses = objective.get("level_losses") if isinstance(objective.get("level_losses"), list) else []
        if level_losses:
            terms = level_losses
    if not terms:
        terms = [{"name": "hr", "weight": 1.0, "target": {"format": "collected_hr_delta", "name": "l_hr_delta"}}]
    resolved_terms = []
    for item in terms:
        term = deepcopy(item)
        target = term.get("target") if isinstance(term.get("target"), dict) else {}
        if "name" not in target and "hr_name" not in target:
            fmt = target.get("format")
            if fmt == "collected_energy_delta":
                target["name"] = "l_e_delta"
            elif fmt == "collected_hr_delta":
                target["name"] = "l_hr_delta"
        term["target"] = target
        resolved_terms.append(term)
    return resolved_terms


def package_config(config):
    task_type = config.get("type")
    if task_type not in PAYLOAD_KEYS:
        raise ValueError(f"Unknown type: {task_type}")

    if task_type == "train":
        payload = {
            "type": "train",
            "recipe": deepcopy(config.get("recipe")),
            "runtime": {},
            "data": {},
            "physics": {},
            "ml": {},
        }
        runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        for key in ("device", "seed", "dtype", "verbose", "test_log"):
            if key in runtime:
                payload["runtime"][key] = deepcopy(runtime[key])
        if isinstance(runtime.get("io"), dict):
            payload["runtime"]["io"] = deepcopy(runtime["io"])

        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        for key in ("train", "test", "loader", "targets", "stages"):
            if key in data:
                payload["data"][key] = deepcopy(data[key])

        physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
        if isinstance(physics.get("representation"), dict):
            payload["physics"]["representation"] = deepcopy(physics["representation"])
        if isinstance(physics.get("hierarchy"), dict):
            payload["physics"]["hierarchy"] = deepcopy(physics["hierarchy"])

        ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
        for key in ("model", "preprocess", "objective", "train", "checkpoint", "fit_elem"):
            if key in ml:
                if key == "objective":
                    payload["ml"][key] = _normalize_objective_for_packed(ml[key])
                else:
                    payload["ml"][key] = deepcopy(ml[key])

    elif task_type == "test":
        payload = {
            "type": "test",
            "recipe": deepcopy(config.get("recipe")),
            "runtime": {},
            "data": {},
            "physics": {},
            "ml": {},
        }
        runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        for key in ("device", "seed", "dtype", "verbose"):
            if key in runtime:
                payload["runtime"][key] = deepcopy(runtime[key])
        if isinstance(runtime.get("io"), dict):
            payload["runtime"]["io"] = deepcopy(runtime["io"])

        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        for key in ("test", "loader", "targets"):
            if key in data:
                payload["data"][key] = deepcopy(data[key])

        physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
        if isinstance(physics.get("representation"), dict):
            payload["physics"]["representation"] = deepcopy(physics["representation"])

        ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
        for key in ("model", "checkpoint"):
            if key in ml:
                payload["ml"][key] = deepcopy(ml[key])

    elif task_type == "scf":
        payload = {
            "type": "scf",
            "runtime": {},
            "data": {},
            "physics": {},
            "ml": {},
        }
        runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        for key in ("device", "seed", "dtype", "verbose"):
            if key in runtime:
                payload["runtime"][key] = deepcopy(runtime[key])
        if isinstance(runtime.get("scf"), dict):
            payload["runtime"]["scf"] = deepcopy(runtime["scf"])

        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        if "systems" in data:
            payload["data"]["systems"] = deepcopy(data["systems"])

        physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
        if isinstance(physics.get("backend"), dict):
            payload["physics"]["backend"] = deepcopy(physics["backend"])
        if isinstance(physics.get("representation"), dict):
            payload["physics"]["representation"] = deepcopy(physics["representation"])
        if isinstance(physics.get("hierarchy"), dict):
            payload["physics"]["hierarchy"] = deepcopy(physics["hierarchy"])

        ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
        if isinstance(ml.get("model"), dict):
            payload["ml"]["model"] = deepcopy(ml["model"])
        if isinstance(ml.get("checkpoint"), dict):
            payload["ml"]["checkpoint"] = deepcopy(ml["checkpoint"])
        if isinstance(ml.get("objective"), dict):
            # SCF subtasks do not train, but the iterate stats / data
            # collection step needs to know the model's primary output and
            # which supervision terms (e.g. ``collected_hr_delta``) are in
            # play to size the per-stage Hamiltonian data arrays correctly.
            scf_objective = {}
            primary_output = ml["objective"].get("primary_output")
            if primary_output is not None:
                scf_objective["primary_output"] = deepcopy(primary_output)
            terms = ml["objective"].get("terms")
            if isinstance(terms, list) and terms:
                scf_objective["terms"] = deepcopy(terms)
            if scf_objective:
                payload["ml"]["objective"] = scf_objective

    elif task_type == "stats":
        payload = {
            "type": "stats",
            "runtime": {},
            "data": {},
            "physics": {},
        }
        runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        for key in ("verbose",):
            if key in runtime:
                payload["runtime"][key] = deepcopy(runtime[key])
        if isinstance(runtime.get("io"), dict):
            payload["runtime"]["io"] = deepcopy(runtime["io"])

        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        for key in ("systems", "test", "loader"):
            if key in data:
                payload["data"][key] = deepcopy(data[key])

        physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
        if isinstance(physics.get("backend"), dict):
            payload["physics"]["backend"] = deepcopy(physics["backend"])

    else:
        def materialize(value, phase):
            if isinstance(value, dict):
                phase_keys = {key for key in value if key in {"main", "init"}}
                if phase_keys:
                    base = {
                        key: materialize(val, phase)
                        for key, val in value.items()
                        if key not in {"main", "init"}
                    }
                    selected = value.get(phase, value.get("main", value.get("init")))
                    selected = materialize(selected, phase)
                    if isinstance(selected, dict):
                        merged = dict(base)
                        merged.update(selected)
                        return merged
                    return selected
                return {key: materialize(val, phase) for key, val in value.items()}
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and not any(isinstance(item, (dict, list, tuple)) for item in value)
            ):
                return deepcopy(value[0 if phase == "main" else 1])
            if isinstance(value, list):
                return [materialize(item, phase) for item in value]
            if isinstance(value, tuple):
                return tuple(materialize(item, phase) for item in value)
            return deepcopy(value)

        def pack_child(child_type, child_config):
            return package_config({**deepcopy(child_config), "type": child_type})

        main_config = materialize(config, "main")
        init_config = materialize(config, "init")

        runtime = main_config.get("runtime") if isinstance(main_config.get("runtime"), dict) else {}
        data = main_config.get("data") if isinstance(main_config.get("data"), dict) else {}
        physics = main_config.get("physics") if isinstance(main_config.get("physics"), dict) else {}
        ml = main_config.get("ml") if isinstance(main_config.get("ml"), dict) else {}
        iterate_cfg = main_config.get("iterate") if isinstance(main_config.get("iterate"), dict) else {}
        backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
        representation = physics.get("representation") if isinstance(physics.get("representation"), dict) else {}
        rep_params = representation.get("params") if isinstance(representation.get("params"), dict) else {}
        checkpoint = ml.get("checkpoint") if isinstance(ml.get("checkpoint"), dict) else {}

        initial_model = None
        for key in ("file", "restart"):
            value = checkpoint.get(key)
            if value not in (None, False, "", "None", "NONE"):
                initial_model = deepcopy(value)
                break

        scf_soft = str(backend.get("name", "pyscf")).lower()
        proj_basis = rep_params.get("proj_basis")
        child_proj_basis = "proj_basis.npz" if scf_soft == "pyscf" and proj_basis else proj_basis
        use_init = bool(iterate_cfg.get("use_init", False))
        hierarchical_levels = (
            resolve_hierarchical_model_levels(main_config)
            if get_recipe_name(config=config) == HIERARCHICAL_REGRESSION_RECIPE_NAME
            else []
        )
        hierarchical_train_groups = data.get("train", []) if isinstance(data.get("train"), list) else []
        hierarchical_test_groups = data.get("test", []) if isinstance(data.get("test"), list) else []

        main_scf = {
            "recipe": deepcopy(main_config.get("recipe")),
            "runtime": {
                key: deepcopy(runtime[key]) for key in ("device", "seed", "dtype", "verbose") if key in runtime
            },
            "data": {"systems": [] if data.get("train") is None else deepcopy(data.get("train", []))},
            "physics": {
                "backend": deepcopy(backend),
                "representation": deepcopy(representation) if representation else {},
            },
            "ml": {},
        }
        if isinstance(runtime.get("scf"), dict):
            main_scf["runtime"]["scf"] = deepcopy(runtime["scf"])
        if child_proj_basis:
            main_scf["physics"].setdefault("representation", {}).setdefault("params", {})["proj_basis"] = child_proj_basis
        if use_init or initial_model:
            main_scf["ml"]["checkpoint"] = {"file": "model.pth"}
        if hierarchical_levels:
            main_scf["ml"]["model"] = {"args": {"levels": deepcopy(hierarchical_levels)}}
            resolved_terms = _resolve_hierarchical_terms(ml)
            if resolved_terms:
                main_scf.setdefault("ml", {}).setdefault("objective", {})["terms"] = resolved_terms
            if isinstance(ml.get("objective"), dict) and ml["objective"].get("primary_output") is not None:
                main_scf.setdefault("ml", {}).setdefault("objective", {})["primary_output"] = deepcopy(
                    ml["objective"]["primary_output"]
                )
        main_scf["physics"].setdefault("backend", {}).setdefault("output", {})["dump_dir"] = "data_train"

        main_train = {
            "recipe": deepcopy(main_config.get("recipe")),
            "runtime": {
                key: deepcopy(runtime[key]) for key in ("device", "seed", "dtype", "verbose", "test_log") if key in runtime
            },
            "data": {
                "train": "data_train/*",
                "loader": deepcopy(data.get("loader", {})),
                "targets": deepcopy(data.get("targets", {})),
            },
            "physics": {},
            "ml": {
                key: deepcopy(ml[key])
                for key in ("model", "preprocess", "objective", "train", "fit_elem")
                if key in ml
            },
        }
        if hierarchical_levels:
            main_train["data"]["stages"] = [
                {
                    "level": int(level_cfg["level"]),
                    "name": deepcopy(level_cfg.get("name")),
                    "train": [f"../00.scf/level.{int(level_cfg['level']):02d}/data_train/*"],
                    "test": [f"../00.scf/level.{int(level_cfg['level']):02d}/data_test/*"] if int(level_cfg["level"]) < len(hierarchical_test_groups) and hierarchical_test_groups[int(level_cfg["level"])] is not None else None,
                }
                for level_cfg in hierarchical_levels
            ]
            resolved_terms = _resolve_hierarchical_terms(ml)
            if resolved_terms:
                main_train.setdefault("ml", {}).setdefault("objective", {})["terms"] = resolved_terms
        if data.get("test") is not None:
            main_train["data"]["test"] = "data_test/*"
        if isinstance(runtime.get("io"), dict):
            main_train["runtime"]["io"] = deepcopy(runtime["io"])
        if child_proj_basis:
            main_train["physics"].setdefault("representation", {}).setdefault("params", {})["proj_basis"] = child_proj_basis
        main_train["ml"]["checkpoint"] = {"restart": "model.pth" if (use_init or initial_model) else None}
        main_train["runtime"].setdefault("io", {})["ckpt_file"] = "model.pth"

        payload = {
            "type": "iterate",
            "recipe": deepcopy(config.get("recipe")),
            "runtime": {
                "workdir": runtime.get("workdir", "."),
                "share_folder": runtime.get("share_folder", "share"),
                "scf": {"execute": deepcopy(runtime.get("scf", {}).get("execute", {}))} if isinstance(runtime.get("scf"), dict) else {},
                "train": {"execute": deepcopy(runtime.get("train", {}).get("execute", {}))} if isinstance(runtime.get("train"), dict) else {},
            },
            "data": {
                "train": [] if data.get("train") is None else deepcopy(data.get("train", [])),
                "test": deepcopy(data.get("test")),
            },
            "physics": {
                "backend": deepcopy(backend) if backend else {"name": scf_soft},
                "representation": {"params": {"proj_basis": proj_basis}} if proj_basis is not None else {},
            },
            "ml": {
                "model": deepcopy(ml.get("model", {})) if isinstance(ml.get("model"), dict) else {},
                "checkpoint": {"file": initial_model} if initial_model else {},
            },
            "iterate": {
                "n_iter": iterate_cfg.get("n_iter", 0),
                "use_init": use_init,
                "cleanup": iterate_cfg.get("cleanup", False),
                "strict": iterate_cfg.get("strict", True),
                "tasks": {
                    "main": {
                        "scf": pack_child("scf", main_scf),
                        "train": pack_child("train", main_train),
                    }
                },
            },
        }
        if use_init:
            init_runtime = init_config.get("runtime") if isinstance(init_config.get("runtime"), dict) else {}
            init_ml = init_config.get("ml") if isinstance(init_config.get("ml"), dict) else {}
            init_physics = init_config.get("physics") if isinstance(init_config.get("physics"), dict) else {}
            init_backend = init_physics.get("backend") if isinstance(init_physics.get("backend"), dict) else {}
            init_representation = init_physics.get("representation") if isinstance(init_physics.get("representation"), dict) else {}

            init_scf = deepcopy(main_scf)
            init_scf["runtime"]["scf"] = deepcopy(init_runtime.get("scf", runtime.get("scf", {}))) if isinstance(init_runtime.get("scf"), dict) else deepcopy(runtime.get("scf", {}))
            init_scf["physics"]["backend"] = deepcopy(init_backend if init_backend else backend)
            init_scf["physics"]["representation"] = deepcopy(init_representation if init_representation else representation)
            if child_proj_basis:
                init_scf["physics"].setdefault("representation", {}).setdefault("params", {})["proj_basis"] = child_proj_basis
            if initial_model:
                init_scf["ml"]["checkpoint"] = {"file": "model.pth"}
            if hierarchical_levels:
                init_scf["ml"]["model"] = {"args": {"levels": deepcopy(hierarchical_levels)}}
                resolved_terms = _resolve_hierarchical_terms(init_ml)
                if resolved_terms:
                    init_scf.setdefault("ml", {}).setdefault("objective", {})["terms"] = resolved_terms
                if isinstance(init_ml.get("objective"), dict) and init_ml["objective"].get("primary_output") is not None:
                    init_scf.setdefault("ml", {}).setdefault("objective", {})["primary_output"] = deepcopy(
                        init_ml["objective"]["primary_output"]
                    )
            else:
                init_scf["ml"] = {}

            init_train = deepcopy(main_train)
            init_train["runtime"]["train"] = {"execute": deepcopy(init_runtime.get("train", runtime.get("train", {})).get("execute", {}))} if isinstance(init_runtime.get("train"), dict) else deepcopy(main_train["runtime"].get("train", {}))
            for key in ("model", "preprocess", "objective", "train", "fit_elem"):
                if key in init_ml:
                    init_train["ml"][key] = deepcopy(init_ml[key])
            init_train["ml"]["checkpoint"] = {"restart": "model.pth" if initial_model else None}

            payload["iterate"]["tasks"]["init"] = {
                "scf": pack_child("scf", init_scf),
                "train": pack_child("train", init_train),
            }

    return {
        INTERNAL_PACKED_MARKER: True,
        "type": task_type,
        get_payload_key(task_type): payload,
    }
