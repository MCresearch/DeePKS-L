"""Iterate child-step parameter helpers."""

from copy import deepcopy
from typing import Any, Dict, List

from deepks.io.input.packager import get_packed_payload, get_payload_key, is_packed_config
from deepks.io.utils import load_yaml
from deepks.interface.adapters import resolve_hierarchical_model_levels
from deepks.interface.registry import HIERARCHICAL_REGRESSION_RECIPE_NAME, get_recipe_name

_ABACUS_RUNTIME_KEYS = {"abacus_path", "run_cmd", "python"}


def _merge_nested(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _merge_nested(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(override)


def _objective_requests_hr(objective_cfg: Dict[str, Any]) -> bool:
    terms = objective_cfg.get("terms") if isinstance(objective_cfg.get("terms"), list) else []
    for term in terms:
        target = term.get("target") if isinstance(term.get("target"), dict) else {}
        if target.get("format") == "collected_hr_delta":
            return True
    return False


def _resolve_hr_target_shape_from_level(level_cfg: Dict[str, Any], objective_cfg: Dict[str, Any]):
    primary_output = str(objective_cfg.get("primary_output", "energy")).strip().lower()
    if primary_output != "hamiltonian" and not _objective_requests_hr(objective_cfg):
        return None
    target_shape = level_cfg.get("target_shape")
    if isinstance(target_shape, (list, tuple)) and len(target_shape) == 5:
        return list(target_shape)
    return None


def _resolve_objective_from_iterate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
    if isinstance(ml.get("objective"), dict):
        return deepcopy(ml["objective"])
    iterate_cfg = config.get("iterate") if isinstance(config.get("iterate"), dict) else {}
    tasks = iterate_cfg.get("tasks") if isinstance(iterate_cfg.get("tasks"), dict) else {}
    main_task = tasks.get("main") if isinstance(tasks.get("main"), dict) else {}
    train_task = main_task.get("train") if isinstance(main_task.get("train"), dict) else {}
    train_param = train_task.get("train_param") if isinstance(train_task.get("train_param"), dict) else {}
    train_ml = train_param.get("ml") if isinstance(train_param.get("ml"), dict) else {}
    if isinstance(train_ml.get("objective"), dict):
        return deepcopy(train_ml["objective"])
    return {}


def build_abacus_iterate_scf_kwargs(scf_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ABACUS step kwargs from a packed SCF child task config."""
    scf_config_view = get_packed_payload(scf_config) if is_packed_config(scf_config) else (scf_config or {})
    physics = scf_config_view.get("physics") if isinstance(scf_config_view.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    model = scf_config_view.get("ml") if isinstance(scf_config_view.get("ml"), dict) else {}
    model_cfg = model.get("model") if isinstance(model.get("model"), dict) else {}
    model_args = model_cfg.get("args") if isinstance(model_cfg.get("args"), dict) else {}
    objective_cfg = model.get("objective") if isinstance(model.get("objective"), dict) else {}
    model_levels = resolve_hierarchical_model_levels(model_args)
    backend_input = dict(backend.get("input", {})) if isinstance(backend.get("input"), dict) else {}
    runtime = scf_config_view.get("runtime") if isinstance(scf_config_view.get("runtime"), dict) else {}
    scf_runtime = runtime.get("scf") if isinstance(runtime.get("scf"), dict) else {}
    command_cfg = dict(scf_runtime.get("command", {})) if isinstance(scf_runtime.get("command"), dict) else {}

    passthrough = {
        key: value
        for key, value in backend_input.items()
        if key not in {"orb_files", "pp_files", "proj_file"} | _ABACUS_RUNTIME_KEYS
    }
    if model_levels:
        hr_target_shape = _resolve_hr_target_shape_from_level(model_levels[0], objective_cfg)
        if hr_target_shape is not None:
            passthrough["target_shape"] = deepcopy(hr_target_shape)

    return {
        "orb_files": backend_input.get("orb_files", []),
        "pp_files": backend_input.get("pp_files", []),
        "proj_file": backend_input.get("proj_file", []),
        "run_cmd": command_cfg.get("run_cmd", "mpirun"),
        "abacus_path": command_cfg.get("abacus_path", "abacus"),
        "backend_kwargs": passthrough,
    }


def materialize_hierarchical_level_scf_config(base_scf_config: Dict[str, Any], level_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize the effective child SCF config for a hierarchy level."""

    config = deepcopy(base_scf_config)
    profile_cfg = level_meta.get("profile", {}) if isinstance(level_meta.get("profile"), dict) else {}
    template_path = profile_cfg.get("input_template")
    if template_path is not None:
        template = load_yaml(template_path)
        if not isinstance(template, dict):
            raise TypeError(f"SCF input template must contain a mapping, got {type(template)!r}")
        config = _merge_nested(config, template)

    if is_packed_config(config):
        payload_key = get_payload_key(config["type"])
        payload = config[payload_key]
    else:
        payload = config
    physics = payload.setdefault("physics", {})
    backend = physics.setdefault("backend", {})
    backend["input"] = deepcopy(level_meta.get("merged_backend_input", {}))
    ml = payload.setdefault("ml", {})
    model_cfg = ml.setdefault("model", {})
    model_args = model_cfg.setdefault("args", {})
    if "model_level" in level_meta:
        model_args["levels"] = [deepcopy(level_meta["model_level"])]
    if isinstance(level_meta.get("objective"), dict):
        ml["objective"] = deepcopy(level_meta["objective"])

    runtime_command = profile_cfg.get("runtime_command")
    if isinstance(runtime_command, dict):
        runtime = payload.setdefault("runtime", {})
        scf_runtime = runtime.setdefault("scf", {})
        command_cfg = scf_runtime.get("command", {}) if isinstance(scf_runtime.get("command"), dict) else {}
        scf_runtime["command"] = _merge_nested(command_cfg, runtime_command)

    return config


def resolve_hierarchical_iterate_levels(config: Dict[str, Any], *, require_complete: bool = False) -> List[Dict[str, Any]]:
    """Resolve level-indexed iterate metadata for the hierarchical Hamiltonian recipe.

    This helper only interprets configuration; it does not materialize any workflow
    objects yet. Old recipes always return an empty list.
    """

    if not isinstance(config, dict) or get_recipe_name(config=config) != HIERARCHICAL_REGRESSION_RECIPE_NAME:
        return []

    raw_levels = resolve_hierarchical_model_levels(config)
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    global_backend_input = dict(backend.get("input", {})) if isinstance(backend.get("input"), dict) else {}
    raw_profiles = backend.get("profiles") if isinstance(backend.get("profiles"), list) else []
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    train_groups = data_cfg.get("train") if isinstance(data_cfg.get("train"), list) else []
    test_groups = data_cfg.get("test") if isinstance(data_cfg.get("test"), list) else []
    ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
    objective_cfg = _resolve_objective_from_iterate_config(config)
    train_cfg = ml.get("train") if isinstance(ml.get("train"), dict) else {}
    stage_schedule = train_cfg.get("stage_schedule") if isinstance(train_cfg.get("stage_schedule"), list) else []

    resolved: Dict[int, Dict[str, Any]] = {}

    for index, raw_level in enumerate(raw_levels):
        level_index = int(raw_level.get("level", index))
        if level_index in resolved:
            raise ValueError(f"Duplicate hierarchy level index: {level_index}")
        resolved[level_index] = {
            "level": level_index,
            "model_level": deepcopy(raw_level),
            "objective": deepcopy(objective_cfg),
        }

    referenced_levels = {int(stage["level"]) for stage in stage_schedule if isinstance(stage, dict) and "level" in stage}
    if not referenced_levels and raw_levels:
        referenced_levels = set(resolved)

    def _normalize_group(value, label):
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return deepcopy(value)
        raise TypeError(f"{label} must be a string or list, got {type(value)!r}")

    for index, level_meta in resolved.items():
        if index >= len(train_groups):
            continue
        systems = {"level": index, "train_paths": _normalize_group(train_groups[index], f"data.train[{index}]")}
        if index < len(test_groups) and test_groups[index] is not None:
            systems["test_paths"] = _normalize_group(test_groups[index], f"data.test[{index}]")
        level_meta["systems"] = systems
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, dict):
            raise TypeError(f"physics.backend.profiles[{index}] must be a dict, got {type(raw_profile)!r}")
        if index not in resolved:
            continue
        profile = deepcopy(raw_profile)
        profile.setdefault("level", index)
        if "name" not in profile:
            profile["name"] = resolved[index]["model_level"].get("name", f"level_{index}")
        resolved[index]["profile"] = profile

    for level_index, level_meta in resolved.items():
        profile_cfg = level_meta.get("profile", {})
        level_name = level_meta["model_level"].get("name")
        profile_name = profile_cfg.get("name") if isinstance(profile_cfg, dict) else None
        if level_name is not None and profile_name is not None and level_name != profile_name:
            raise ValueError(
                f"Level/profile name mismatch at index {level_index}: ml.model.args.levels[{level_index}].name="
                f"{level_name!r} != physics.backend.profiles[{level_index}].name={profile_name!r}"
            )
        override_backend_input = (
            profile_cfg.get("input")
            if isinstance(profile_cfg.get("input"), dict)
            else {}
        )
        level_meta["merged_backend_input"] = _merge_nested(global_backend_input, override_backend_input)

    if require_complete:
        if not raw_levels:
            raise ValueError("hierarchical-regression iterate config requires ml.model.args.levels")
        if not train_groups:
            raise ValueError("hierarchical-regression iterate config requires data.train as a per-level list")
        for level_index in sorted(referenced_levels):
            level_meta = resolved.get(level_index)
            if level_meta is None or "model_level" not in level_meta:
                raise ValueError(f"Missing ml.model.args.levels entry for level {level_index}")
            if "systems" not in level_meta:
                raise ValueError(f"Missing data.train entry for level {level_index}")
            profile_cfg = level_meta.get("profile", {}) if isinstance(level_meta.get("profile"), dict) else {}
            has_template = profile_cfg.get("input_template") is not None
            merged_backend_input = level_meta.get("merged_backend_input", {})
            has_effective_backend_input = isinstance(merged_backend_input, dict) and bool(merged_backend_input)
            if not (has_template or has_effective_backend_input):
                raise ValueError(
                    f"Level {level_index} requires either physics.backend.profiles[{level_index}].input_template "
                    "or an effective backend input (global physics.backend.input merged with profile input)"
                )

    return [resolved[level] for level in sorted(resolved)]
