"""Configuration validator for DeePKS."""

import os
from deepks.interface.registry import HIERARCHICAL_REGRESSION_RECIPE_NAME, get_recipe_name


VALID_TASK_TYPES = ("train", "test", "scf", "stats", "iterate")
VALID_SCF_BACKENDS = ("pyscf", "abacus")


def validate_config(config, command=None):
    """Validate configuration.

    Args:
        config: Configuration dictionary
        command: Command name ('scf', 'train', 'test', 'stats', 'iterate')

    Raises:
        ValueError: If configuration is invalid
        TypeError: If parameter types are incorrect
    """
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a dictionary, got {type(config)}")

    if command is None:
        command = config.get('type')

    if command not in VALID_TASK_TYPES:
        raise ValueError(
            f"Unknown type: {command}. Valid types: {', '.join(sorted(VALID_TASK_TYPES))}"
        )

    if command == 'scf':
        _validate_scf_config(config)
    elif command == 'train':
        _validate_train_config(config)
    elif command == 'test':
        _validate_test_config(config)
    elif command == 'stats':
        _validate_stats_config(config)
    elif command == 'iterate':
        _validate_iterate_config(config)


def _validate_scf_backend_name(config):
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_name = backend.get("name")
    if backend_name is None:
        raise ValueError("SCF backend must be explicitly provided as physics.backend.name")
    if backend_name not in VALID_SCF_BACKENDS:
        raise ValueError(
            f"Invalid physics.backend.name: {backend_name}. Must be 'pyscf' or 'abacus'"
        )
    return backend_name


def _validate_path_list(value, param_name):
    if not isinstance(value, (list, str)):
        raise TypeError(f"'{param_name}' must be list or string, got {type(value)}")


def _validate_pyscf_block(config):
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_input = backend.get("input", {}) if isinstance(backend.get("input"), dict) else {}
    basis = backend_input.get('basis')
    if basis is not None and not isinstance(basis, str):
        raise TypeError(f"'basis' must be string, got {type(basis)}")

    proj_basis = backend_input.get('proj_basis')
    if proj_basis is not None and not isinstance(proj_basis, (str, list, tuple, dict)):
        raise TypeError(f"'proj_basis' must be str/list/tuple/dict, got {type(proj_basis)}")

    mol_args = backend_input.get('mol_args')
    if mol_args is not None and not isinstance(mol_args, dict):
        raise TypeError(f"'mol_args' must be dict, got {type(mol_args)}")

    scf_args = backend_input.get('scf_args')
    if scf_args is not None and not isinstance(scf_args, dict):
        raise TypeError(f"'scf_args' must be dict, got {type(scf_args)}")


def _validate_abacus_block(config, *, require_files=True):
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_input = backend.get("input", {}) if isinstance(backend.get("input"), dict) else {}

    for param in ('orb_files', 'pp_files'):
        value = backend_input.get(param)
        if value is None:
            if require_files:
                raise ValueError(f"physics.backend.input requires '{param}' for ABACUS")
            continue
        if not isinstance(value, list):
            raise TypeError(f"'{param}' must be list, got {type(value)}")

    numerical_params = ['ecutwfc', 'scf_thr', 'lattice_constant']
    for param in numerical_params:
        value = backend_input.get(param)
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{param}' must be number, got {type(value)}")
            if value <= 0:
                raise ValueError(f"'{param}' must be positive, got {value}")


def _validate_scf_config(config):
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    systems = [] if data.get("systems") is None else data.get("systems", [])
    _validate_path_list(systems, 'data.systems')
    if not systems:
        raise ValueError("Scf configuration requires 'data.systems' parameter")
    backend_name = _validate_scf_backend_name(config)
    if backend_name == 'pyscf':
        _validate_pyscf_block(config)
    else:
        _validate_abacus_block(config)


def _validate_train_config(config):
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    train_paths = [] if data.get("train") is None else data.get("train", [])
    _validate_path_list(train_paths, 'data.train')
    if not train_paths:
        raise ValueError("Train configuration requires 'data.train' parameter")

def _validate_test_config(config):
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    test_paths = data.get("test")
    _validate_path_list(test_paths, 'data.test')
    if test_paths is None:
        raise ValueError("Test configuration requires 'data.test' parameter")


def _validate_stats_config(config):
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    systems = [] if data.get("systems") is None else data.get("systems", [])
    _validate_path_list(systems, 'data.systems')
    if not systems:
        raise ValueError("Stats configuration requires 'data.systems' parameter")
    backend_name = _validate_scf_backend_name(config)
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_input = backend.get("input", {}) if isinstance(backend.get("input"), dict) else {}
    if backend_name == 'pyscf':
        if backend_input:
            _validate_pyscf_block(config)
    else:
        if backend_input:
            _validate_abacus_block(config)


def _validate_iterate_config(config):
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    recipe_name = get_recipe_name(config=config)
    iterate_cfg = config.get('iterate') if isinstance(config.get('iterate'), dict) else {}

    if recipe_name == HIERARCHICAL_REGRESSION_RECIPE_NAME:
        _validate_hierarchical_iterate_config(config)
    else:
        train_paths = [] if data.get("train") is None else data.get("train", [])
        _validate_path_list(train_paths, 'data.train')
        if not train_paths:
            raise ValueError("Iterate configuration requires 'data.train' parameter")

    n_iter = iterate_cfg.get('n_iter')
    if n_iter is not None:
        if not isinstance(n_iter, int):
            raise TypeError(f"'n_iter' must be integer, got {type(n_iter)}")
        if n_iter < 0:
            raise ValueError(f"'n_iter' must be non-negative, got {n_iter}")

    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    scf_profiles = backend.get("profiles") if isinstance(backend.get("profiles"), list) else []
    backend_name = _validate_scf_backend_name(config)
    if backend_name == 'pyscf':
        _validate_pyscf_block(config)
    else:
        # Per-basis SCF profiles supply orb_files individually, so the global
        # physics.backend.input.orb_files requirement is relaxed when present.
        require_files = recipe_name != HIERARCHICAL_REGRESSION_RECIPE_NAME and not scf_profiles
        _validate_abacus_block(config, require_files=require_files)
        # Profile-driven per-basis SCF (non-hierarchical): each profile must
        # carry its own orb_files (or an input_template). The hierarchical
        # recipe validates its own profiles in _validate_hierarchical_iterate_config.
        if recipe_name != HIERARCHICAL_REGRESSION_RECIPE_NAME:
            for i, prof in enumerate(scf_profiles):
                prof_input = prof.get("input", {}) if isinstance(prof, dict) else {}
                has_orb = isinstance(prof_input, dict) and prof_input.get("orb_files")
                has_template = isinstance(prof, dict) and prof.get("input_template") is not None
                if not (has_orb or has_template):
                    raise ValueError(
                        f"physics.backend.profiles[{i}] requires 'input.orb_files' "
                        "(or 'input_template') for per-basis SCF"
                    )


def _validate_hierarchical_iterate_config(config):
    from deepks.workflows.iterate.support.task_params import resolve_hierarchical_iterate_levels
    from deepks.interface.adapters import resolve_hierarchical_model_levels

    hierarchy_levels = resolve_hierarchical_model_levels(config)
    if not hierarchy_levels:
        raise ValueError("hierarchical-regression iterate configuration requires 'ml.model.args.levels'")

    resolved_levels = resolve_hierarchical_iterate_levels(config, require_complete=True)
    if not resolved_levels:
        raise ValueError("hierarchical-regression iterate configuration requires level-specific iterate metadata")

    for level_meta in resolved_levels:
        level_index = level_meta["level"]
        systems_cfg = level_meta["systems"]
        train_paths = [] if systems_cfg.get("train_paths") is None else systems_cfg.get("train_paths", [])
        _validate_path_list(train_paths, f"data.train[{level_index}]")
        if not train_paths:
            raise ValueError(f"data.train[{level_index}] is required")
        test_paths = systems_cfg.get("test_paths")
        if test_paths is not None:
            _validate_path_list(test_paths, f"data.test[{level_index}]")

        profile_cfg = level_meta.get("profile", {}) if isinstance(level_meta.get("profile"), dict) else {}
        has_template = profile_cfg.get("input_template") is not None
        merged_backend_input = level_meta.get("merged_backend_input", {})
        has_effective_backend_input = isinstance(merged_backend_input, dict) and bool(merged_backend_input)
        if not (has_template or has_effective_backend_input):
            raise ValueError(
                f"Level {level_index} requires either physics.backend.profiles[{level_index}].input_template "
                "or an effective backend input (global physics.backend.input merged with profile input)"
            )


def validate_file_exists(path, param_name):
    """Validate that a file exists.

    Args:
        path: File path
        param_name: Parameter name for error message

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{param_name}: File not found: {path}")
