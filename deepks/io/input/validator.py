"""Configuration validator for DeePKS."""

import os

from .config import (
    ABACUS_BACKEND_KEYS,
    PYSCF_BACKEND_KEYS,
    REQUIRED_TASK_KEYS,
    VALID_SCF_BACKENDS,
    VALID_TASK_TYPES,
    get_backend_block,
    infer_scf_backend,
)


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

    for key in REQUIRED_TASK_KEYS.get(command, ()):
        if key not in config:
            raise ValueError(f"{command.capitalize()} configuration requires '{key}' parameter")

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
    scf_soft = infer_scf_backend(config)
    if scf_soft not in VALID_SCF_BACKENDS:
        raise ValueError(f"Invalid scf_soft: {scf_soft}. Must be 'pyscf' or 'abacus'")
    return scf_soft


def _validate_path_list(value, param_name):
    if not isinstance(value, (list, str)):
        raise TypeError(f"'{param_name}' must be list or string, got {type(value)}")


def _validate_pyscf_block(config):
    scf_pyscf = get_backend_block(config, 'pyscf')
    basis = config.get('basis', scf_pyscf.get('basis'))
    if basis is not None and not isinstance(basis, str):
        raise TypeError(f"'basis' must be string, got {type(basis)}")

    proj_basis = config.get('proj_basis', scf_pyscf.get('proj_basis'))
    if proj_basis is not None and not isinstance(proj_basis, (str, list, tuple, dict)):
        raise TypeError(f"'proj_basis' must be str/list/tuple/dict, got {type(proj_basis)}")

    mol_args = config.get('mol_args', scf_pyscf.get('mol_args'))
    if mol_args is not None and not isinstance(mol_args, dict):
        raise TypeError(f"'mol_args' must be dict, got {type(mol_args)}")

    scf_args = config.get('scf_args', scf_pyscf.get('scf_args'))
    if scf_args is not None and not isinstance(scf_args, dict):
        raise TypeError(f"'scf_args' must be dict, got {type(scf_args)}")


def _validate_abacus_block(config, block_name='scf_abacus'):
    backend = 'abacus'
    scf_abacus = get_backend_block(config, backend, init=(block_name == 'init_scf_abacus'))

    for param in ('orb_files', 'pp_files'):
        value = config.get(param, scf_abacus.get(param))
        if value is None:
            raise ValueError(f"ABACUS backend requires '{param}' parameter")
        if not isinstance(value, list):
            raise TypeError(f"'{param}' must be list, got {type(value)}")

    abacus_path = config.get('abacus_path', scf_abacus.get('abacus_path'))
    if abacus_path is not None and not isinstance(abacus_path, str):
        raise TypeError(f"'abacus_path' must be string, got {type(abacus_path)}")

    numerical_params = ['ecutwfc', 'scf_thr', 'lattice_constant']
    for param in numerical_params:
        value = config.get(param, scf_abacus.get(param))
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{param}' must be number, got {type(value)}")
            if value <= 0:
                raise ValueError(f"'{param}' must be positive, got {value}")


def _validate_scf_config(config):
    _validate_path_list(config['systems'], 'systems')
    scf_soft = _validate_scf_backend_name(config)
    if scf_soft == 'pyscf':
        _validate_pyscf_block(config)
    else:
        _validate_abacus_block(config)


def _validate_train_config(config):
    _validate_path_list(config['systems_train'], 'systems_train')

    model_args = config.get('model_args')
    if model_args is not None and not isinstance(model_args, dict):
        raise TypeError(f"'model_args' must be dict, got {type(model_args)}")

    train_args = config.get('train_args')
    if train_args is not None and not isinstance(train_args, dict):
        raise TypeError(f"'train_args' must be dict, got {type(train_args)}")


def _validate_test_config(config):
    _validate_path_list(config['systems_test'], 'systems_test')


def _validate_stats_config(config):
    _validate_path_list(config['systems'], 'systems')
    scf_soft = _validate_scf_backend_name(config)
    if scf_soft == 'pyscf':
        _validate_pyscf_block(config)
    else:
        _validate_abacus_block(config)


def _validate_iterate_config(config):
    _validate_path_list(config['systems_train'], 'systems_train')

    n_iter = config.get('n_iter')
    if n_iter is not None:
        if not isinstance(n_iter, int):
            raise TypeError(f"'n_iter' must be integer, got {type(n_iter)}")
        if n_iter < 0:
            raise ValueError(f"'n_iter' must be non-negative, got {n_iter}")

    scf_soft = _validate_scf_backend_name(config)
    if scf_soft == 'pyscf':
        _validate_pyscf_block(config)
    else:
        _validate_abacus_block(config)
        init_scf_abacus = config.get('init_scf_abacus')
        if init_scf_abacus is not None and not isinstance(init_scf_abacus, dict):
            raise TypeError(f"'init_scf_abacus' must be dict, got {type(init_scf_abacus)}")
        if isinstance(init_scf_abacus, dict):
            init_config = dict(config)
            init_config.update(init_scf_abacus)
            init_config['init_scf_abacus'] = init_scf_abacus
            _validate_abacus_block(init_config, 'init_scf_abacus')


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
