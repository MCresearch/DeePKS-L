"""Configuration validator for DeePKS."""

import os


def validate_config(config, command=None):
    """Validate configuration.

    Args:
        config: Configuration dictionary
        command: Command name ('scf', 'train', 'test', 'iterate')

    Raises:
        ValueError: If configuration is invalid
        TypeError: If parameter types are incorrect
    """
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a dictionary, got {type(config)}")

    # Validate based on command
    if command == 'scf':
        _validate_scf_config(config)
    elif command == 'train':
        _validate_train_config(config)
    elif command == 'test':
        _validate_test_config(config)
    elif command == 'iterate':
        _validate_iterate_config(config)


def _validate_scf_config(config):
    """Validate SCF configuration."""
    # Check required parameters
    if 'systems' not in config:
        raise ValueError("SCF configuration requires 'systems' parameter")

    # Validate systems
    systems = config['systems']
    if not isinstance(systems, (list, str)):
        raise TypeError(f"'systems' must be list or string, got {type(systems)}")

    # Validate scf_soft
    scf_soft = config.get('scf_soft', 'pyscf')
    if scf_soft.lower() not in ['pyscf', 'abacus']:
        raise ValueError(f"Invalid scf_soft: {scf_soft}. Must be 'pyscf' or 'abacus'")

    # Validate backend-specific parameters
    if scf_soft.lower() == 'pyscf':
        _validate_pyscf_params(config)
    elif scf_soft.lower() == 'abacus':
        _validate_abacus_params(config)


def _validate_pyscf_params(config):
    """Validate PySCF-specific parameters."""
    # Validate basis
    basis = config.get('basis')
    if basis is not None and not isinstance(basis, str):
        raise TypeError(f"'basis' must be string, got {type(basis)}")

    # Validate mol_args
    mol_args = config.get('mol_args')
    if mol_args is not None and not isinstance(mol_args, dict):
        raise TypeError(f"'mol_args' must be dict, got {type(mol_args)}")

    # Validate scf_args
    scf_args = config.get('scf_args')
    if scf_args is not None and not isinstance(scf_args, dict):
        raise TypeError(f"'scf_args' must be dict, got {type(scf_args)}")


def _validate_abacus_params(config):
    """Validate ABACUS-specific parameters."""
    # Check required ABACUS parameters
    required = ['orb_files', 'pp_files']
    for param in required:
        if param not in config:
            raise ValueError(f"ABACUS backend requires '{param}' parameter")
        if not isinstance(config[param], list):
            raise TypeError(f"'{param}' must be list, got {type(config[param])}")

    # Validate abacus_path
    abacus_path = config.get('abacus_path')
    if abacus_path is not None and not isinstance(abacus_path, str):
        raise TypeError(f"'abacus_path' must be string, got {type(abacus_path)}")

    # Validate numerical parameters
    numerical_params = ['ecutwfc', 'scf_thr', 'lattice_constant']
    for param in numerical_params:
        value = config.get(param)
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{param}' must be number, got {type(value)}")
            if value <= 0:
                raise ValueError(f"'{param}' must be positive, got {value}")


def _validate_train_config(config):
    """Validate training configuration."""
    # Check required parameters
    if 'systems_train' not in config:
        raise ValueError("Training configuration requires 'systems_train' parameter")

    # Validate systems_train
    systems_train = config['systems_train']
    if not isinstance(systems_train, (list, str)):
        raise TypeError(f"'systems_train' must be list or string, got {type(systems_train)}")

    # Validate model_args
    model_args = config.get('model_args')
    if model_args is not None and not isinstance(model_args, dict):
        raise TypeError(f"'model_args' must be dict, got {type(model_args)}")

    # Validate train_args
    train_args = config.get('train_args')
    if train_args is not None and not isinstance(train_args, dict):
        raise TypeError(f"'train_args' must be dict, got {type(train_args)}")


def _validate_test_config(config):
    """Validate test configuration."""
    # Check required parameters
    if 'systems_test' not in config:
        raise ValueError("Test configuration requires 'systems_test' parameter")

    # Validate systems_test
    systems_test = config['systems_test']
    if not isinstance(systems_test, (list, str)):
        raise TypeError(f"'systems_test' must be list or string, got {type(systems_test)}")

    # Validate model_file
    if 'model_file' not in config:
        raise ValueError("Test configuration requires 'model_file' parameter")


def _validate_iterate_config(config):
    """Validate iterate configuration."""
    # Check required parameters
    if 'systems_train' not in config:
        raise ValueError("Iterate configuration requires 'systems_train' parameter")

    # Validate n_iter
    n_iter = config.get('n_iter')
    if n_iter is not None:
        if not isinstance(n_iter, int):
            raise TypeError(f"'n_iter' must be integer, got {type(n_iter)}")
        if n_iter < 0:
            raise ValueError(f"'n_iter' must be non-negative, got {n_iter}")

    # Validate scf_soft
    scf_soft = config.get('scf_soft', 'pyscf')
    if scf_soft.lower() not in ['pyscf', 'abacus']:
        raise ValueError(f"Invalid scf_soft: {scf_soft}. Must be 'pyscf' or 'abacus'")


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
