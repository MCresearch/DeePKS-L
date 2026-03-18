"""Configuration merger for DeePKS."""


def merge_configs(base, override):
    """Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        dict: Merged configuration
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override

    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def apply_parameter_inheritance(config):
    """Apply parameter inheritance rules.

    Init parameters inherit from non-init parameters:
    - init_scf inherits from scf_input
    - init_train inherits from train_input
    - init_scf_machine inherits from scf_machine
    - init_train_machine inherits from train_machine

    Args:
        config: Configuration dictionary

    Returns:
        dict: Configuration with inheritance applied
    """
    config = config.copy()

    # SCF input inheritance
    if 'scf_input' in config and 'init_scf' not in config:
        config['init_scf'] = config['scf_input'].copy() if isinstance(config['scf_input'], dict) else config['scf_input']
    elif 'scf_input' in config and 'init_scf' in config:
        if isinstance(config['scf_input'], dict) and isinstance(config['init_scf'], dict):
            config['init_scf'] = merge_configs(config['scf_input'], config['init_scf'])

    # Train input inheritance
    if 'train_input' in config and 'init_train' not in config:
        config['init_train'] = config['train_input'].copy() if isinstance(config['train_input'], dict) else config['train_input']
    elif 'train_input' in config and 'init_train' in config:
        if isinstance(config['train_input'], dict) and isinstance(config['init_train'], dict):
            config['init_train'] = merge_configs(config['train_input'], config['init_train'])

    # SCF machine inheritance
    if 'scf_machine' in config and 'init_scf_machine' not in config:
        config['init_scf_machine'] = config['scf_machine'].copy() if isinstance(config['scf_machine'], dict) else config['scf_machine']
    elif 'scf_machine' in config and 'init_scf_machine' in config:
        if isinstance(config['scf_machine'], dict) and isinstance(config['init_scf_machine'], dict):
            config['init_scf_machine'] = merge_configs(config['scf_machine'], config['init_scf_machine'])

    # Train machine inheritance
    if 'train_machine' in config and 'init_train_machine' not in config:
        config['init_train_machine'] = config['train_machine'].copy() if isinstance(config['train_machine'], dict) else config['train_machine']
    elif 'train_machine' in config and 'init_train_machine' in config:
        if isinstance(config['train_machine'], dict) and isinstance(config['init_train_machine'], dict):
            config['init_train_machine'] = merge_configs(config['train_machine'], config['init_train_machine'])

    return config


def separate_backend_params(config):
    """Separate backend-specific parameters.

    Extracts scf.pyscf.* and scf.abacus.* into separate dicts.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (config, pyscf_params, abacus_params)
    """
    config = config.copy()
    pyscf_params = {}
    abacus_params = {}

    # Extract scf.pyscf.* parameters
    if 'scf' in config and isinstance(config['scf'], dict):
        if 'pyscf' in config['scf']:
            pyscf_params = config['scf'].pop('pyscf')
        if 'abacus' in config['scf']:
            abacus_params = config['scf'].pop('abacus')

    return config, pyscf_params, abacus_params
