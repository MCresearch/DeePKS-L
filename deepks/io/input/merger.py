"""Configuration merger for DeePKS."""

from copy import deepcopy

from .config import TASK_PARAM_KEYS


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
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def _pick_params(config, keys):
    return {key: deepcopy(config[key]) for key in keys if key in config}


def package_config(config):
    """Package merged config into explicit runtime parameter bundles."""
    packaged = {
        name: _pick_params(config, keys)
        for name, keys in TASK_PARAM_KEYS.items()
    }

    test_param = packaged['test_param']
    if 'systems_test' in test_param and 'data_paths' not in test_param:
        test_param['data_paths'] = deepcopy(test_param['systems_test'])

    return {
        'type': config.get('type'),
        'raw_config': deepcopy(config),
        **packaged,
    }


def apply_parameter_inheritance(config):
    """Apply parameter inheritance rules.

    Init parameters inherit from non-init parameters:
    - init_scf inherits from scf_input
    - init_train inherits from train_input
    - init_scf_machine inherits from scf_machine
    - init_train_machine inherits from train_machine
    - init_scf_abacus inherits from scf_abacus

    Args:
        config: Configuration dictionary

    Returns:
        dict: Configuration with inheritance applied
    """
    config = config.copy()

    if 'scf_input' in config and 'init_scf' not in config:
        config['init_scf'] = config['scf_input'].copy() if isinstance(config['scf_input'], dict) else config['scf_input']
    elif 'scf_input' in config and 'init_scf' in config:
        if isinstance(config['scf_input'], dict) and isinstance(config['init_scf'], dict):
            config['init_scf'] = merge_configs(config['scf_input'], config['init_scf'])

    if 'train_input' in config and 'init_train' not in config:
        config['init_train'] = config['train_input'].copy() if isinstance(config['train_input'], dict) else config['train_input']
    elif 'train_input' in config and 'init_train' in config:
        if isinstance(config['train_input'], dict) and isinstance(config['init_train'], dict):
            config['init_train'] = merge_configs(config['train_input'], config['init_train'])

    if 'scf_machine' in config and 'init_scf_machine' not in config:
        config['init_scf_machine'] = config['scf_machine'].copy() if isinstance(config['scf_machine'], dict) else config['scf_machine']
    elif 'scf_machine' in config and 'init_scf_machine' in config:
        if isinstance(config['scf_machine'], dict) and isinstance(config['init_scf_machine'], dict):
            config['init_scf_machine'] = merge_configs(config['scf_machine'], config['init_scf_machine'])

    if 'train_machine' in config and 'init_train_machine' not in config:
        config['init_train_machine'] = config['train_machine'].copy() if isinstance(config['train_machine'], dict) else config['train_machine']
    elif 'train_machine' in config and 'init_train_machine' in config:
        if isinstance(config['train_machine'], dict) and isinstance(config['init_train_machine'], dict):
            config['init_train_machine'] = merge_configs(config['train_machine'], config['init_train_machine'])

    if 'scf_abacus' in config and 'init_scf_abacus' not in config:
        config['init_scf_abacus'] = config['scf_abacus'].copy() if isinstance(config['scf_abacus'], dict) else config['scf_abacus']
    elif 'scf_abacus' in config and 'init_scf_abacus' in config:
        if isinstance(config['scf_abacus'], dict) and isinstance(config['init_scf_abacus'], dict):
            config['init_scf_abacus'] = merge_configs(config['scf_abacus'], config['init_scf_abacus'])

    return config
