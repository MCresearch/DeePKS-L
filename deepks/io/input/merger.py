"""Configuration merge helpers for DeePKS."""


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
