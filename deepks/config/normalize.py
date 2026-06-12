"""Input format normalization for DeePKS configuration."""

from copy import deepcopy

def merge_expanded_value(base, override, path):
    if isinstance(base, dict) and isinstance(override, dict):
        result = deepcopy(base)
        for key, value in override.items():
            child_path = f"{path}.{key}" if path else key
            if key in result:
                result[key] = merge_expanded_value(result[key], value, child_path)
            else:
                result[key] = deepcopy(value)
        return result
    if base == override:
        return deepcopy(base)
    raise ValueError(f"Conflicting configuration values at '{path}': {base!r} vs {override!r}")

def normalize_config(config):
    """Expand dotted keys into the standard nested input structure."""

    expanded = {}
    for raw_key, raw_value in deepcopy(config).items():
        key = str(raw_key)
        value = normalize_config(raw_value) if isinstance(raw_value, dict) else deepcopy(raw_value)
        if "." not in key:
            if key in expanded:
                expanded[key] = merge_expanded_value(expanded[key], value, key)
            else:
                expanded[key] = value
            continue

        nested_value = value
        for part in reversed(key.split(".")):
            nested_value = {part: nested_value}
        expanded = merge_expanded_value(expanded, nested_value, "")
    return expanded
