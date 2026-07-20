"""Factory helpers for property-recovery schemes."""

from deepks.physics.schemes.energy_descriptor import EnergyDescriptorScheme


_SCHEME_REGISTRY = {
    "energy_descriptor": EnergyDescriptorScheme,
    "energy-descriptor": EnergyDescriptorScheme,
}


def normalize_scheme_name(name):
    normalized = (name or "energy_descriptor").strip().lower()
    if normalized not in _SCHEME_REGISTRY:
        raise KeyError(f"Unknown property scheme: {name!r}")
    return normalized


def build_property_scheme(name, **kwargs):
    return _SCHEME_REGISTRY[normalize_scheme_name(name)](**kwargs)
