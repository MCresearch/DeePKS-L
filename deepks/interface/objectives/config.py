"""Objective configuration normalization for interface-owned task assembly."""

from copy import deepcopy


_OBJECTIVE_OPTION_KEYS = (
    "grad_penalty",
    "energy_per_atom",
    "vd_divide_by_nlocal",
    "vd_masked_loss",
    "vd_masked_S_threshold",
    "vd_masked_H_threshold",
    "vd_masked_width",
    "use_safe_eigh",
)

_LOSS_NAME_MAP = {
    "energy": ("energy_factor", "energy_lossfn", None),
    "force": ("force_factor", "force_lossfn", None),
    "stress": ("stress_factor", "stress_lossfn", None),
    "orbital": ("orbital_factor", "orbital_lossfn", None),
    "bandgap_orbital": ("orbital_factor", "orbital_lossfn", None),
    "v_delta": ("v_delta_factor", "v_delta_lossfn", None),
    "v_delta_r": ("v_delta_r_factor", "v_delta_r_lossfn", None),
    "phi": ("phi_factor", "phi_lossfn", "phi_occ"),
    "band": ("band_factor", "band_lossfn", "band_occ"),
    "bandgap": ("bandgap_factor", "bandgap_lossfn", "bandgap_occ"),
    "density_m": ("density_m_factor", "density_m_lossfn", "density_m_occ"),
    "phi_align": ("phi_align_factor", "phi_align_lossfn", "phi_align_occ"),
    "density": ("density_factor", None, None),
}

_FACTOR_KEYS = tuple(mapped[0] for mapped in _LOSS_NAME_MAP.values())
_LOSS_KEYS = tuple(mapped[1] for mapped in _LOSS_NAME_MAP.values() if mapped[1] is not None)
_OCC_KEYS = tuple(mapped[2] for mapped in _LOSS_NAME_MAP.values() if mapped[2] is not None)


def build_descriptor_property_objective_args(objective_config):
    """Normalize packed task objective config into adapter kwargs."""
    objective = objective_config if isinstance(objective_config, dict) else {}
    objective_args = {
        key: value
        for key, value in {
            option_key: objective.get(option_key)
            for option_key in _OBJECTIVE_OPTION_KEYS
        }.items()
        if value is not None
    }

    losses = objective.get("losses")
    if isinstance(losses, dict):
        normalized_losses = []
        for name, value in losses.items():
            if isinstance(value, dict) and set(value).issubset({"main", "init"}):
                continue
            if isinstance(value, dict):
                normalized_losses.append({"name": name, **value})
            else:
                normalized_losses.append({"name": name, "weight": value})
    elif isinstance(losses, list):
        normalized_losses = [deepcopy(item) for item in losses]
    else:
        normalized_losses = []

    for item in normalized_losses:
        if not isinstance(item, dict) or "name" not in item:
            continue
        mapped = _LOSS_NAME_MAP.get(str(item["name"]).strip().lower())
        if mapped is None:
            continue
        factor_key, loss_key, occ_key = mapped
        if "weight" in item:
            objective_args[factor_key] = item["weight"]
        elif "factor" in item:
            objective_args[factor_key] = item["factor"]
        if loss_key and "loss" in item:
            objective_args[loss_key] = deepcopy(item["loss"])
        if occ_key and "occ" in item:
            objective_args[occ_key] = deepcopy(item["occ"])
    return objective_args


def build_descriptor_property_eval_args(objective_args, *, detailed=False):
    """Build evaluation objective kwargs from normalized training objective kwargs."""
    normalized = dict(objective_args or {})
    energy_per_atom = normalized.get("energy_per_atom", 0)
    if energy_per_atom is None:
        energy_per_atom = 0

    if not detailed:
        return {
            "energy_factor": 1.0,
            "force_factor": 0.0,
            "density_factor": 0.0,
            "grad_penalty": 0.0,
            "energy_per_atom": energy_per_atom,
        }

    eval_args = {}
    for factor_key in _FACTOR_KEYS:
        if factor_key in normalized:
            eval_args[factor_key] = 0.0 if normalized[factor_key] == 0 else 1.0
    for copied_key in (*_LOSS_KEYS, *_OCC_KEYS, *_OBJECTIVE_OPTION_KEYS):
        if copied_key in normalized:
            eval_args[copied_key] = deepcopy(normalized[copied_key])
    eval_args["energy_per_atom"] = energy_per_atom
    return eval_args


def select_energy_only_objective_args(objective_args):
    """Filter normalized objective kwargs down to the energy-only subset."""
    normalized = dict(objective_args or {})
    selected = {}
    for key in ("energy_factor", "energy_lossfn", "energy_per_atom"):
        if key in normalized:
            selected[key] = deepcopy(normalized[key])
    return selected
