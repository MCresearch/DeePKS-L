"""Config-translation helpers for the hierarchical-regression recipe.

These helpers used to live in ``deepks.workflows.train.runtime`` but the
recipe (interface layer) needed to call them too — which created an
``interface → workflows`` upward dependency. Moving them to the interface
layer keeps that arrow pointing the right way: workflows pull recipe
config translators from the interface, never the other way around.
"""

from copy import deepcopy
from typing import Any, Dict, List, Sequence

from deepks.interface.objectives import build_descriptor_property_objective_args
from deepks.io import utils as _io_utils


_RESERVED_NAME_KEYS = (
    "e_name",
    "f_name",
    "gvx_name",
    "s_name",
    "gvepsl_name",
    "o_name",
    "op_name",
    "h_name",
    "vdp_name",
    "vdrp_name",
    "phialpha_name",
    "gevdm_name",
    "h_base_name",
    "h_ref_name",
    "hamiltonian_name",
    "hr_name",
    "csr_hr_name",
    "overlap_name",
    "eg_name",
    "gveg_name",
    "gldv_name",
    "iR_mat_name",
    "phialpha_r_name",
)


def normalize_hierarchical_terms(objective_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pick the active objective term list from the recipe config."""
    terms = objective_cfg.get("terms") if isinstance(objective_cfg.get("terms"), list) else []
    if terms:
        return [deepcopy(term) for term in terms]
    level_losses = (
        objective_cfg.get("level_losses")
        if isinstance(objective_cfg.get("level_losses"), list)
        else []
    )
    if level_losses:
        return [deepcopy(term) for term in level_losses]
    return [{"name": "hr", "target": {"format": "collected_hr_delta", "name": "l_hr_delta"}, "weight": 1.0}]


def normalize_hierarchical_primary_output(objective_cfg: Dict[str, Any]) -> str:
    """Return the only supported primary output (``energy``) or raise."""
    primary_output = objective_cfg.get("primary_output")
    if primary_output is None:
        return "energy"
    normalized = str(primary_output).strip().lower()
    if normalized != "energy":
        raise ValueError(
            "Hierarchical regression only supports primary_output='energy', got "
            f"{primary_output!r}"
        )
    return normalized


def term_property_name(term: Dict[str, Any]) -> str:
    """Map an objective term to its target property name."""
    target = term.get("target") if isinstance(term.get("target"), dict) else {}
    target_format = target.get("format")
    if target_format == "collected_energy_delta":
        return "energy"
    if target_format == "collected_force_delta":
        return "force"
    if target_format == "collected_stress_delta":
        return "stress"
    if target_format == "collected_hr_delta":
        return "v_delta_r"
    return str(term.get("name", "term"))


def _restrict_stage_loader_fields(loader_args: Dict[str, Any], *, keep: Sequence[str]) -> None:
    """Null out loader name fields not in the ``keep`` set."""
    keep_set = set(keep)
    for key in _RESERVED_NAME_KEYS:
        if key not in keep_set:
            loader_args[key] = None


def build_stage_data_specs(
    stage_cfgs,
    global_loader_cfg,
    rep_name,
    objective_terms,
    *,
    primary_output,
):
    """Translate per-stage data declarations into reader-ready specs."""
    specs = []
    for stage_cfg in stage_cfgs:
        loader_args = deepcopy(global_loader_cfg)
        stage_loader = stage_cfg.get("loader") if isinstance(stage_cfg.get("loader"), dict) else {}
        loader_args.update(deepcopy(stage_loader))
        if rep_name and "d_name" not in loader_args:
            loader_args["d_name"] = rep_name
        target_formats = {}
        keep_fields = {"d_name", "conv_name", "atom_name", "box_name"}
        for term in objective_terms:
            target_cfg = term.get("target") if isinstance(term.get("target"), dict) else {}
            target_format = target_cfg.get("format", "collected_hr_delta")
            target_name = target_cfg.get("name") or target_cfg.get("hr_name")
            term_name = term_property_name(term)
            target_formats[str(term.get("name", term_name))] = target_format
            if target_format == "collected_energy_delta":
                loader_args["e_name"] = target_name or "l_e_delta"
                keep_fields.add("e_name")
            elif target_format == "collected_force_delta":
                loader_args["f_name"] = target_name or "l_f_delta"
                keep_fields.update({"f_name", "gvx_name"})
            elif target_format == "collected_stress_delta":
                loader_args["s_name"] = target_name or "l_s_delta"
                keep_fields.update({"s_name", "gvepsl_name"})
            elif target_format == "collected_hr_delta":
                loader_args["hr_name"] = target_name or "l_hr_delta"
                keep_fields.add("hr_name")
                if primary_output == "energy":
                    # Two-strategy support for V_delta(R) chain-rule recovery:
                    #   - deepks_v_delta=-1 → vdr_precalc.npy (single tensor)
                    #   - deepks_v_delta=-2 → grad_evdm.npy + iR_mat.npy + phialpha_r.npy
                    keep_fields.update({
                        "vdrp_name", "gevdm_name", "iR_mat_name", "phialpha_r_name",
                    })
            else:
                raise ValueError(f"Unsupported stage target format: {target_format}")
        _restrict_stage_loader_fields(loader_args, keep=tuple(keep_fields))
        train_paths = _io_utils.load_dirs(stage_cfg.get("train", []))
        test_paths = stage_cfg.get("test")
        if test_paths is not None:
            test_paths = _io_utils.load_dirs(test_paths)
        specs.append(
            {
                "level": int(stage_cfg["level"]),
                "name": stage_cfg.get("name"),
                "loader_args": loader_args,
                "target_formats": target_formats,
                "train_paths": train_paths,
                "test_paths": test_paths,
            }
        )
    return specs


def build_hierarchical_descriptor_objective_args(
    objective_cfg: Dict[str, Any],
    objective_terms: List[Dict[str, Any]],
):
    """Convert hierarchical term list into descriptor-property objective args."""
    synthetic = {
        key: deepcopy(objective_cfg[key])
        for key in (
            "grad_penalty",
            "energy_per_atom",
            "vd_divide_by_nlocal",
            "vd_masked_loss",
            "vd_masked_S_threshold",
            "vd_masked_H_threshold",
            "vd_masked_width",
            "use_safe_eigh",
        )
        if key in objective_cfg
    }
    losses = []
    for term in objective_terms:
        mapped_name = term_property_name(term)
        if mapped_name not in {"energy", "force", "stress", "v_delta_r"}:
            continue
        item = {
            "name": mapped_name,
            "weight": deepcopy(term.get("weight", 1.0)),
        }
        if isinstance(term.get("loss"), dict):
            item["loss"] = deepcopy(term["loss"])
        losses.append(item)
    synthetic["losses"] = losses
    return build_descriptor_property_objective_args(synthetic)


__all__ = [
    "normalize_hierarchical_terms",
    "normalize_hierarchical_primary_output",
    "term_property_name",
    "build_stage_data_specs",
    "build_hierarchical_descriptor_objective_args",
]
