"""Shared builders for descriptor-property objectives."""

from deepks.interface.objectives.config import (
    build_descriptor_property_eval_args,
    select_energy_only_objective_args,
)
from deepks.interface.objectives.descriptor_properties import DescriptorPropertyObjectiveAdapter
from deepks.ml.utils import make_loss
from deepks.physics.engine import PropertyEngine
from deepks.physics.schemes import build_property_scheme


def _build_property_engine(*, property_engine=None, property_scheme=None, use_safe_eigh=False):
    if property_engine is not None:
        return property_engine
    if property_scheme is None:
        raise ValueError("Descriptor-property objective construction requires an explicit property_scheme")
    scheme = build_property_scheme(property_scheme, use_safe_eigh=use_safe_eigh)
    return PropertyEngine(scheme=scheme)


def build_descriptor_property_objective(
    objective_args=None,
    *,
    property_engine=None,
    property_scheme=None,
    primary_property="energy",
    primary_input="descriptor",
    output_reducers=None,
    primary_output_reducer="sum_over_atoms",
):
    objective_args = dict(objective_args or {})
    engine = _build_property_engine(
        property_engine=property_engine,
        property_scheme=property_scheme,
        use_safe_eigh=bool(objective_args.get("use_safe_eigh", False)),
    )
    objective_args.setdefault("primary_property", primary_property)
    objective_args.setdefault("primary_input", primary_input)
    objective_args.setdefault("output_reducers", output_reducers)
    objective_args.setdefault("primary_output_reducer", primary_output_reducer)
    return DescriptorPropertyObjectiveAdapter(property_engine=engine, **objective_args)


def build_descriptor_property_eval_objective(
    objective_args=None,
    *,
    detailed=False,
    property_engine=None,
    property_scheme=None,
    primary_property="energy",
    primary_input="descriptor",
    output_reducers=None,
    primary_output_reducer="sum_over_atoms",
):
    eval_args = build_descriptor_property_eval_args(objective_args or {}, detailed=bool(detailed))
    if "energy_lossfn" not in eval_args:
        eval_args["energy_lossfn"] = make_loss()
    engine = _build_property_engine(
        property_engine=property_engine,
        property_scheme=property_scheme,
        use_safe_eigh=bool(eval_args.get("use_safe_eigh", False)),
    )
    eval_args.setdefault("primary_property", primary_property)
    eval_args.setdefault("primary_input", primary_input)
    eval_args.setdefault("output_reducers", output_reducers)
    eval_args.setdefault("primary_output_reducer", primary_output_reducer)
    return DescriptorPropertyObjectiveAdapter(property_engine=engine, **eval_args)


def build_energy_only_objective_args(objective_args=None):
    return select_energy_only_objective_args(objective_args or {})
