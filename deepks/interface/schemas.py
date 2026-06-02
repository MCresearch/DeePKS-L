"""Schema definitions for interface recipes.

These schemas describe interface-layer assemblies: each recipe binds together
an ML model family, a physics/property recovery scheme, and the expected high-
level input/output fields. They are orchestration metadata, so they stay in the
interface layer rather than in ML or physics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


def _frozen_output_reducers(reducers):
    """Return an immutable mapping of output_name -> reducer spec."""

    if not reducers:
        return {}
    return dict(reducers)


@dataclass(frozen=True)
class RecipeSchema:
    """Minimal schema metadata for a training/evaluation recipe.

    Fields
    ------
    name
        Canonical recipe name (alias lookups land here).
    model_family
        Which ML model class to instantiate (resolved via ``ml.model_io``).
    property_scheme
        Which physics ``PropertyScheme`` recovers supervised quantities.
    input_fields
        Names of the ``TaskBatch.model_inputs`` entries this recipe's model
        consumes. ``("descriptor",)`` for the existing descriptor-energy
        recipes; multi-input GNN-style recipes may declare e.g.
        ``("nodes", "edges", "coords")``.
    output_fields
        Names of the supervision-ready quantities produced after reduction.
        Drives the objective adapter's ``primary_property`` resolution and
        the property-scheme requests.
    output_reducers
        Optional ``Dict[output_name, reducer_spec]``. ``reducer_spec`` is a
        string (looked up in :mod:`deepks.interface.reducers`) or an
        ``OutputReducer`` instance. When absent for an output, the objective
        adapter defaults to :class:`Identity` — i.e., the model's raw output
        for that name IS the supervision-ready quantity. Descriptor-energy
        recipes set ``{"energy": "sum_over_atoms"}`` so the per-atom model
        contribution is summed to a per-system scalar.
    input_field_mapping
        Optional ``Dict[sample_key, model_input_key]``. The sample-to-
        TaskBatch adapter consults this to route reader fields into the
        right ``model_inputs`` slot for multi-input models. Defaults
        gracefully to the legacy ``{"eig": "descriptor"}`` when absent.
    """

    name: str
    model_family: str
    property_scheme: str
    input_fields: Tuple[str, ...]
    output_fields: Tuple[str, ...]
    output_reducers: Mapping[str, Any] = field(default_factory=dict)
    input_field_mapping: Mapping[str, str] = field(default_factory=dict)


DEFAULT_RECIPE_NAME = "corrnet-energy"
ENERGY_ONLY_RECIPE_NAME = "corrnet-energy-only"
LINEAR_RECIPE_NAME = "linear-energy"
HIERARCHICAL_REGRESSION_RECIPE_NAME = "hierarchical-regression"


_DESCRIPTOR_ENERGY_DEFAULTS = {
    "output_reducers": {"energy": "sum_over_atoms"},
    "input_field_mapping": {"eig": "descriptor"},
}


CORRNET_ENERGY_SCHEMA = RecipeSchema(
    name=DEFAULT_RECIPE_NAME,
    model_family="corrnet",
    property_scheme="energy_descriptor",
    input_fields=("descriptor",),
    output_fields=("energy",),
    **_DESCRIPTOR_ENERGY_DEFAULTS,
)


CORRNET_ENERGY_ONLY_SCHEMA = RecipeSchema(
    name=ENERGY_ONLY_RECIPE_NAME,
    model_family="corrnet",
    property_scheme="energy_descriptor",
    input_fields=("descriptor",),
    output_fields=("energy",),
    **_DESCRIPTOR_ENERGY_DEFAULTS,
)


LINEAR_ENERGY_SCHEMA = RecipeSchema(
    name=LINEAR_RECIPE_NAME,
    model_family="linear",
    property_scheme="energy_descriptor",
    input_fields=("descriptor",),
    output_fields=("energy",),
    **_DESCRIPTOR_ENERGY_DEFAULTS,
)


HIERARCHICAL_REGRESSION_SCHEMA = RecipeSchema(
    name=HIERARCHICAL_REGRESSION_RECIPE_NAME,
    model_family="hierarchical_regression",
    property_scheme="energy_descriptor",
    input_fields=("descriptor",),
    output_fields=("energy",),
    **_DESCRIPTOR_ENERGY_DEFAULTS,
)


RECIPE_SCHEMAS: Dict[str, RecipeSchema] = {
    CORRNET_ENERGY_SCHEMA.name: CORRNET_ENERGY_SCHEMA,
    CORRNET_ENERGY_ONLY_SCHEMA.name: CORRNET_ENERGY_ONLY_SCHEMA,
    LINEAR_ENERGY_SCHEMA.name: LINEAR_ENERGY_SCHEMA,
    HIERARCHICAL_REGRESSION_SCHEMA.name: HIERARCHICAL_REGRESSION_SCHEMA,
}
