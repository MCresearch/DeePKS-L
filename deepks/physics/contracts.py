"""Shared contract names for physics-side property execution.

These string constants are the canonical keys used to communicate model
outputs and model-input derivatives between the interface objective
adapter and the physics property scheme. Holding them in one place
prevents the boundary from drifting back into the implicit "everyone
just knows it's called 'eig'" pattern that earlier versions suffered
from.

Naming conventions
------------------

``ModelOutputKeys.PRIMARY_OUTPUT``
    The model's main supervision-ready output, after the interface-layer
    reducer (see :mod:`deepks.interface.reducers`) has been applied.
    Single-output models always populate this key. Multi-output models
    may populate additional named outputs alongside it; property schemes
    request specific output names via
    :meth:`PropertyScheme.required_model_outputs`.

``ModelDerivativeKeys.INPUT``
    Autograd of the primary output with respect to the model's primary
    input tensor — used for chain-rule recoveries (force, V_delta(R),
    etc.). For multi-input models (e.g. a graph network consuming
    nodes + edges + coords), schemes may extend the derivatives dict
    with additional input names and the objective adapter computes a
    gradient per requested name. The default ``"input"`` key keeps the
    single-input case wire-compatible with everything written before
    multi-input recipes exist.
"""


class PropertyNames:
    ENERGY = "energy"
    FORCE = "force"
    STRESS = "stress"
    ORBITAL = "orbital"
    V_DELTA = "v_delta"
    BAND = "band"
    PHI = "phi"
    VDR = "vdr"

    ALL = frozenset(
        {
            ENERGY,
            FORCE,
            STRESS,
            ORBITAL,
            V_DELTA,
            BAND,
            PHI,
            VDR,
        }
    )


class ModelOutputKeys:
    """Canonical keys in the ``model_outputs`` dict passed to a PropertyScheme."""

    PRIMARY_OUTPUT = "primary_output"


class ModelDerivativeKeys:
    """Canonical keys in the ``model_derivatives`` dict passed to a PropertyScheme."""

    INPUT = "input"
