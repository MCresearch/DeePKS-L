"""Current DeePKS scheme: primary output is energy-like over descriptor inputs."""

from deepks.physics.base import PropertyScheme
from deepks.physics.contracts import ModelDerivativeKeys, ModelOutputKeys, PropertyNames
from deepks.physics.properties import (
    band_from_solution,
    energy_from_primary_output,
    force_from_descriptor_gradient,
    orbital_from_descriptor_gradient,
    phi_from_solution,
    solve_band_phi,
    stress_from_descriptor_gradient,
    v_delta_from_context,
    vdr_from_context,
)


class EnergyDescriptorScheme(PropertyScheme):
    """Recover physical quantities for the current energy-over-descriptor setup."""

    def __init__(self, *, use_safe_eigh=False):
        self.use_safe_eigh = use_safe_eigh

    def supported_properties(self):
        return PropertyNames.ALL

    def required_model_outputs(self, requested_properties):
        requested = set(requested_properties or ())
        if not requested:
            return set()
        return {ModelOutputKeys.PRIMARY_OUTPUT}

    def required_model_derivatives(self, requested_properties):
        requested = set(requested_properties or ())
        return {ModelDerivativeKeys.INPUT: bool(requested - {PropertyNames.ENERGY})}

    def validate_context(self, requested_properties, context):
        requested = set(requested_properties or ())
        required_by_property = {
            PropertyNames.FORCE: ({"gvx"},),
            PropertyNames.STRESS: ({"gvepsl"},),
            PropertyNames.ORBITAL: ({"op", "orbital_shape"},),
            PropertyNames.V_DELTA: ({"vdp"}, {"phialpha", "gevdm"}),
            PropertyNames.BAND: ({"h_base", "vdp"}, {"h_base", "phialpha", "gevdm"}),
            PropertyNames.PHI: ({"h_base", "vdp"}, {"h_base", "phialpha", "gevdm"}),
            PropertyNames.VDR: ({"vdrp"}, {"gevdm", "iR_mat", "overlap", "data_shape"}),
        }
        available = set(context)
        for name in requested:
            options = required_by_property.get(name, ())
            if not options:
                continue
            if any(option <= available for option in options):
                continue
            formatted = " or ".join(str(sorted(option)) for option in options)
            raise KeyError(f"Missing context for property '{name}', expected {formatted}")

    def compute_property(self, name, *, model_outputs, model_derivatives, context, cache):
        input_grad = model_derivatives.get(ModelDerivativeKeys.INPUT)
        if name == PropertyNames.ENERGY:
            return energy_from_primary_output(model_outputs[ModelOutputKeys.PRIMARY_OUTPUT])
        if name == PropertyNames.FORCE:
            return force_from_descriptor_gradient(input_grad, context["gvx"])
        if name == PropertyNames.STRESS:
            return stress_from_descriptor_gradient(input_grad, context["gvepsl"])
        if name == PropertyNames.ORBITAL:
            return orbital_from_descriptor_gradient(
                input_grad, context["op"], context["orbital_shape"]
            )
        if name == PropertyNames.V_DELTA:
            if PropertyNames.V_DELTA not in cache:
                cache[PropertyNames.V_DELTA] = v_delta_from_context(input_grad, context)
            return cache[PropertyNames.V_DELTA]
        if name in {PropertyNames.BAND, PropertyNames.PHI}:
            if "band_solution" not in cache:
                vd_pred = self.compute_property(
                    PropertyNames.V_DELTA,
                    model_outputs=model_outputs,
                    model_derivatives=model_derivatives,
                    context=context,
                    cache=cache,
                )
                cache["band_solution"] = solve_band_phi(context, vd_pred, self.use_safe_eigh)
            if name == PropertyNames.BAND:
                return band_from_solution(cache["band_solution"])
            return phi_from_solution(cache["band_solution"])
        if name == PropertyNames.VDR:
            if PropertyNames.VDR not in cache:
                cache[PropertyNames.VDR] = vdr_from_context(input_grad, context)
            return cache[PropertyNames.VDR]
        raise KeyError(f"Unsupported property '{name}' for EnergyDescriptorScheme")
