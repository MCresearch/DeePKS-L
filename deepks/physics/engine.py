"""Physics property engine."""

from deepks.physics.contracts import PropertyNames


class PropertyEngine:
    """Resolve requested physical quantities through a concrete scheme."""

    def __init__(self, scheme):
        if scheme is None:
            raise ValueError("PropertyEngine requires an explicit scheme")
        self.scheme = scheme

    def _normalize_requested(self, names):
        requested = set(names or ())
        unknown = requested - PropertyNames.ALL
        if unknown:
            raise KeyError(f"Unknown requested properties: {sorted(unknown)}")
        unsupported = requested - set(self.scheme.supported_properties())
        if unsupported:
            raise KeyError(
                f"Unsupported properties for {self.scheme.__class__.__name__}: {sorted(unsupported)}"
            )
        return requested

    def _validate_model_signals(self, requested_properties, model_outputs, model_derivatives):
        required_outputs = set(self.scheme.required_model_outputs(requested_properties))
        missing_outputs = required_outputs - set(model_outputs)
        if missing_outputs:
            raise KeyError(f"Missing model outputs: {sorted(missing_outputs)}")

        required_derivatives = self.scheme.required_model_derivatives(requested_properties)
        missing_derivatives = {
            key for key, required in required_derivatives.items() if required and model_derivatives.get(key) is None
        }
        if missing_derivatives:
            raise KeyError(f"Missing model derivatives: {sorted(missing_derivatives)}")

    def required_model_derivatives(self, requested_properties):
        requested = self._normalize_requested(requested_properties)
        return self.scheme.required_model_derivatives(requested)

    def available_properties(self, requested_properties, context):
        """Return the subset of requested properties whose context is satisfied.

        Properties whose ``PropertyScheme.validate_context`` would raise are
        silently dropped from the returned set. Callers that need a hard
        failure should keep using ``get`` / ``get_many``.
        """
        requested = self._normalize_requested(requested_properties)
        available = set()
        for name in requested:
            try:
                self.scheme.validate_context({name}, context)
            except KeyError:
                continue
            available.add(name)
        return available

    def get(self, name, *, model_outputs, model_derivatives, context, cache=None):
        requested = self._normalize_requested({name})
        self._validate_model_signals(requested, model_outputs, model_derivatives)
        self.scheme.validate_context(requested, context)
        cache = {} if cache is None else cache
        return self.scheme.compute_property(
            name,
            model_outputs=model_outputs,
            model_derivatives=model_derivatives,
            context=context,
            cache=cache,
        )

    def get_many(self, names, *, model_outputs, model_derivatives, context):
        requested = self._normalize_requested(names)
        self._validate_model_signals(requested, model_outputs, model_derivatives)
        self.scheme.validate_context(requested, context)
        cache = {}
        return {
            name: self.scheme.compute_property(
                name,
                model_outputs=model_outputs,
                model_derivatives=model_derivatives,
                context=context,
                cache=cache,
            )
            for name in names
        }

    def get_energy(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("energy", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)

    def get_force(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("force", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)

    def get_stress(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("stress", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)

    def get_orbital(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("orbital", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)

    def get_v_delta(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("v_delta", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)

    def get_band(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("band", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)

    def get_phi(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("phi", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)

    def get_vdr(self, *, model_outputs, model_derivatives, context, cache=None):
        return self.get("vdr", model_outputs=model_outputs, model_derivatives=model_derivatives, context=context, cache=cache)
