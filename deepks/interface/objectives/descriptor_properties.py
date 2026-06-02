"""Objective adapter for descriptor-driven property supervision."""

import torch

from deepks.interface.adapters import TaskBatchRepresentationBuilder
from deepks.interface.batch import TaskBatch
from deepks.interface.objectives.terms import build_descriptor_property_terms
from deepks.interface.base import ObjectiveAdapter
from deepks.interface.reducers import Identity, OutputReducer, SumOverAtoms, build_reducer


class DescriptorPropertyObjectiveAdapter(ObjectiveAdapter):
    """Task objective for descriptor-input models with supervised physical properties."""

    def __init__(
        self,
        energy_factor=1.0,
        force_factor=0.0,
        stress_factor=0.0,
        orbital_factor=0.0,
        v_delta_factor=0.0,
        v_delta_r_factor=0.0,
        phi_factor=0.0,
        phi_occ=0,
        band_factor=0.0,
        band_occ=0,
        bandgap_factor=0.0,
        bandgap_occ=0,
        density_m_factor=0.0,
        density_m_occ=0,
        phi_align_factor=0.0,
        phi_align_occ=0,
        density_factor=0.0,
        grad_penalty=0.0,
        energy_lossfn=None,
        force_lossfn=None,
        stress_lossfn=None,
        orbital_lossfn=None,
        v_delta_lossfn=None,
        v_delta_r_lossfn=None,
        phi_lossfn=None,
        phi_align_lossfn=None,
        band_lossfn=None,
        bandgap_lossfn=None,
        density_m_lossfn=None,
        energy_per_atom=0,
        vd_divide_by_nlocal=False,
        vd_masked_loss=0,
        vd_masked_S_threshold=1e-6,
        vd_masked_H_threshold=1e-6,
        vd_masked_width=1,
        use_safe_eigh=False,
        property_engine=None,
        primary_property="energy",
        primary_input="descriptor",
        output_reducers=None,
        primary_output_reducer="sum_over_atoms",
    ):
        if energy_per_atom is None:
            energy_per_atom = 0
        # The recipe owns which physical quantity is the model's primary output.
        # The objective should not assume a specific scalar like "energy"; this
        # ``primary_property`` is the property name that is always requested
        # from the property engine even when no term explicitly requires it,
        # and that is used as the dtype/device template for the zero loss
        # accumulator. Pass ``None`` or an empty string to disable.
        primary_property = (primary_property or "").strip().lower() or None
        self.primary_property = primary_property
        # R3: recipe declares which TaskBatch.model_inputs slot carries the
        # tensor that the property scheme will autograd through. Defaults to
        # ``"descriptor"`` to preserve behavior for existing recipes.
        self.primary_input = primary_input or "descriptor"
        # R1: reducers split per output. ``primary_output_reducer`` is applied
        # to the model's primary output. ``output_reducers`` lets the recipe
        # plug in per-named-output reducers (currently only used for the
        # primary output; future multi-output models will expand this).
        reducers_map = dict(output_reducers or {})
        if self.primary_property and self.primary_property not in reducers_map:
            reducers_map[self.primary_property] = primary_output_reducer
        # SumOverAtoms is the conventional descriptor-energy reducer; ship it
        # as the default so existing recipes keep their per-atom-sum behavior.
        self.output_reducers = {
            name: build_reducer(spec)
            for name, spec in reducers_map.items()
        }
        self._primary_reducer = self.output_reducers.get(
            self.primary_property, SumOverAtoms()
        )
        self.e_factor = energy_factor
        self.f_factor = force_factor
        self.s_factor = stress_factor
        self.o_factor = orbital_factor
        self.vd_factor = v_delta_factor
        self.vdr_factor = v_delta_r_factor
        self.phi_factor = phi_factor
        self.band_factor = band_factor
        self.bandgap_factor = bandgap_factor
        self.density_m_factor = density_m_factor
        self.phi_align_factor = phi_align_factor
        self.d_factor = density_factor
        self.g_penalty = grad_penalty
        self.energy_per_atom = energy_per_atom
        self.vd_divide_by_nlocal = vd_divide_by_nlocal
        self.vd_masked_loss = vd_masked_loss
        self.vd_masked_S_threshold = vd_masked_S_threshold
        self.vd_masked_H_threshold = vd_masked_H_threshold
        self.vd_masked_width = vd_masked_width

        if property_engine is None:
            raise ValueError("DescriptorPropertyObjectiveAdapter requires an explicit property_engine")
        self.property_engine = property_engine
        # R3: ``primary_input`` declares which TaskBatch.model_inputs slot
        # carries the model's primary input. The mapping from raw sample keys
        # to ``model_inputs`` slots is the recipe's ``input_field_mapping``
        # (threaded into the builder); when absent, the default
        # ``{"eig": "descriptor"}`` mapping keeps existing recipes working.
        self.batch_builder = TaskBatchRepresentationBuilder(
            input_field_mapping={
                self.primary_input: self.primary_input
            } if self.primary_input not in {"descriptor"} else None
        )
        self.objective_args = {
            "energy_factor": energy_factor,
            "force_factor": force_factor,
            "stress_factor": stress_factor,
            "orbital_factor": orbital_factor,
            "v_delta_factor": v_delta_factor,
            "v_delta_r_factor": v_delta_r_factor,
            "phi_factor": phi_factor,
            "phi_occ": phi_occ,
            "band_factor": band_factor,
            "band_occ": band_occ,
            "bandgap_factor": bandgap_factor,
            "bandgap_occ": bandgap_occ,
            "density_m_factor": density_m_factor,
            "density_m_occ": density_m_occ,
            "phi_align_factor": phi_align_factor,
            "phi_align_occ": phi_align_occ,
            "density_factor": density_factor,
            "grad_penalty": grad_penalty,
            "energy_lossfn": energy_lossfn,
            "force_lossfn": force_lossfn,
            "stress_lossfn": stress_lossfn,
            "orbital_lossfn": orbital_lossfn,
            "v_delta_lossfn": v_delta_lossfn,
            "v_delta_r_lossfn": v_delta_r_lossfn,
            "phi_lossfn": phi_lossfn,
            "phi_align_lossfn": phi_align_lossfn,
            "band_lossfn": band_lossfn,
            "bandgap_lossfn": bandgap_lossfn,
            "density_m_lossfn": density_m_lossfn,
            "energy_per_atom": self.energy_per_atom,
            "vd_divide_by_nlocal": vd_divide_by_nlocal,
            "vd_masked_loss": vd_masked_loss,
            "vd_masked_S_threshold": vd_masked_S_threshold,
            "vd_masked_H_threshold": vd_masked_H_threshold,
            "vd_masked_width": vd_masked_width,
            "use_safe_eigh": use_safe_eigh,
        }
        self.terms = build_descriptor_property_terms(self.objective_args)

    def compute_losses(self, model, batch):
        batch = self.batch_builder.build_batch(batch)
        assert isinstance(batch, TaskBatch)

        device = next(model.parameters()).device
        batch = batch.to_device(device, complex_cpu_context_keys=("phialpha",))
        # Only request properties whose physics context the current batch can
        # satisfy. The term loop below will then naturally skip any term whose
        # prediction never lands in ``predictions``. This prevents a single
        # missing chain-rule helper (e.g. ``vdr_precalc`` / ``grad_vx`` that
        # the SCF backend was not configured to emit) from killing the entire
        # training run; the user gets a one-shot warning per dropped term.
        full_requested = self._requested_properties()
        requested_properties = self.property_engine.available_properties(
            full_requested, batch.context
        )
        dropped = full_requested - requested_properties
        if dropped:
            self._warn_dropped_properties(dropped)
        derivative_spec = self.property_engine.required_model_derivatives(requested_properties)

        # R1 + R2: call the model in dict-in / dict-out form, then apply
        # interface-side reducers to obtain the supervision-ready primary
        # output. Autograd of the reduced primary output w.r.t. the named
        # model input(s) is owned by the objective adapter — the model body
        # stays free of "what the primary output is reduced to" logic.
        model_input = self._lookup_primary_input(batch)
        needs_input_grad = bool(derivative_spec.get("input"))
        raw_outputs, derivatives_hint = self._call_model_forward(
            model, model_input, batch, derivative_spec
        )
        model_outputs = self._apply_reducers(raw_outputs, model_meta=model)
        input_grad = self._compute_input_grad(
            model_outputs.get("primary_output"), model_input, needs_input_grad
        ) if derivatives_hint is None else derivatives_hint.get("input")
        model_derivatives = {"input": input_grad}

        calc_context = dict(batch.context)
        predictions = self.property_engine.get_many(
            requested_properties,
            model_outputs=model_outputs,
            model_derivatives=model_derivatives,
            context=calc_context,
        )
        if self.g_penalty > 0 and input_grad is not None and "eg0" in batch.context:
            eg_base, gveg = batch.context["eg0"], batch.context["gveg"]
            predictions["grad_total"] = torch.einsum("...apg,...ap->...g", gveg, input_grad) + eg_base
        if self.d_factor > 0 and input_grad is not None and "gldv" in batch.context:
            predictions["density_regularizer"] = (batch.context["gldv"] * input_grad).mean(0).sum()

        zero_template = predictions.get(self.primary_property) if self.primary_property else None
        if zero_template is None:
            for value in predictions.values():
                if torch.is_tensor(value):
                    zero_template = value
                    break
        if zero_template is None:
            zero_template = torch.zeros((), dtype=torch.float64, device=device)
        total_loss = zero_template.new_tensor(0.0)
        loss_terms = []
        for term in self.terms:
            required_predictions = term.required_prediction_keys()
            required_targets = term.required_target_keys()
            if any(key not in predictions for key in required_predictions):
                continue
            if any(key not in batch.targets for key in required_targets):
                continue
            term_loss = term.compute_loss(predictions, batch.targets, batch)
            total_loss = total_loss + term_loss
            loss_terms.append(term_loss)

        loss_terms.append(total_loss)
        return loss_terms

    def compute_metrics(self, model, batch):
        return self.compute_losses(model, batch)

    def _lookup_primary_input(self, batch):
        """Resolve the primary model input tensor by recipe-declared name."""

        model_inputs = batch.model_inputs
        if self.primary_input in model_inputs:
            return model_inputs[self.primary_input]
        # Fall back to the legacy ``"descriptor"`` slot so old recipes that
        # don't declare ``primary_input`` still work.
        if "descriptor" in model_inputs:
            return model_inputs["descriptor"]
        raise KeyError(
            f"Primary model input '{self.primary_input}' not found in batch.model_inputs "
            f"(available: {tuple(model_inputs)})"
        )

    def _call_model_forward(self, model, model_input, batch, derivative_spec):
        """Call ``model.forward`` and normalize the result to a dict.

        Returns ``(raw_outputs, derivatives_hint)``. If the model is one
        of the legacy implementations still providing
        ``forward_with_derivatives`` and the framework wants to use that
        autograd path, ``derivatives_hint`` carries the gradients
        produced by the model; otherwise it is ``None`` and the objective
        adapter computes derivatives itself.
        """

        # Models with the new contract: ``forward(model_inputs_dict_or_tensor)``
        # returns a tensor (single-output) or a dict (multi-output). Wrap a bare
        # tensor under the recipe-declared primary output name.
        if isinstance(model_input, torch.Tensor):
            model_input.requires_grad_(bool(derivative_spec.get("input")))
        outputs = model.forward(model_input)
        if isinstance(outputs, dict):
            raw_outputs = dict(outputs)
        else:
            raw_outputs = {"primary_output": outputs}
        return raw_outputs, None

    def _apply_reducers(self, raw_outputs, *, model_meta):
        """Apply per-output reducers, keying the primary output as 'primary_output'."""

        reduced = {}
        primary_reducer = self._primary_reducer
        primary_tensor = raw_outputs.get("primary_output")
        if primary_tensor is None and len(raw_outputs) == 1:
            # Single-output model — treat its sole entry as the primary.
            primary_tensor = next(iter(raw_outputs.values()))
        if primary_tensor is not None:
            reduced["primary_output"] = primary_reducer(primary_tensor, model_meta=model_meta)
        # Carry through any other named outputs with their own reducers (default
        # to Identity when none configured).
        for name, tensor in raw_outputs.items():
            if name == "primary_output":
                continue
            reducer = self.output_reducers.get(name, Identity())
            reduced[name] = reducer(tensor, model_meta=model_meta)
        return reduced

    @staticmethod
    def _compute_input_grad(primary_scalar, model_input, needs_grad):
        if not needs_grad or primary_scalar is None or not isinstance(model_input, torch.Tensor):
            return None
        [grad] = torch.autograd.grad(
            primary_scalar,
            model_input,
            grad_outputs=torch.ones_like(primary_scalar),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )
        return grad

    def _requested_properties(self):
        requested = set()
        if self.primary_property:
            requested.add(self.primary_property)
        for term in self.terms:
            requested.update(term.required_property_keys())
        requested.discard("grad_total")
        requested.discard("density_regularizer")
        return requested

    def _warn_dropped_properties(self, dropped_properties):
        """Print one warning per supervision property that lacks batch context."""

        if not hasattr(self, "_warned_dropped"):
            self._warned_dropped = set()
        new = set(dropped_properties) - self._warned_dropped
        if not new:
            return
        self._warned_dropped.update(new)
        for name in sorted(new):
            print(
                f"# [warn] dropping supervision property '{name}': the current batch "
                "context does not contain the inputs the property scheme needs. "
                "Check that the SCF backend emitted the corresponding chain-rule helpers "
                "(e.g. grad_vx for force, vdr_precalc for V_delta(R))."
            )

    def print_head(self, name, data_keys, align_len=20):
        data_keys = self._normalize_field_keys(data_keys)
        info = f"{name}_energy".rjust(align_len)
        if self.g_penalty > 0 and "eg0" in data_keys:
            info += f"{name}_grad".rjust(align_len)
        if self.f_factor > 0 and "force" in data_keys:
            info += f"{name}_force".rjust(align_len)
        if self.s_factor > 0 and "stress" in data_keys:
            info += f"{name}_stress".rjust(align_len)
        if self.o_factor > 0 and "orbital" in data_keys:
            info += f"{name}_orbital".rjust(align_len)
        if self.vd_factor > 0 and "v_delta" in data_keys:
            info += f"{name}_v_delta".rjust(align_len)
        if self.vdr_factor > 0 and "vdr" in data_keys:
            info += f"{name}_v_delta_r".rjust(align_len)
        if self.phi_factor > 0 and "phi" in data_keys:
            info += f"{name}_phi".rjust(align_len)
        if self.band_factor > 0 and "band" in data_keys:
            info += f"{name}_band".rjust(align_len)
        if self.bandgap_factor > 0 and "band" in data_keys:
            info += f"{name}_bandgap".rjust(align_len)
        if self.density_m_factor > 0 and "phi" in data_keys:
            info += f"{name}_dm".rjust(align_len)
        if self.phi_align_factor > 0 and "phi" in data_keys and "band" in data_keys:
            info += f"{name}_phi_align".rjust(align_len)
        if self.d_factor > 0 and "gldv" in data_keys:
            info += f"{name}_density".rjust(align_len)
        print(info, end="")

    @staticmethod
    def _normalize_field_keys(data_keys):
        aliases = {
            "eig": "descriptor",
            "lb_e": "energy",
            "lb_f": "force",
            "lb_s": "stress",
            "lb_o": "orbital",
            "lb_vd": "v_delta",
            "lb_vdr": "vdr",
            "lb_phi": "phi",
            "lb_band": "band",
        }
        return {aliases.get(key, key) for key in data_keys}
