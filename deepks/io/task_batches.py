"""Sample <-> TaskBatch adapters shared by readers and interface recipes."""

import re
from typing import Any, Dict, Mapping, Optional

from deepks.io.batch import TaskBatch


_TARGET_KEY_MAP = {
    "lb_e": "energy",
    "lb_f": "force",
    "lb_s": "stress",
    "lb_o": "orbital",
    "lb_vd": "v_delta",
    "lb_vdr": "vdr",
    "lb_phi": "phi",
    "lb_band": "band",
}

_DEFAULT_INPUT_FIELD_MAPPING = {
    "eig": "descriptor",
}

_REVERSE_TARGET_KEY_MAP = {value: key for key, value in _TARGET_KEY_MAP.items()}
_REVERSE_INPUT_KEY_MAP = {value: key for key, value in _DEFAULT_INPUT_FIELD_MAPPING.items()}
_HAM_LEVEL_PATTERN = re.compile(r"^lb_ham_level_(\d+)$")


def sample_to_task_batch(
    sample: Dict[str, Any],
    *,
    input_field_mapping: Optional[Mapping[str, str]] = None,
) -> TaskBatch:
    """Convert a reader-produced sample dict into a :class:`TaskBatch`.

    R3: ``input_field_mapping`` declares which sample keys become
    ``TaskBatch.model_inputs`` entries (and under what model-input names).
    Recipes for multi-input models such as a graph network can therefore
    route e.g. ``{"force": "nodes", "coord": "coords"}``. The default
    mapping ``{"eig": "descriptor"}`` preserves backward compatibility
    for descriptor-energy recipes that don't override it.

    Sample keys not matched by ``input_field_mapping``, not a recognized
    label / hamiltonian field, fall through to ``TaskBatch.context``.
    """

    mapping = dict(_DEFAULT_INPUT_FIELD_MAPPING if input_field_mapping is None else input_field_mapping)

    model_inputs: Dict[str, Any] = {}
    targets: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

    for sample_key, model_input_key in mapping.items():
        if sample_key in sample:
            model_inputs[model_input_key] = sample[sample_key]

    hamiltonian_levels = []
    handled_ham_keys = set()
    for key, value in sample.items():
        matched = _HAM_LEVEL_PATTERN.match(key)
        if matched is None:
            continue
        index = int(matched.group(1))
        while len(hamiltonian_levels) <= index:
            hamiltonian_levels.append(None)
        hamiltonian_levels[index] = value
        handled_ham_keys.add(key)
    if hamiltonian_levels:
        targets["hamiltonian_levels"] = hamiltonian_levels

    if "lb_hamiltonian" in sample:
        targets["hamiltonian"] = sample["lb_hamiltonian"]
    if "lb_csr_hamiltonian" in sample:
        targets["csr_hamiltonian"] = sample["lb_csr_hamiltonian"]

    consumed_sample_keys = set(mapping) | {"lb_hamiltonian", "lb_csr_hamiltonian"} | handled_ham_keys
    for key, value in sample.items():
        if key in consumed_sample_keys:
            continue
        target_name = _TARGET_KEY_MAP.get(key)
        if target_name is not None:
            targets[target_name] = value
            if target_name == "orbital":
                context["orbital_shape"] = tuple(value.shape)
        else:
            context[key] = value

    # ``group_key`` is used by the trainer for grouped loss aggregation; default
    # to the first model_input's per-atom axis if available, falling back to the
    # legacy ``sample["eig"]`` path so existing recipes behave identically.
    group_key_source = None
    if model_inputs:
        first_input = next(iter(model_inputs.values()))
        if hasattr(first_input, "shape") and first_input.ndim >= 2:
            group_key_source = first_input.shape[1]
    elif "eig" in sample:
        group_key_source = sample["eig"].shape[1]

    return TaskBatch(
        model_inputs=model_inputs,
        targets=targets,
        context=context,
        meta={
            "group_key": group_key_source,
            "source_keys": tuple(sample.keys()),
            "display_keys": tuple(sample.keys()),
            "normalized_keys": tuple(model_inputs) + tuple(targets) + tuple(context),
        },
    )


def task_batch_to_sample(batch: TaskBatch) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}
    for key, value in batch.model_inputs.items():
        sample[_REVERSE_INPUT_KEY_MAP.get(key, key)] = value
    for key, value in batch.targets.items():
        if key == "hamiltonian_levels":
            for index, level_value in enumerate(value):
                sample[f"lb_ham_level_{index}"] = level_value
            continue
        if key == "hamiltonian":
            sample["lb_hamiltonian"] = value
            continue
        if key == "csr_hamiltonian":
            sample["lb_csr_hamiltonian"] = value
            continue
        sample[_REVERSE_TARGET_KEY_MAP.get(key, key)] = value
    sample.update(batch.context)
    return sample


__all__ = ["sample_to_task_batch", "task_batch_to_sample"]
