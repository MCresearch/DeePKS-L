import numpy as np


__all__ = [
    "build_group_sampling_cache",
    "build_system_probabilities",
    "reader_shape",
]


def _normalize_probabilities(weights, *, context):
    probs = np.asarray(weights, dtype=float)
    if probs.size == 0:
        raise ValueError(f"{context}: empty probability weights")
    total = probs.sum()
    if total <= 0:
        raise ValueError(f"{context}: non-positive probability sum")
    return probs / total


def build_system_probabilities(frame_counts):
    return _normalize_probabilities(frame_counts, context="system sampling")


def reader_shape(reader):
    return (reader.natm, getattr(reader, "neg", None))


def build_group_sampling_cache(readers):
    if not readers:
        raise ValueError("group sampling: readers must not be empty")

    group_dict = {}
    for reader in readers:
        shape = reader_shape(reader)
        group_dict.setdefault(shape, []).append(reader)

    group_weights = np.asarray(
        [sum(reader.nframes for reader in group_readers) for group_readers in group_dict.values()],
        dtype=float,
    )
    group_probs = _normalize_probabilities(group_weights, context="group sampling")

    group_shapes = tuple(group_dict.keys())
    group_prob = {shape: float(prob) for shape, prob in zip(group_shapes, group_probs)}

    batch_prob_raw = {
        shape: np.asarray([reader.nframes / reader.batch_size for reader in group_readers], dtype=float)
        for shape, group_readers in group_dict.items()
    }
    batch_prob = {
        shape: _normalize_probabilities(raw_prob, context=f"batch sampling[{shape}]")
        for shape, raw_prob in batch_prob_raw.items()
    }

    return {
        "group_dict": group_dict,
        "group_prob": group_prob,
        "batch_prob_raw": batch_prob_raw,
        "batch_prob": batch_prob,
        "group_shapes": group_shapes,
        "group_probs": group_probs,
    }
