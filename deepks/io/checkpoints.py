"""Checkpoint serialization helpers owned by the io layer."""

import torch


def save_model_checkpoint(filename, payload):
    torch.save(payload, filename)


def load_model_checkpoint(filename):
    return torch.load(filename, map_location="cpu", weights_only=False)


def load_compiled_or_checkpoint(filename):
    try:
        return torch.jit.load(filename)
    except RuntimeError:
        return load_model_checkpoint(filename)


__all__ = [
    "save_model_checkpoint",
    "load_model_checkpoint",
    "load_compiled_or_checkpoint",
]
