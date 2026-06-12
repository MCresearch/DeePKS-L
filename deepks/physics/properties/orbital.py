"""Orbital property helpers."""

import torch


def orbital_from_descriptor_gradient(input_grad, op, orbital_shape):
    orbital_projection = op.contiguous().view(
        op.shape[0],
        orbital_shape[1],
        orbital_shape[2],
        op.shape[-2],
        op.shape[-1],
    )
    return torch.einsum("...kiap,...ap->...ki", orbital_projection, input_grad)
