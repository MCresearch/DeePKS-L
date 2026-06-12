"""Force property helpers."""

import torch


def force_from_descriptor_gradient(input_grad, gvx):
    return -torch.einsum("...bxap,...ap->...bx", gvx, input_grad)
