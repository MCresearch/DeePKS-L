"""Stress property helpers."""

import torch


def stress_from_descriptor_gradient(input_grad, gvepsl):
    return torch.einsum("...iap,...ap->...i", gvepsl, input_grad)
