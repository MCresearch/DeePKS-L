"""Linear algebra helpers shared by reader/evaluator modules."""

import torch


def generalized_eigh(h, l_inv):
    symm_h = l_inv @ h @ l_inv.mT
    e, v = torch.linalg.eigh(symm_h)
    phi = l_inv.mT @ v
    return e, phi
