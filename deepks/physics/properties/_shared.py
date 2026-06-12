"""Internal shared helpers for property implementations."""

import torch

from deepks.io.transforms.linalg import generalized_eigh, eigh_wrapper


def solve_band_phi(context, vd_pred, use_safe_eigh=False):
    h_total = context["h_base"] + vd_pred
    if "trans_matrix" in context:
        return generalized_eigh(h_total, context["trans_matrix"], use_safe_eigh)
    return eigh_wrapper(h_total)
