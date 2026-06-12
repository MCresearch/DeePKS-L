import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")


def make_loss(cap=None, shrink=None, reduction="mean"):
    def loss_fn(input, target):
        diff = target - input
        if shrink and shrink > 0:
            diff = F.softshrink(diff, shrink)
        sqdf = torch.abs(diff)**2 # use abs to avoid complex number
        if cap and cap > 0: # SmoothL2 loss
            abdf = diff.abs()
            sqdf = torch.where(abdf < cap, sqdf, cap * (2*abdf - cap))
        if reduction is None or reduction.lower() == "none":
            return sqdf
        elif reduction.lower() == "mean":
            return sqdf.mean()
        elif reduction.lower() == "sum":
            return sqdf.sum()
        elif reduction.lower() in ("batch", "bmean"):
            return sqdf.sum() / sqdf.shape[0]
        else:
            raise ValueError(f"{reduction} is not a valid reduction type")
    return loss_fn
