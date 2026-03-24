"""Linear algebra helpers shared by reader/evaluator modules."""

import torch
from deepks.ml.utils import safe_eigh

def eigh_wrapper(a, use_safe_eigh=False):
    """
    Wrapper for eigendecomposition that supports safe gradients for degenerate cases.
    Args:
        a: Symmetric/Hermitian matrix.
        use_safe_eigh: If True, uses SafeEigh to prevent NaN gradients.
    """
    if use_safe_eigh:
        return safe_eigh(a)
    else:
        return torch.linalg.eigh(a, UPLO='U')

def generalized_eigh(h,trans_matrix,use_safe_eigh=False):
    symm_h=trans_matrix.mT @ h @ trans_matrix
    e,v=eigh_wrapper(symm_h, use_safe_eigh=use_safe_eigh)
    phi=trans_matrix @ v 
    return e,phi
