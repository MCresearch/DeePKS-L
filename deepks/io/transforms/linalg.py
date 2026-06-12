"""Linear algebra helpers shared by reader/evaluator modules."""

import torch


class SafeEigh(torch.autograd.Function):
    """Stable eigendecomposition for symmetric matrices with degenerate spectra."""

    @staticmethod
    def forward(ctx, a):
        e, v = torch.linalg.eigh(a)
        ctx.save_for_backward(e, v)
        return e, v

    @staticmethod
    def backward(ctx, grad_e, grad_v):
        e, v = ctx.saved_tensors
        if grad_e is None:
            grad_e = torch.zeros_like(e)
        if grad_v is None:
            grad_v = torch.zeros_like(v)

        e_diff = e.unsqueeze(-2) - e.unsqueeze(-1)
        epsilon = 1e-8
        mask = torch.abs(e_diff) > epsilon
        f_matrix = torch.zeros_like(e_diff)
        f_matrix[mask] = 1.0 / e_diff[mask]

        vt = v.transpose(-2, -1)
        v_t_grad_v = vt @ grad_v
        mid_term = torch.diag_embed(grad_e) + f_matrix * v_t_grad_v
        grad_a = v @ mid_term @ vt
        grad_a = 0.5 * (grad_a + grad_a.transpose(-2, -1))
        return grad_a


def safe_eigh(input_tensor):
    return SafeEigh.apply(input_tensor)

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
