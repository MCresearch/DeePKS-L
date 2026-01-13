import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from deepks.default import DEVICE
# import psutil
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")


def fit_elem_const(g_reader, test_reader=None, elem_table=None, ridge_alpha=0.):
    if elem_table is None:
        elem_table = g_reader.compute_elem_const(ridge_alpha)
    elem_list, elem_const = elem_table
    g_reader.collect_elems(elem_list)
    g_reader.subtract_elem_const(elem_const)
    if test_reader is not None:
        test_reader.collect_elems(elem_list)
        test_reader.subtract_elem_const(elem_const)
    return elem_table

def preprocess(model, g_reader, 
                preshift=True, prescale=False, prescale_sqrt=False, prescale_clip=0,
                prefit=True, prefit_ridge=10, prefit_trainable=False):
    """
    Preprocess data for training, including normalization and prefitting.
    preshift: whether to shift the input data to zero mean
    prescale: whether to scale the input data to unit variance
    prescale_sqrt: whether to take the square root of the scaled data
    prescale_clip: whether to clip the scaled data
    prefit: whether to prefit the data
    prefit_ridge: the ridge regression alpha for prefitting
    """
    shift = model.input_shift.cpu().detach().numpy()
    scale = model.input_scale.cpu().detach().numpy()
    symm_sec = model.shell_sec # will be None if no embedding
    prefit_trainable = prefit_trainable and symm_sec is None # no embedding
    if preshift or prescale:
        davg, dstd = g_reader.compute_data_stat(symm_sec)
        if preshift: 
            shift = davg
        if prescale: 
            scale = dstd
            if prescale_sqrt: 
                scale = np.sqrt(scale)
            if prescale_clip: 
                scale = scale.clip(prescale_clip)
        model.set_normalization(shift, scale)
    if prefit:
        weight, bias = g_reader.compute_prefitting(
            shift=shift, scale=scale, 
            ridge_alpha=prefit_ridge, symm_sections=symm_sec)
        model.set_prefitting(weight, bias, trainable=prefit_trainable)

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

## The following four functions are used only in Evaluator class
def cal_v_delta(gev,gevdm,phialpha,device=DEVICE):
    # process = psutil.Process(os.getpid())
    # before_memory_usage = process.memory_info().rss

    mmax=phialpha.size(-1)
    lmax=int((mmax-1)/2)
    n=int(phialpha.size(2)/(lmax+1)) # number of orbitals in each l

    dtype=phialpha.dtype
    gev=gev.to(dtype)
    gevdm=gevdm.to(dtype)

    n_batch=phialpha.size(0)
    nks=phialpha.size(-3)
    nlocal=phialpha.size(-2)
    v_delta=torch.zeros([n_batch,nks,nlocal,nlocal],dtype=dtype,device=device)
    for l in range(lmax+1):
        gevdm_l=gevdm[...,n*l:n*(l+1),:2*l+1,:2*l+1,:2*l+1]
        gev_l=gev[...,n*l**2:n*(l+1)**2]
        # print(gevdm_l.shape,gev_l.shape)

        gev_l=gev_l.view(gev_l.size(0),gev_l.size(1),n,2*l+1)
        #gev_l=gev_l.permute(0,2,1,3)
        # print(gev_l.shape)

        temp_1=torch.einsum("...v,...vmn->...mn", gev_l, gevdm_l)
        # print(temp_1.shape)
        del gev_l, gevdm_l

        phialpha_l=phialpha[...,n*l:n*(l+1),:,:,:2*l+1]
        phialpha_l = phialpha_l.to(device)
        # print(phialpha_l.shape)
        temp_2 = torch.einsum("...mn,...kxn->...kxm",temp_1, phialpha_l)
        # print(temp_2.shape)
        del temp_1

        vdp_nl=torch.einsum("...alkxm,...alkym->...kxy",temp_2, phialpha_l.conj())
        #vdp_nl=torch.einsum("...alkxy->kxy",temp_3)
        # print(vdp_nl.shape)
        del temp_2, phialpha_l

        v_delta+=vdp_nl
        # print(v_delta.shape)
        del vdp_nl

    # after_memory_usage = process.memory_info().rss
    # memory_growth = after_memory_usage - before_memory_usage
    # print(f"Memory growth during cal vdp: {memory_growth / 1024 /1024 } MB")

    # print("v_delta.shape",v_delta.shape)
    return v_delta

def get_density_matrix(phi,density_m_occ):
    phi_occ=phi[...,:density_m_occ]
    batch_size,nks,nlocal,nocc=phi_occ.size()
    phi_occ=phi_occ.view(batch_size*nks,nlocal,nocc)

    #batch matrix multiplication, phi_occ@phi_occ.T
    density_m = torch.bmm(phi_occ, phi_occ.transpose(-1,-2))

    #reshape to batch_size,nks,nlocal,nlocal
    density_m=density_m.view(batch_size,nks,nlocal,nlocal)
    phi_occ=phi_occ.view(batch_size,nks,nlocal,nocc)

    return density_m

# use every phi_pred and -1*phi_pred to compare with corresponding phi_label, given that phi can have coeficient freedom of +-1 (for gamma only)
def cal_phi_loss(phi_pred,phi_label,phi_occ):
    occ_phi_pred=phi_pred[...,:phi_occ].clone()
    occ_phi_label=phi_label[...,:phi_occ].clone()
    # print("occ_phi.shape",occ_phi_pred.shape,occ_phi_label.shape)
    # just mean reduction
    loss_1=((occ_phi_label-occ_phi_pred)**2).mean(-2) # mean for every component of each phi
    loss_2=((occ_phi_label-(-1)*occ_phi_pred)**2).mean(-2)
    loss=torch.stack([loss_1,loss_2],dim=-1)
    loss=loss.min(dim=-1)[0] # pick min for every phi
    loss=loss.mean()
    #print("loss.shape:",loss.shape)
    return loss

def cal_vd_masked_loss_hs(H_pred, H_label, S_matrix, S_threshold=1e-6, H_threshold=1e-6):
    """
    Computes the Mean Squared Error of Hamiltonian elements filtered by 
    the Overlap matrix and Hamiltonian matrix magnitude. Supports 4D tensors (nframe, nks, nlocal, nlocal).
    
    Args:
        H_pred (torch.Tensor): Predicted Hamiltonian.
        H_label (torch.Tensor): Label Hamiltonian.
        S_matrix (torch.Tensor): Overlap matrix used for masking.
        S_threshold (float): Threshold to ignore insignificant interactions.
        H_threshold (float): Threshold to ignore insignificant interactions.
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    with torch.no_grad():
        # Generate mask where |S| > threshold or |H| > threshold. 
        # Diagonals (S_ii=1) are naturally included if threshold < 1.
        mask = ( (torch.abs(S_matrix) > S_threshold) | (torch.abs(H_label) > H_threshold) ).to(device=H_pred.device, dtype=H_pred.dtype)

    # Compute element-wise squared difference
    diff_sq = (H_pred - H_label) ** 2
    
    # Apply mask and normalize by the number of active elements in the mask
    # to maintain loss consistency across different system sizes or k-points.
    masked_sum = torch.sum(diff_sq * mask)
    active_elements = torch.sum(mask)
    
    return masked_sum / (active_elements + 1e-12)

def cal_vd_masked_loss_width(H_pred, H_label, width=1):
    """
    Computes the Mean Squared Error of Hamiltonian elements filtered by width .
    Args:
        H_pred (torch.Tensor): Predicted Hamiltonian.
        H_label (torch.Tensor): Label Hamiltonian.
        width (int): Width of the mask
    Returns:
        torch.Tensor: Scalar loss value.
    """
    with torch.no_grad():
        nlocal = H_pred.size(-1)
        i = torch.arange(nlocal, device=H_pred.device).view(nlocal, 1)  # (nlocal, 1)
        j = torch.arange(nlocal, device=H_pred.device).view(1, nlocal)  # (1, nlocal)
        
        # calculate the shortest distance on the circle
        diff = torch.abs(i - j)
        dist = torch.minimum(diff, nlocal - diff)
        
        # mask is 1 if distance < width, 0 otherwise
        mask = (dist < width).to(dtype=H_pred.dtype)
    
    diff_sq = (H_pred - H_label) ** 2
    masked_sum = torch.sum(diff_sq * mask)
    active_elements = torch.sum(mask)

    return masked_sum / (active_elements + 1e-12)

def cal_bandgap(band, occ):
    return band[...,occ] - band[...,occ-1]

def get_occ_func(occ):
    # print("type:",type(occ))
    if isinstance(occ, int):
        def get_occ(natom):
            return occ  
    elif isinstance(occ, dict):
        new_occ={int(natom):int(n_occ) for (natom,n_occ) in occ.items()}
        def get_occ(natom):
            return new_occ[natom] 
    return get_occ

class SafeEigh(torch.autograd.Function):
    """
    A custom autograd function for eigendecomposition of real symmetric matrices.
    It handles degenerate eigenvalues by masking out the infinite gradients 
    caused by the term 1/(lambda_i - lambda_j) when lambda_i approx lambda_j.
    
    Reference: 
    Derivatives of Partial Eigendecomposition of a Real Symmetric Matrix 
    for Degenerate Cases (Kasim et al., 2020), Equation (27).
    """
    
    @staticmethod
    def forward(ctx, a):
        """
        Forward pass: Standard eigendecomposition.
        
        Args:
            a: Input symmetric matrix. Shape: (..., N, N), supports batching.
        Returns:
            e: Eigenvalues. Shape: (..., N).
            v: Eigenvectors. Shape: (..., N, N).
        """
        # Ensure the input is float/complex as required by eigh
        # Note: 'U' (Upper) or 'L' (Lower) doesn't matter much for valid symmetric inputs
        e, v = torch.linalg.eigh(a)
        
        # Save tensors for the backward pass
        ctx.save_for_backward(e, v)
        return e, v

    @staticmethod
    def backward(ctx, grad_e, grad_v):
        """
        Backward pass: Computes gradient with respect to input matrix 'a'.
        
        This implementation specifically handles the degeneracy issue where 
        eigenvalues are identical or very close, which would normally cause 
        NaNs or Infs in the gradient of eigenvectors.
        """
        e, v = ctx.saved_tensors
        
        # 1. Handle cases where gradients might be None 
        # (e.g., if eigenvalues or eigenvectors are not used in the loss function)
        if grad_e is None:
            grad_e = torch.zeros_like(e)
        if grad_v is None:
            grad_v = torch.zeros_like(v)

        # 2. Construct the pairwise difference matrix of eigenvalues
        # Shape of e: (Batch, N)
        # Use unsqueeze to broadcast: (Batch, N, 1) - (Batch, 1, N) -> (Batch, N, N)
        # e_diff[..., i, j] = e[..., i] - e[..., j] (column - row)
        e_diff = e.unsqueeze(-2) - e.unsqueeze(-1)

        # 3. Handle Degeneracy (Masking)
        # Define a small threshold to detect degeneracy
        epsilon = 1e-8 
        
        # Create a mask where |lambda_i - lambda_j| > epsilon
        mask = torch.abs(e_diff) > epsilon
        
        # Construct the F matrix: F_ij = 1 / (lambda_j - lambda_i)
        # Note: We use the transposed definition implicit in the matrix formula below.
        # Here we initialize f_matrix with zeros, effectively ignoring degenerate terms.
        f_matrix = torch.zeros_like(e_diff)
        
        # Only compute division for non-degenerate pairs
        # This prevents division by zero and corresponds to setting the gradient 
        # contribution of degenerate subspaces to zero (Gauge Invariance).
        f_matrix[mask] = 1.0 / e_diff[mask]

        # 4. Compute the gradient w.r.t. the input matrix 'a'
        # Formula: grad_a = v @ (diag(grad_e) + F * (v^T @ grad_v)) @ v^T
        
        # Projection of gradients onto the eigenvector basis: v^T @ grad_v
        # transpose(-2, -1) handles the last two dimensions for batch processing
        vt = v.transpose(-2, -1)
        v_t_grad_v = vt @ grad_v
        
        # The middle term: diag(grad_e) + F * (v^T @ grad_v)
        # torch.diag_embed creates a diagonal matrix from the eigenvalue gradients
        # f_matrix * v_t_grad_v performs element-wise multiplication (Hadamard product)
        mid_term = torch.diag_embed(grad_e) + f_matrix * v_t_grad_v
        
        # Transform back to the original basis
        grad_a = v @ mid_term @ vt
        
        # 5. Enforce symmetry
        # Since the input 'a' is symmetric, its gradient must also be symmetric.
        grad_a = 0.5 * (grad_a + grad_a.transpose(-2, -1))
        
        return grad_a

# Wrapper function for easy usage
def safe_eigh(input_tensor):
    return SafeEigh.apply(input_tensor)