import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from deepks.physics.defaults import TYPE_NAME
from deepks.physics.backends.abacus.utils import R2iR
try:
    from pyabacus import ModuleBase as base
    from pyabacus import ModuleNAO as nao
except ImportError:
    base = None
    nao = None
import gc
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

def loss_hr(input, target):
    diff = target - input
    sqdf = torch.abs(diff)**2 
    R_range = sqdf.shape[1]
    nlocal = sqdf.shape[-1]
    return sqdf.sum() / R_range / nlocal / sqdf.shape[0] # original loss
    # indices = torch.arange(R_range, device=sqdf.device)
    # ix, iy, iz = torch.meshgrid(indices, indices, indices, indexing='ij')
    
    # x = (ix + 1) // 2
    # y = (iy + 1) // 2
    # z = (iz + 1) // 2
    # r_squared = x.float()**2 + y.float()**2 + z.float()**2
    
    # # # try 1:
    # # k = 1.0
    # # factor = torch.exp(-k * r_squared)
    # # factor = factor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Adjust dimensions for broadcasting

    # # try 2:
    # r = torch.sqrt(r_squared)
    # spatial_factor = 1.0 / (r * 0.3 + 1.0)
    # spatial_factor = spatial_factor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # i, j = torch.meshgrid(torch.arange(nlocal, device=sqdf.device), 
    #                      torch.arange(nlocal, device=sqdf.device), indexing='ij')
    # diag_dist = torch.abs(i - j).float()
    # matrix_factor = 1.0 / (1.0 + 3.0 * diag_dist)
    # matrix_factor = matrix_factor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # target_abs = torch.abs(target)
    # value_weight = 1.0 + 2.0 * torch.sigmoid(target_abs * 10)

    # factor = spatial_factor * matrix_factor * value_weight

    # sqdf_weighted = sqdf * factor
    # torch.set_printoptions(threshold=sys.maxsize)
    # # print(sqdf_weighted[:,0,0,0,:,:])
    # # return sqdf_weighted.mean()
    # return sqdf_weighted.sum() / R_range / nlocal / sqdf.shape[0]

## The following four functions are used only in Evaluator class
def cal_v_delta(gev,gevdm,phialpha,device="cpu"):
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

def cal_nb_overlap(types, atoms, box, orb, alpha, integrator, nlocal):
    device = atoms.device
    nframes, natoms = atoms.shape[0], atoms.shape[1]
    lmax_alpha = alpha.lmax(0)
    nzeta_alpha = alpha.nzeta(0, 0) # same n for s,p,d,f,... in alpha

    nlocal2idx, idx2nlocal = nlocalVSidx(types.to("cpu"), orb, nlocal)
    neighbors, nn_range = find_neighbor_pair(box.to("cpu"), types.to("cpu"), atoms.to("cpu"), orb, alpha, idx2nlocal)
    nnmax = nn_range.shape[2]

    overlap = torch.zeros((nframes, natoms, nnmax, nlocal, nzeta_alpha * (lmax_alpha + 1)**2), dtype=torch.float64, device=device)
    for iframe in range(nframes):
        for iat in range(natoms):
            for inn, (ibt1, Rx1, Ry1, Rz1, dist1) in enumerate(neighbors[iframe][iat]):
                for ix in range(nn_range[iframe][iat][inn][0], nn_range[iframe][iat][inn][1]):
                    ibt, t1, n1, l1, M1 = nlocal2idx[ix]
                    if M1 % 2 == 0:
                        m1 = -M1 // 2
                    else:
                        m1 = (M1 + 1) // 2
                    overlap_tmp = integrator.snap(t1, l1, n1, m1, 0, dist1, False)
                    overlap[iframe, iat, inn, ix, :] = torch.tensor(overlap_tmp, device=device).reshape(-1)

    vecs = torch.zeros((nframes, natoms, nnmax, 3), dtype=torch.int64, device=device)
    for iframe in range(nframes):
        for iat in range(natoms):
            for inn in range(len(neighbors[iframe][iat])):
                vecs[iframe, iat, inn] = torch.tensor(neighbors[iframe][iat][inn][1:4], device=device)
    iR_mat = R2iR(vecs.unsqueeze(2) - vecs.unsqueeze(3))

    data_shape = [nzeta_alpha, lmax_alpha]
    return overlap, iR_mat, data_shape

def cal_vdr(gedm, overlap, iR_mat, vdr_label, device="cpu"):
    nframes, iRmax, nlocal = vdr_label.shape[0], vdr_label.shape[1], vdr_label.shape[-1]
    vdr_pred = torch.zeros_like(vdr_label, device=device)
    # natoms = 2
    # nnmax = 29
    # for iframe in range(nframes):
    #     for iat in range(natoms):
    #         for inn1 in range(nnmax):
    #             for inn2 in range(nnmax):
    #                 iR = iR_mat[iframe, iat, inn1, inn2]
    #                 if iR.max().item() >= iRmax:
    #                     continue
    #                 overlap1 = overlap[iframe, iat, inn1]
    #                 overlap2 = overlap[iframe, iat, inn2]
    #                 result_nl = overlap1 @ gedm[iframe, iat] @ overlap2.T
    #                 vdr_pred[iframe, iR[0], iR[1], iR[2], :, :].add_(result_nl)

    # Following is a faster implementation using tensor operations
    valid_mask = (iR_mat.max(dim=-1)[0] < iRmax)  # [nframes, natoms, nnmax, nnmax]
    valid_indices = torch.nonzero(valid_mask)  # [N_valid, 4]
    if valid_indices.size(0) == 0:
        pass
    else:
        # result_all shape: [nframes, natoms, nnmax, nnmax, nlocal, nlocal]
        # check cuda memery usage before einsum
        # torch.cuda.empty_cache()
        # gc.collect()
        # print(f"当前显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        # print(f"最大显存使用: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        # torch.cuda.reset_peak_memory_stats()
        # # print the memery of overlap and gedm
        # print(f"overlap memory: {overlap.element_size() * overlap.nelement() / 1024**3:.2f} GB")
        # print(f"gedm memory: {gedm.element_size() * gedm.nelement() / 1024**2:.2f} MB")
        # print(overlap.shape, gedm.shape)
        # result_all = optimized_einsum_fast(overlap, gedm)
        # torch.cuda.empty_cache()
        # gc.collect()
        # temp = torch.einsum('fimkb,fiba->fimka', overlap, gedm)
        # result_all = torch.einsum('fimka,finla->fimnkl', temp, overlap)
        result_all = torch.einsum('fimkb,fiba,finla->fimnkl', overlap, gedm, overlap)
        valid_results = result_all[valid_mask]  # [N_valid, nlocal, nlocal]
        del result_all
        valid_iR = iR_mat[valid_mask]  # [N_valid, 3]
        
        vdr_pred_flat = vdr_pred.view(nframes, iRmax, iRmax, iRmax, -1)
        
        frame_indices = valid_indices[:, 0]
        iR0 = valid_iR[:, 0]
        iR1 = valid_iR[:, 1]
        iR2 = valid_iR[:, 2]
        
        linear_indices = frame_indices * iRmax ** 3 + iR0 * iRmax ** 2 + iR1 * iRmax + iR2
        vdr_pred_flat.view(-1, nlocal * nlocal).index_add_(
            0, linear_indices, valid_results.view(valid_results.size(0), -1)
        )
        vdr_pred = vdr_pred_flat.view(nframes, iRmax, iRmax, iRmax, nlocal, nlocal)

    return vdr_pred

def optimized_einsum_fast(overlap, gedm):
    """
    性能优先的优化方案：
    1. 使用torch.matmul替代部分einsum（底层优化更好）
    2. 智能分块，利用GPU并行性
    3. 最小化内存复制
    """
    # 原始形状: overlap: [4, 8, 29, 152, 81], gedm: [4, 8, 81, 81]
    # 目标形状: [4, 8, 29, 152, 29, 152]
    
    f_size, i_size, m_size, n_size, k_size = overlap.shape
    _, _, _, b_size = gedm.shape
    assert k_size == b_size, "维度不匹配"
    
    # 方法1：使用torch.bmm分块计算（最优化性能）
    # 将计算分解为两个高效矩阵乘法
    result = torch.zeros(f_size, i_size, m_size, n_size, m_size, n_size, 
                         dtype=overlap.dtype, device=overlap.device)
    
    # 预处理：重塑overlap为更适合矩阵乘法的形状
    # overlap_reshaped: [4, 8, 29*152, 81]
    overlap_flat = overlap.reshape(f_size, i_size, m_size * n_size, k_size)
    
    # 第一步: 计算 A = overlap_flat @ gedm^T
    # 形状: [4, 8, 29*152, 81]
    # 使用torch.matmul优化GEMM操作
    A = torch.matmul(overlap_flat, gedm.transpose(-2, -1))
    
    # 第二步: 计算 result = A @ overlap_flat^T
    # 这里我们需要小心地重塑回目标形状
    for m1 in range(m_size):
        # 每次处理一个m位置
        start_idx1 = m1 * n_size
        end_idx1 = (m1 + 1) * n_size
        A_m = A[:, :, start_idx1:end_idx1, :]  # [4, 8, 152, 81]
        
        # 批量计算所有m2
        for m2 in range(m_size):
            start_idx2 = m2 * n_size
            end_idx2 = (m2 + 1) * n_size
            overlap_m2 = overlap_flat[:, :, start_idx2:end_idx2, :]  # [4, 8, 152, 81]
            
            # 核心计算: [4, 8, 152, 81] @ [4, 8, 81, 152] -> [4, 8, 152, 152]
            # 使用einsum的优化版本
            result[:, :, m1, :, m2, :] = torch.einsum('fijk,fikl->fijl', 
                                                      A_m, 
                                                      overlap_m2.transpose(-2, -1))
    
    return result.permute(0,1,2,4,3,5)

def get_gedm(gev, gevdm, nzeta_alpha, lmax_alpha=3):
    nframes, natoms = gev.shape[0], gev.shape[1]
    gedm_dict = {}
    gedm = torch.zeros((nframes, natoms, nzeta_alpha * (lmax_alpha + 1)**2, nzeta_alpha * (lmax_alpha + 1)**2), dtype=torch.float64, device=gev.device)
    start_index = {}
    end_index = {}
    for l in range(lmax_alpha + 1):
        start_index[l] = nzeta_alpha*l**2
        end_index[l] = nzeta_alpha*(l+1)**2
        gev_nl = gev[:, :, start_index[l]:end_index[l]] # (n*(2l+1), )
        gev_nl = gev_nl.reshape(nframes, natoms, nzeta_alpha, 2*l+1) # (n, (2l+1))
        gevdm_nl = gevdm[:, :, nzeta_alpha*l:nzeta_alpha*(l+1), :2*l+1, :2*l+1, :2*l+1] # (n, 2l+1, 2l+1, 2l+1)
        result_nl = torch.einsum("...kv,...kvmn->...kmn", gev_nl, gevdm_nl)
        gedm_dict[l] = result_nl
    for iframe in range(nframes):
        for iat in range(natoms):
            for l in range(lmax_alpha + 1):
                size = 2 * l + 1
                sub_size = nzeta_alpha * size
                gedm_tmp = torch.zeros((sub_size, sub_size), dtype=torch.float64)
                for i_n in range(nzeta_alpha):
                    start_idx = i_n * size
                    end_idx = (i_n + 1) * size
                    gedm_tmp[start_idx:end_idx, start_idx:end_idx] = gedm_dict[l][iframe, iat, i_n]
                gedm[iframe, iat, start_index[l]:end_index[l], start_index[l]:end_index[l]] = gedm_tmp
    return gedm

def find_neighbor_pair(box, types, atoms, orb, alpha, idx2nlocal=None):
    # Find neighboring pairs of atoms within a cutoff distance
    # box is lattice constant, coord is coordinates of atoms
    # find neighbor pairs for all Bravial lattice vectors
    # box: nframes x 3 x 3, coord: nframes x natom x 3
    nframes = atoms.shape[0]
    natom = atoms.shape[1]
    cutoff = orb.rcut_max() + alpha.rcut_max() # unit: a.u., orb.rcut_max ?
    box = box.to(torch.float64)
    dRx = torch.norm(box[:, 0, :], dim=-1)
    dRy = torch.norm(box[:, 1, :], dim=-1)
    dRz = torch.norm(box[:, 2, :], dim=-1)
    neighbors = [{} for _ in range(nframes)]  # list of dicts, each dict for one atom
    for frame in range(nframes):
        Rx_range = int(cutoff / dRx[frame]) + 1
        Ry_range = int(cutoff / dRy[frame]) + 1
        Rz_range = int(cutoff / dRz[frame]) + 1
        for Rx in range(-Rx_range, Rx_range + 1):
            for Ry in range(-Ry_range, Ry_range + 1):
                for Rz in range(-Rz_range, Rz_range + 1):
                    # Calculate the shifted coordinates
                    shifted_coord = atoms + (Rx * box[:, 0, :] + Ry * box[:, 1, :] + Rz * box[:, 2, :]).unsqueeze(1)
                    for i in range(natom):
                        if i not in neighbors[frame].keys():
                            neighbors[frame][i] = []
                        for j in range(natom):
                            dist = atoms[frame, i, :] - shifted_coord[frame, j, :]
                            dist_0 = torch.norm(dist, dim=-1).to("cpu")
                            if dist_0 < cutoff:
                                neighbors[frame][i].append((j, Rx, Ry, Rz, dist))

    # n_neighbors: number of neighbors for each atom in each frame
    n_neighbors = torch.zeros((nframes, natom), dtype=torch.int64)
    for iframe in range(nframes):
        for iat in range(natom):
            n_neighbors[iframe, iat] = len(neighbors[iframe][iat])

    # nn_range: range of local orbital indices for each neighbor
    if idx2nlocal is not None:
        orb_dict = {}
        for i in range(orb.ntype):
            orb_dict[orb.symbol(i)] = i
        nnmax = n_neighbors.max().item()
        nn_range = torch.zeros((nframes, natom, nnmax, 2), dtype=torch.int64)
        for iframe in range(nframes):
            for iat in range(natom):
                for inn in range(n_neighbors[iframe, iat]):
                    ibt = neighbors[iframe][iat][inn][0]
                    t_this = orb_dict[TYPE_NAME[types[iframe, ibt].item()]]
                    nn_range[iframe, iat, inn, 0] = idx2nlocal[(ibt, t_this, 0, 0, 0)]
                    if ibt + 1 == natom:
                        nn_range[iframe, iat, inn, 1] = idx2nlocal[(ibt + 1, -1, 0, 0, 0)]
                    else:
                        t_next = orb_dict[TYPE_NAME[types[iframe, ibt + 1].item()]]
                        nn_range[iframe, iat, inn, 1] = idx2nlocal[(ibt + 1, t_next, 0, 0, 0)]
    if idx2nlocal is not None:
        return neighbors, nn_range
    else:
        return neighbors

def nlocalVSidx(types, orb, nlocal):
    orb_dict = {}
    nlocal2idx = {}
    idx2nlocal = {}
    for i in range(orb.ntype):
        orb_dict[orb.symbol(i)] = i
    ilocal = 0
    for iat in range(types.shape[1]):
        t = orb_dict[TYPE_NAME[types[0, iat].item()]]
        for l in range(orb.lmax(t) + 1):
            for n in range(orb.nzeta(t,l)):
                for m in range(2*l+1):
                    idx = (iat, t, n, l, m)
                    nlocal2idx[ilocal] = idx
                    idx2nlocal[idx] = ilocal
                    ilocal += 1
    idx2nlocal[(types.shape[1], -1, 0, 0, 0)] = ilocal # padding
    assert ilocal == nlocal, "Inconsistent nlocal"
    return nlocal2idx, idx2nlocal

def make_integrator(orb_files, alpha_files):
    if base is None or nao is None:
        raise ImportError("pyabacus is required for make_integrator()")
    orb = nao.RadialCollection()
    alpha = nao.RadialCollection()
    orb.build(len(orb_files), orb_files)
    alpha.build(len(alpha_files), alpha_files)

    dr = 0.01
    rmax = max(orb.rcut_max(), alpha.rcut_max())
    cutoff = 2.0 * rmax
    nr = int(rmax / dr) + 1
    orb.set_uniform_grid(True, nr, cutoff, 'i', True)
    alpha.set_uniform_grid(True, nr, cutoff, 'i', True)

    sbt = base.SphericalBesselTransformer()
    orb.set_transformer(sbt)
    alpha.set_transformer(sbt)

    integrator = nao.TwoCenterIntegrator()
    integrator.tabulate(orb, alpha, 'S', nr, cutoff)
    return orb, alpha, integrator

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