"""Real-space Hamiltonian response helpers."""

import torch


def _cal_vdr_from_inputs(gedm, overlap, iR_mat):
    nframes = gedm.shape[0]
    iRmax = int(iR_mat.max().item()) + 1
    nlocal = overlap.shape[3]
    vdr_pred = torch.zeros(
        (nframes, iRmax, iRmax, iRmax, nlocal, nlocal),
        dtype=gedm.dtype,
        device=gedm.device,
    )

    valid_mask = iR_mat.max(dim=-1)[0] < iRmax
    valid_indices = torch.nonzero(valid_mask)
    if valid_indices.size(0) != 0:
        result_all = torch.einsum("fimkb,fiba,finla->fimnkl", overlap, gedm, overlap)
        valid_results = result_all[valid_mask]
        valid_iR = iR_mat[valid_mask]

        vdr_pred_flat = vdr_pred.view(nframes, iRmax, iRmax, iRmax, -1)
        frame_indices = valid_indices[:, 0]
        iR0 = valid_iR[:, 0]
        iR1 = valid_iR[:, 1]
        iR2 = valid_iR[:, 2]

        linear_indices = frame_indices * iRmax**3 + iR0 * iRmax**2 + iR1 * iRmax + iR2
        vdr_pred_flat.view(-1, nlocal * nlocal).index_add_(
            0, linear_indices, valid_results.view(valid_results.size(0), -1)
        )
        vdr_pred = vdr_pred_flat.view(nframes, iRmax, iRmax, iRmax, nlocal, nlocal)

    return vdr_pred


def _get_gedm(gev, gevdm, nzeta_alpha, lmax_alpha=3):
    nframes, natoms = gev.shape[0], gev.shape[1]
    gedm_dict = {}
    gedm = torch.zeros(
        (nframes, natoms, nzeta_alpha * (lmax_alpha + 1) ** 2, nzeta_alpha * (lmax_alpha + 1) ** 2),
        dtype=torch.float64,
        device=gev.device,
    )
    start_index = {}
    end_index = {}
    for l in range(lmax_alpha + 1):
        start_index[l] = nzeta_alpha * l**2
        end_index[l] = nzeta_alpha * (l + 1) ** 2
        gev_nl = gev[:, :, start_index[l] : end_index[l]]
        gev_nl = gev_nl.reshape(nframes, natoms, nzeta_alpha, 2 * l + 1)
        gevdm_nl = gevdm[
            :, :, nzeta_alpha * l : nzeta_alpha * (l + 1), : 2 * l + 1, : 2 * l + 1, : 2 * l + 1
        ]
        gedm_dict[l] = torch.einsum("...kv,...kvmn->...kmn", gev_nl, gevdm_nl)
    for iframe in range(nframes):
        for iat in range(natoms):
            for l in range(lmax_alpha + 1):
                size = 2 * l + 1
                sub_size = nzeta_alpha * size
                gedm_tmp = torch.zeros((sub_size, sub_size), dtype=torch.float64, device=gev.device)
                for i_n in range(nzeta_alpha):
                    start_idx = i_n * size
                    end_idx = (i_n + 1) * size
                    gedm_tmp[start_idx:end_idx, start_idx:end_idx] = gedm_dict[l][iframe, iat, i_n]
                gedm[iframe, iat, start_index[l] : end_index[l], start_index[l] : end_index[l]] = gedm_tmp
    return gedm


def vdr_from_context(input_grad, context):
    if "vdrp" in context:
        return torch.einsum("...bcdxyap,...ap->...bcdxy", context["vdrp"], input_grad)
    if "gevdm" in context and "iR_mat" in context and "overlap" in context and "data_shape" in context:
        gedm = _get_gedm(input_grad, context["gevdm"], context["data_shape"][0], context["data_shape"][1])
        return _cal_vdr_from_inputs(gedm, context["overlap"], context["iR_mat"])
    raise KeyError("context does not contain vdr inputs")
