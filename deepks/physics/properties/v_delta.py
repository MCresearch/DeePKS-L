"""v_delta property helpers."""

import torch


def v_delta_from_context(input_grad, context):
    if "vdp" in context:
        return torch.einsum("...kxyap,...ap->...kxy", context["vdp"], input_grad)
    if "phialpha" in context and "gevdm" in context:
        mmax = context["phialpha"].size(-1)
        lmax = int((mmax - 1) / 2)
        n = int(context["phialpha"].size(2) / (lmax + 1))

        dtype = context["phialpha"].dtype
        gev = input_grad.to(dtype)
        gevdm = context["gevdm"].to(dtype)

        n_batch = context["phialpha"].size(0)
        nks = context["phialpha"].size(-3)
        nlocal = context["phialpha"].size(-2)
        v_delta = torch.zeros([n_batch, nks, nlocal, nlocal], dtype=dtype, device=gev.device)
        for l in range(lmax + 1):
            gevdm_l = gevdm[..., n * l : n * (l + 1), : 2 * l + 1, : 2 * l + 1, : 2 * l + 1]
            gev_l = gev[..., n * l**2 : n * (l + 1) ** 2]
            gev_l = gev_l.view(gev_l.size(0), gev_l.size(1), n, 2 * l + 1)

            temp_1 = torch.einsum("...v,...vmn->...mn", gev_l, gevdm_l)
            phialpha_l = context["phialpha"][..., n * l : n * (l + 1), :, :, : 2 * l + 1].to(gev.device)
            temp_2 = torch.einsum("...mn,...kxn->...kxm", temp_1, phialpha_l)
            v_delta += torch.einsum("...alkxm,...alkym->...kxy", temp_2, phialpha_l.conj())
        return v_delta
    raise KeyError("context does not contain v_delta inputs")
