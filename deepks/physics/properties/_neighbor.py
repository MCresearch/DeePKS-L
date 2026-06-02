"""Shared neighbor/overlap helpers used by property-side preprocessing."""

import torch

from deepks.physics.constants import TYPE_NAME
from deepks.physics.backends.abacus.utils import R2iR


def cal_nb_overlap(types, atoms, box, orb, alpha, integrator, nlocal):
    device = atoms.device
    nframes, natoms = atoms.shape[0], atoms.shape[1]
    lmax_alpha = alpha.lmax(0)
    nzeta_alpha = alpha.nzeta(0, 0)

    nlocal2idx, idx2nlocal = nlocalVSidx(types.to("cpu"), orb, nlocal)
    neighbors, nn_range = find_neighbor_pair(
        box.to("cpu"), types.to("cpu"), atoms.to("cpu"), orb, alpha, idx2nlocal
    )
    nnmax = nn_range.shape[2]

    overlap = torch.zeros(
        (nframes, natoms, nnmax, nlocal, nzeta_alpha * (lmax_alpha + 1) ** 2),
        dtype=torch.float64,
        device=device,
    )
    for iframe in range(nframes):
        for iat in range(natoms):
            for inn, (ibt1, rx1, ry1, rz1, dist1) in enumerate(neighbors[iframe][iat]):
                del ibt1, rx1, ry1, rz1
                for ix in range(nn_range[iframe][iat][inn][0], nn_range[iframe][iat][inn][1]):
                    ibt, t1, n1, l1, m1_encoded = nlocal2idx[ix]
                    del ibt
                    if m1_encoded % 2 == 0:
                        m1 = -m1_encoded // 2
                    else:
                        m1 = (m1_encoded + 1) // 2
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


def find_neighbor_pair(box, types, atoms, orb, alpha, idx2nlocal=None):
    nframes = atoms.shape[0]
    natom = atoms.shape[1]
    cutoff = orb.rcut_max() + alpha.rcut_max()
    box = box.to(torch.float64)
    d_rx = torch.norm(box[:, 0, :], dim=-1)
    d_ry = torch.norm(box[:, 1, :], dim=-1)
    d_rz = torch.norm(box[:, 2, :], dim=-1)
    neighbors = [{} for _ in range(nframes)]
    for frame in range(nframes):
        rx_range = int(cutoff / d_rx[frame]) + 1
        ry_range = int(cutoff / d_ry[frame]) + 1
        rz_range = int(cutoff / d_rz[frame]) + 1
        for rx in range(-rx_range, rx_range + 1):
            for ry in range(-ry_range, ry_range + 1):
                for rz in range(-rz_range, rz_range + 1):
                    shifted_coord = atoms + (rx * box[:, 0, :] + ry * box[:, 1, :] + rz * box[:, 2, :]).unsqueeze(1)
                    for i in range(natom):
                        if i not in neighbors[frame]:
                            neighbors[frame][i] = []
                        for j in range(natom):
                            dist = atoms[frame, i, :] - shifted_coord[frame, j, :]
                            dist_0 = torch.norm(dist, dim=-1).to("cpu")
                            if dist_0 < cutoff:
                                neighbors[frame][i].append((j, rx, ry, rz, dist))

    n_neighbors = torch.zeros((nframes, natom), dtype=torch.int64)
    for iframe in range(nframes):
        for iat in range(natom):
            n_neighbors[iframe, iat] = len(neighbors[iframe][iat])

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
        return neighbors, nn_range
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
            for n in range(orb.nzeta(t, l)):
                for m in range(2 * l + 1):
                    idx = (iat, t, n, l, m)
                    nlocal2idx[ilocal] = idx
                    idx2nlocal[idx] = ilocal
                    ilocal += 1
    idx2nlocal[(types.shape[1], -1, 0, 0, 0)] = ilocal
    assert ilocal == nlocal, "Inconsistent nlocal"
    return nlocal2idx, idx2nlocal
