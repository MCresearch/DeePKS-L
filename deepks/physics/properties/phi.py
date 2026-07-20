"""Phi property helpers."""

import torch


def density_matrix_from_phi(phi, density_m_occ):
    phi_occ = phi[..., :density_m_occ]
    batch_size, nks, nlocal, nocc = phi_occ.size()
    phi_occ = phi_occ.view(batch_size * nks, nlocal, nocc)
    density_m = torch.bmm(phi_occ, phi_occ.transpose(-1, -2))
    density_m = density_m.view(batch_size, nks, nlocal, nlocal)
    return density_m


def phi_from_solution(band_solution):
    _, phi_pred = band_solution
    return phi_pred
