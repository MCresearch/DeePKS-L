"""Reader-side feature derivation helpers owned by the io layer."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from deepks.io.transforms.linalg import generalized_eigh


def _select_preferred_optional_path(
    primary_path: Optional[str],
    secondary_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Keep the newer of two alternative precalc sources, matching Reader precedence."""

    if primary_path is None or secondary_path is None:
        return primary_path, secondary_path
    if os.path.getmtime(primary_path) >= os.path.getmtime(secondary_path):
        return primary_path, None
    return None, secondary_path


def _build_transformation_matrix(overlap: torch.Tensor, eigh_method: int) -> torch.Tensor:
    """Construct the generalized-eigh transformation matrix for overlap handling."""

    if eigh_method == 1:
        chol = torch.linalg.cholesky(overlap)
        return torch.linalg.inv(chol).mT
    if eigh_method == 2:
        overlap_eigenvalue, overlap_eigenvector = torch.linalg.eigh(overlap)
        epsilon = 1e-16
        overlap_eigenvalue = torch.clamp(overlap_eigenvalue, min=epsilon)
        sigma_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(overlap_eigenvalue))
        return overlap_eigenvector @ sigma_inv_sqrt
    raise ValueError(f"Unsupported eigh_method: {eigh_method}")


def load_kspace_hamiltonian_fields(
    *,
    raw_nframes: int,
    conv,
    natm: int,
    ndesc: int,
    h_path: Optional[str],
    vdp_path: Optional[str],
    phialpha_path: Optional[str],
    gevdm_path: Optional[str],
    h_base_path: Optional[str],
    h_ref_path: Optional[str],
    read_overlap: bool,
    overlap_path: Optional[str],
    eigh_method: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[int]]:
    """Load k-space Hamiltonian labels and derived fields for readers."""

    t_data: Dict[str, torch.Tensor] = {}
    if h_path is None or (vdp_path is None and (phialpha_path is None or gevdm_path is None)):
        return t_data, None

    h_shape = np.load(h_path).shape
    assert h_shape[-1] == h_shape[-2], (
        "The last two dimension of H must have the same size , which is nlocal"
    )
    nlocal = h_shape[-1]
    t_data["lb_vd"] = torch.tensor(
        np.load(h_path).reshape(raw_nframes, -1, nlocal, nlocal)[conv]
    )

    resolved_vdp_path, resolved_phialpha_path = _select_preferred_optional_path(vdp_path, phialpha_path)
    resolved_gevdm_path = gevdm_path if resolved_phialpha_path is not None else None
    if resolved_vdp_path is not None:
        t_data["vdp"] = torch.tensor(
            np.load(resolved_vdp_path).reshape(
                raw_nframes, -1, nlocal, nlocal, natm, ndesc
            )[conv]
        )
    elif resolved_phialpha_path is not None and resolved_gevdm_path is not None:
        phialpha = np.load(resolved_phialpha_path)
        nl = phialpha.shape[2]
        mmax = phialpha.shape[-1]
        t_data["phialpha"] = torch.tensor(
            phialpha.reshape(raw_nframes, natm, nl, -1, nlocal, mmax)[conv]
        )
        t_data["gevdm"] = torch.tensor(
            np.load(resolved_gevdm_path).reshape(raw_nframes, natm, nl, mmax, mmax, mmax)[conv]
        )

    if h_base_path is not None:
        t_data["h_base"] = torch.tensor(
            np.load(h_base_path).reshape(raw_nframes, -1, nlocal, nlocal)[conv]
        )
    if h_ref_path is not None:
        h_ref = torch.tensor(np.load(h_ref_path))
        if read_overlap and overlap_path is not None:
            overlap = torch.tensor(np.load(overlap_path))
            t_data["overlap"] = overlap.reshape(raw_nframes, -1, nlocal, nlocal)[conv].clone()
            trans_matrix = _build_transformation_matrix(overlap, eigh_method)
            t_data["trans_matrix"] = trans_matrix.reshape(raw_nframes, -1, nlocal, nlocal)[conv].clone()
            band_ref, phi_ref = generalized_eigh(h_ref, trans_matrix)
        else:
            band_ref, phi_ref = torch.linalg.eigh(h_ref, UPLO="U")
        t_data["lb_band"] = band_ref.reshape(raw_nframes, -1, nlocal)[conv].clone()
        t_data["lb_phi"] = phi_ref.reshape(raw_nframes, -1, nlocal, nlocal)[conv].clone()

    return t_data, nlocal


def _infer_data_shape_from_gevdm(gevdm: torch.Tensor):
    """Derive ``[nzeta_alpha, lmax_alpha]`` from the loaded ``gevdm`` tensor.

    ``gevdm`` has shape ``(..., nl, mmax, mmax, mmax)`` where ``mmax = 2*lmax+1``
    and ``nl = nzeta*(lmax+1)``. The objective adapter's ``vdr`` recovery uses
    these two integers to slice ``gevdm`` per ``l``-channel
    (see ``physics.properties.vdr._get_gedm``).
    """

    mmax = int(gevdm.shape[-1])
    lmax_alpha = (mmax - 1) // 2
    nl = int(gevdm.shape[-4])
    nzeta_alpha = nl // (lmax_alpha + 1)
    return [nzeta_alpha, lmax_alpha]


def load_rspace_hamiltonian_fields(
    *,
    raw_nframes: int,
    conv,
    hr_path: Optional[str],
    vdrp_path: Optional[str],
    gevdm_path: Optional[str],
    orb_list,
    alpha_list,
    atom_info: Dict[str, np.ndarray],
    iR_mat_path: Optional[str] = None,
    phialpha_r_path: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[int]]:
    """Load R-space Hamiltonian labels and derived neighbor-overlap fields.

    For ``deepks_v_delta=-2`` data, the recommended (and fastest) path is to
    have ABACUS dump ``deepks_iRmat.npy`` and ``deepks_phialpha_r.npy``
    alongside ``deepks_gevdm.npy``; ``iterate_ops`` then collects them as
    ``iR_mat.npy`` / ``phialpha_r.npy`` at the system level and this loader
    feeds them directly into the batch context. The legacy fallback (call
    ``cal_nb_overlap`` to recompute overlap/iR_mat from atomic structure +
    orbital files) is retained for callers that didn't precompute these
    helpers and that explicitly threaded ``orb_list`` / ``alpha_list``
    through the loader config.
    """

    t_data: Dict[str, torch.Tensor] = {}
    if hr_path is None:
        return t_data, None

    t_data["lb_vdr"] = torch.tensor(np.load(hr_path)[conv])
    nlocal = t_data["lb_vdr"].shape[-1]

    resolved_vdrp_path, resolved_gevdm_path = _select_preferred_optional_path(vdrp_path, gevdm_path)
    if resolved_gevdm_path is not None:
        gevdm = np.load(resolved_gevdm_path)
        t_data["gevdm"] = torch.tensor(gevdm[conv])
        # Fast path: ABACUS already produced the R-space overlap + iR mapping.
        if iR_mat_path is not None and phialpha_r_path is not None:
            t_data["iR_mat"] = torch.tensor(np.load(iR_mat_path)[conv])
            t_data["overlap"] = torch.tensor(np.load(phialpha_r_path)[conv])
            t_data["data_shape"] = _infer_data_shape_from_gevdm(t_data["gevdm"])
        elif (
            orb_list is not None
            and alpha_list is not None
            and "elems" in atom_info
            and "coords" in atom_info
            and "lattice" in atom_info
        ):
            # Legacy fallback: recompute overlap / iR_mat from atomic structure.
            # These helpers live in the physics layer because they depend on
            # pyabacus radial integrators; we import lazily so io/ stays
            # importable when pyabacus isn't installed and the fast path is in use.
            from deepks.physics.backends.abacus.integrator import make_integrator
            from deepks.physics.properties._neighbor import cal_nb_overlap

            types = torch.tensor(atom_info["elems"])
            atoms = torch.tensor(atom_info["coords"])
            box = torch.tensor(atom_info["lattice"])
            orb, alpha, integrator = make_integrator(orb_list, alpha_list)
            overlap, iR_mat, data_shape = cal_nb_overlap(
                types, atoms, box, orb, alpha, integrator, nlocal
            )
            t_data["overlap"] = overlap
            t_data["iR_mat"] = iR_mat
            t_data["data_shape"] = data_shape
    elif resolved_vdrp_path is not None:
        vdrp = np.load(resolved_vdrp_path)[conv]
        target_r = tuple(int(v) for v in t_data["lb_vdr"].shape[1:4])
        current_r = tuple(int(v) for v in vdrp.shape[1:4])
        if current_r != target_r:
            pad_width = [(0, 0)]
            for axis in range(3):
                n_add = target_r[axis] - current_r[axis]
                if n_add < 0:
                    raise ValueError(
                        f"vdr_precalc R-shape {current_r} exceeds label R-shape {target_r}; "
                        "recollect data with a consistent target range"
                    )
                pad_width.append((0, n_add))
            pad_width.extend((0, 0) for _ in range(vdrp.ndim - 4))
            vdrp = np.pad(vdrp, pad_width)
        t_data["vdrp"] = torch.tensor(vdrp)

    return t_data, nlocal


__all__ = [
    "load_kspace_hamiltonian_fields",
    "load_rspace_hamiltonian_fields",
]
