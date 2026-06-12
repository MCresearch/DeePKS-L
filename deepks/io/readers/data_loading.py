"""Reader-side dataset decoding helpers owned by the io layer."""

import os

import numpy as np
import torch

from deepks.io.schemas.reader_fields import resolve_numpy_path

from .feature_loading import (
    load_kspace_hamiltonian_fields,
    load_rspace_hamiltonian_fields,
)


def load_reader_raw_data(
    *,
    data_path,
    e_path,
    d_path,
    c_path,
    a_path,
    b_path,
    natm,
    ndesc,
):
    """Load raw arrays and common metadata for readers."""

    data_ec = None
    if e_path is not None:
        data_ec = np.load(e_path).reshape(-1, 1)
        raw_nframes = data_ec.shape[0]
    else:
        raw_nframes = np.load(d_path).shape[0]
    data_dm = np.load(d_path).reshape(raw_nframes, natm, ndesc)
    if c_path is not None:
        conv = np.load(c_path).reshape(raw_nframes).astype(bool)
    else:
        conv = np.ones(raw_nframes, dtype=bool)

    atom_info = {}
    if a_path is not None:
        atoms = np.load(a_path).reshape(raw_nframes, natm, 4)
        atom_info["elems"] = atoms[:, :, 0][conv].round().astype(int)
        atom_info["coords"] = atoms[:, :, 1:][conv]
    if b_path is not None:
        box = np.load(b_path).reshape(raw_nframes, 3, 3)
        atom_info["lattice"] = box[conv]

    return {
        "data_path": data_path,
        "raw_nframes": raw_nframes,
        "conv": conv,
        "data_ec": None if data_ec is None else data_ec[conv],
        "data_dm": data_dm[conv],
        "atom_info": atom_info,
    }


def build_reader_tensor_data(
    *,
    raw_data,
    natm,
    ndesc,
    f_path,
    gvx_path,
    s_path,
    gvepsl_path,
    o_path,
    op_path,
    h_path,
    vdp_path,
    phialpha_path,
    gevdm_path,
    h_base_path,
    h_ref_path,
    read_overlap,
    overlap_path,
    eigh_method,
    hr_path,
    vdrp_path,
    orb_list,
    alpha_list,
    eg_path,
    gveg_path,
    gldv_path,
    iR_mat_path=None,
    phialpha_r_path=None,
    hamiltonian_level_names=None,
    hamiltonian_name=None,
    csr_hr_name=None,
):
    """Build reader tensor data from already-loaded raw arrays."""

    raw_nframes = raw_data["raw_nframes"]
    conv = raw_data["conv"]
    data_ec = raw_data["data_ec"]
    data_dm = raw_data["data_dm"]
    atom_info = raw_data["atom_info"]

    t_data = {
        "eig": torch.tensor(data_dm),
    }
    if data_ec is not None:
        t_data["lb_e"] = torch.tensor(data_ec)
    extra = {}

    if f_path is not None and gvx_path is not None:
        t_data["lb_f"] = torch.tensor(np.load(f_path).reshape(raw_nframes, -1, 3)[conv])
        t_data["gvx"] = torch.tensor(
            np.load(gvx_path).reshape(raw_nframes, natm, 3, natm, ndesc)[conv]
        )
    if s_path is not None and gvepsl_path is not None:
        t_data["lb_s"] = torch.tensor(np.load(s_path).reshape(raw_nframes, 6)[conv])
        t_data["gvepsl"] = torch.tensor(
            np.load(gvepsl_path).reshape(raw_nframes, 6, natm, ndesc)[conv]
        )
    if o_path is not None and op_path is not None:
        t_data["lb_o"] = torch.tensor(np.load(o_path)[conv])
        t_data["op"] = torch.tensor(np.load(op_path)[conv])

    kspace_data, kspace_nlocal = load_kspace_hamiltonian_fields(
        raw_nframes=raw_nframes,
        conv=conv,
        natm=natm,
        ndesc=ndesc,
        h_path=h_path,
        vdp_path=vdp_path,
        phialpha_path=phialpha_path,
        gevdm_path=gevdm_path,
        h_base_path=h_base_path,
        h_ref_path=h_ref_path,
        read_overlap=read_overlap,
        overlap_path=overlap_path,
        eigh_method=eigh_method,
    )
    t_data.update(kspace_data)
    if kspace_nlocal is not None:
        extra["nlocal"] = kspace_nlocal

    rspace_data, rspace_nlocal = load_rspace_hamiltonian_fields(
        raw_nframes=raw_nframes,
        conv=conv,
        hr_path=hr_path,
        vdrp_path=vdrp_path,
        gevdm_path=gevdm_path,
        orb_list=orb_list,
        alpha_list=alpha_list,
        atom_info=atom_info,
        iR_mat_path=iR_mat_path,
        phialpha_r_path=phialpha_r_path,
    )
    t_data.update(rspace_data)
    if rspace_nlocal is not None:
        extra["nlocal"] = rspace_nlocal

    if eg_path is not None and gveg_path is not None:
        t_data["eg0"] = torch.tensor(np.load(eg_path).reshape(raw_nframes, -1)[conv])
        t_data["gveg"] = torch.tensor(
            np.load(gveg_path).reshape(raw_nframes, natm, ndesc, -1)[conv]
        )
        extra["neg"] = t_data["eg0"].shape[-1]

    if gldv_path is not None:
        t_data["gldv"] = torch.tensor(
            np.load(gldv_path).reshape(raw_nframes, natm, ndesc)[conv]
        )

    if hamiltonian_level_names:
        for level_index, field_name in enumerate(hamiltonian_level_names):
            level_path = resolve_numpy_path(raw_data["data_path"], field_name)
            if level_path is None:
                raise FileNotFoundError(
                    f"Missing hierarchical Hamiltonian target '{field_name}.npy' under {raw_data['data_path']}"
                )
            level_data = np.load(level_path)
            if level_data.shape[0] != raw_nframes:
                level_data = level_data.reshape(raw_nframes, *level_data.shape[1:])
            t_data[f"lb_ham_level_{level_index}"] = torch.tensor(level_data[conv])

    if hamiltonian_name:
        hamiltonian_path = resolve_numpy_path(raw_data["data_path"], hamiltonian_name)
        if hamiltonian_path is None:
            raise FileNotFoundError(
                f"Missing Hamiltonian target '{hamiltonian_name}.npy' under {raw_data['data_path']}"
            )
        hamiltonian_data = np.load(hamiltonian_path)
        if hamiltonian_data.shape[0] != raw_nframes:
            hamiltonian_data = hamiltonian_data.reshape(raw_nframes, *hamiltonian_data.shape[1:])
        t_data["lb_hamiltonian"] = torch.tensor(hamiltonian_data[conv])

    if csr_hr_name:
        csr_targets = _load_csr_hr_targets(raw_data["data_path"], csr_hr_name, conv)
        t_data["lb_csr_hamiltonian"] = csr_targets

    return t_data, extra


def _load_csr_hr_targets(data_path, stem, conv):
    data_path_np = os.path.join(data_path, f"{stem}_data.npy")
    indices_path = os.path.join(data_path, f"{stem}_indices.npy")
    indptr_path = os.path.join(data_path, f"{stem}_indptr.npy")
    shape_path = os.path.join(data_path, f"{stem}_shape.npy")
    for path in (data_path_np, indices_path, indptr_path, shape_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSR Hamiltonian component: {path}")

    data_arr = np.load(data_path_np, allow_pickle=True)
    indices_arr = np.load(indices_path, allow_pickle=True)
    indptr_arr = np.load(indptr_path, allow_pickle=True)
    shape_arr = np.load(shape_path, allow_pickle=True)

    frame_indices = np.nonzero(conv)[0]
    targets = []
    for frame_index in frame_indices:
        rows, cols = _csr_to_coo(indices_arr[frame_index], indptr_arr[frame_index])
        shape = shape_arr[frame_index] if np.ndim(shape_arr) > 1 else shape_arr
        targets.append(
            {
                "shape": tuple(int(v) for v in np.asarray(shape).tolist()),
                "row": torch.tensor(rows, dtype=torch.long),
                "col": torch.tensor(cols, dtype=torch.long),
                "data": torch.tensor(np.asarray(data_arr[frame_index]), dtype=torch.float64),
            }
        )
    return targets


def _csr_to_coo(indices, indptr):
    indices = np.asarray(indices, dtype=int)
    indptr = np.asarray(indptr, dtype=int)
    row = np.repeat(np.arange(len(indptr) - 1, dtype=int), np.diff(indptr))
    return row, indices


__all__ = ["load_reader_raw_data", "build_reader_tensor_data"]
