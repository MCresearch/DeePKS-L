"""Reader file-name schema utilities for DeePKS data loading."""

from dataclasses import asdict, dataclass
import os
from typing import Dict, Optional


@dataclass(frozen=True)
class ReaderFieldNames:
    e_name: str = "l_e_delta"
    d_name: str = "dm_eig"
    f_name: str = "l_f_delta"
    gvx_name: str = "grad_vx"
    s_name: str = "l_s_delta"
    gvepsl_name: str = "grad_vepsl"
    o_name: str = "l_o_delta"
    op_name: str = "orbital_precalc"
    h_name: str = "l_h_delta"
    vdp_name: str = "v_delta_precalc"
    vdrp_name: str = "vdr_precalc"
    phialpha_name: str = "phialpha"
    gevdm_name: str = "grad_evdm"
    hr_name: str = "l_hr_delta"
    h_base_name: str = "h_base"
    h_ref_name: str = "hamiltonian"
    overlap_name: str = "overlap"
    eg_name: str = "eg_base"
    gveg_name: str = "grad_veg"
    gldv_name: str = "grad_ldv"
    conv_name: str = "conv"
    atom_name: str = "atom"
    box_name: str = "box"
    # deepks_v_delta=-2 R-space chain-rule helpers (precomputed by ABACUS,
    # collected by iterate_ops). When present alongside ``grad_evdm.npy`` the
    # reader's R-space loader skips the Python-side cal_nb_overlap and feeds
    # these directly into the batch context for the chain-rule V_delta(R) path.
    iR_mat_name: str = "iR_mat"
    phialpha_r_name: str = "phialpha_r"


DEFAULT_READER_FIELD_NAMES = ReaderFieldNames()


READER_PATH_ATTR_MAP = {
    "e_name": "e_path",
    "d_name": "d_path",
    "f_name": "f_path",
    "gvx_name": "gvx_path",
    "s_name": "s_path",
    "gvepsl_name": "gvepsl_path",
    "o_name": "o_path",
    "op_name": "op_path",
    "h_name": "h_path",
    "vdp_name": "vdp_path",
    "vdrp_name": "vdrp_path",
    "phialpha_name": "phialpha_path",
    "gevdm_name": "gevdm_path",
    "hr_name": "hr_path",
    "h_base_name": "h_base_path",
    "h_ref_name": "h_ref_path",
    "overlap_name": "overlap_path",
    "eg_name": "eg_path",
    "gveg_name": "gveg_path",
    "gldv_name": "gldv_path",
    "conv_name": "c_path",
    "atom_name": "a_path",
    "box_name": "b_path",
    "iR_mat_name": "iR_mat_path",
    "phialpha_r_name": "phialpha_r_path",
}


def resolve_numpy_path(data_path: str, stem: Optional[str]) -> Optional[str]:
    if stem is None:
        return None
    fpath = os.path.join(data_path, f"{stem}.npy")
    if os.path.exists(fpath):
        return fpath
    return None


def resolve_reader_paths(data_path: str, names: ReaderFieldNames) -> Dict[str, Optional[str]]:
    return {
        READER_PATH_ATTR_MAP[name]: resolve_numpy_path(data_path, stem)
        for name, stem in asdict(names).items()
    }
