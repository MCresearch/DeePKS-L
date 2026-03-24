import os
import sys

import numpy as np
import torch

from deepks.ml.utils import cal_nb_overlap, make_integrator
from deepks.io.schemas.reader_fields import (
    DEFAULT_READER_FIELD_NAMES,
    ReaderFieldNames,
    resolve_reader_paths,
)
from deepks.io.transforms.linalg import generalized_eigh


class Reader(object):
    def __init__(
        self,
        data_path,
        batch_size,
        e_name=DEFAULT_READER_FIELD_NAMES.e_name,
        d_name=DEFAULT_READER_FIELD_NAMES.d_name,
        f_name=DEFAULT_READER_FIELD_NAMES.f_name,
        gvx_name=DEFAULT_READER_FIELD_NAMES.gvx_name,
        s_name=DEFAULT_READER_FIELD_NAMES.s_name,
        gvepsl_name=DEFAULT_READER_FIELD_NAMES.gvepsl_name,
        o_name=DEFAULT_READER_FIELD_NAMES.o_name,
        op_name=DEFAULT_READER_FIELD_NAMES.op_name,
        h_name=DEFAULT_READER_FIELD_NAMES.h_name,
        vdp_name=DEFAULT_READER_FIELD_NAMES.vdp_name,
        vdrp_name=DEFAULT_READER_FIELD_NAMES.vdrp_name,
        phialpha_name=DEFAULT_READER_FIELD_NAMES.phialpha_name,
        gevdm_name=DEFAULT_READER_FIELD_NAMES.gevdm_name,
        hr_name=DEFAULT_READER_FIELD_NAMES.hr_name,
        h_base_name=DEFAULT_READER_FIELD_NAMES.h_base_name,
        h_ref_name=DEFAULT_READER_FIELD_NAMES.h_ref_name,
        read_overlap=False,
        overlap_name=DEFAULT_READER_FIELD_NAMES.overlap_name,
        eg_name=DEFAULT_READER_FIELD_NAMES.eg_name,
        gveg_name=DEFAULT_READER_FIELD_NAMES.gveg_name,
        gldv_name=DEFAULT_READER_FIELD_NAMES.gldv_name,
        conv_name=DEFAULT_READER_FIELD_NAMES.conv_name,
        atom_name=DEFAULT_READER_FIELD_NAMES.atom_name,
        box_name=DEFAULT_READER_FIELD_NAMES.box_name,
        orb_list=None,
        alpha_list=None,
        **kwargs,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        field_names = ReaderFieldNames(
            e_name=e_name,
            d_name=d_name,
            f_name=f_name,
            gvx_name=gvx_name,
            s_name=s_name,
            gvepsl_name=gvepsl_name,
            o_name=o_name,
            op_name=op_name,
            h_name=h_name,
            vdp_name=vdp_name,
            vdrp_name=vdrp_name,
            phialpha_name=phialpha_name,
            gevdm_name=gevdm_name,
            hr_name=hr_name,
            h_base_name=h_base_name,
            h_ref_name=h_ref_name,
            overlap_name=overlap_name,
            eg_name=eg_name,
            gveg_name=gveg_name,
            gldv_name=gldv_name,
            conv_name=conv_name,
            atom_name=atom_name,
            box_name=box_name,
        )
        for path_name, path in resolve_reader_paths(self.data_path, field_names).items():
            setattr(self, path_name, path)

        self.system_raw_path = os.path.join(self.data_path, "system.raw")
        self.read_overlap = read_overlap
        self.orb_list = ["../../" + orb for orb in orb_list] if orb_list is not None else None
        self.alpha_list = ["../../" + alpha for alpha in alpha_list] if alpha_list is not None else None
        # load data
        self.load_meta()
        self.prepare()
        # initialize sample index queue
        self.idx_queue = []

    def load_meta(self):
        try:
            sys_meta = np.loadtxt(self.system_raw_path, converters=float).astype(int).reshape([-1])
            self.natm = sys_meta[0]
            self.nproj = sys_meta[-1]
        except Exception:
            print("#", self.data_path, "no system.raw, infer meta from data", file=sys.stderr)
            sys_shape = np.load(self.d_path).shape
            assert len(sys_shape) == 3, (
                "descriptor has to be an order-3 array with shape [nframes, natom, nproj]"
            )
            self.natm = sys_shape[1]
            self.nproj = sys_shape[2]
        self.ndesc = self.nproj

    def prepare(self):
        ## Load energy and check nframes
        data_ec = np.load(self.e_path).reshape(-1, 1)  # energy
        raw_nframes = data_ec.shape[0]  # number of total frames
        data_dm = np.load(self.d_path).reshape(raw_nframes, self.natm, self.ndesc)  # descriptor
        # Use convergent structure only
        if self.c_path is not None:
            conv = np.load(self.c_path).reshape(raw_nframes)
        else:
            conv = np.ones(raw_nframes, dtype=bool)
        self.data_ec = data_ec[conv]
        self.data_dm = data_dm[conv]
        self.nframes = conv.sum()
        # reset batch size if nframes < batch_size
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print("#", self.data_path, f"reset batch size to {self.batch_size}", file=sys.stderr)

        ## Handle atom and element data
        self.atom_info = {}
        if self.a_path is not None:
            atoms = np.load(self.a_path).reshape(raw_nframes, self.natm, 4)  # atom.npy
            self.atom_info["elems"] = atoms[:, :, 0][conv].round().astype(int)
            self.atom_info["coords"] = atoms[:, :, 1:][conv]
        if self.b_path is not None:
            box = np.load(self.b_path).reshape(raw_nframes, 3, 3)
            self.atom_info["lattice"] = box[conv]
        # Energy and descriptor
        # load data in torch
        self.t_data = {}
        # Energy
        self.t_data["lb_e"] = torch.tensor(self.data_ec)
        self.t_data["eig"] = torch.tensor(self.data_dm)
        # Force
        if self.f_path is not None and self.gvx_path is not None:
            self.t_data["lb_f"] = torch.tensor(np.load(self.f_path).reshape(raw_nframes, -1, 3)[conv])
            self.t_data["gvx"] = torch.tensor(
                np.load(self.gvx_path).reshape(raw_nframes, self.natm, 3, self.natm, self.ndesc)[conv]
            )
        # Stress
        if self.s_path is not None and self.gvepsl_path is not None:
            self.t_data["lb_s"] = torch.tensor(np.load(self.s_path).reshape(raw_nframes, 6)[conv])
            self.t_data["gvepsl"] = torch.tensor(
                np.load(self.gvepsl_path).reshape(raw_nframes, 6, self.natm, self.ndesc)[conv]
            )
        # Orbital
        if self.o_path is not None and self.op_path is not None:
            self.t_data["lb_o"] = torch.tensor(np.load(self.o_path)[conv])
            self.t_data["op"] = torch.tensor(np.load(self.op_path)[conv])
        # Hamiltonian in k space
        if self.h_path is not None and (
            self.vdp_path is not None or (self.phialpha_path is not None and self.gevdm_path is not None)
        ):
            h_shape = np.load(self.h_path).shape
            assert h_shape[-1] == h_shape[-2], (
                "The last two dimension of H must have the same size , which is nlocal"
            )
            self.nlocal = h_shape[-1]
            self.t_data["lb_vd"] = torch.tensor(
                np.load(self.h_path).reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv]
            )  # -1 for nks

            # for v_delta_precalc
            if self.vdp_path is not None and (
                self.phialpha_path is not None and self.gevdm_path is not None
            ):  # both file exist, choose newer ones
                if os.path.getmtime(self.vdp_path) >= os.path.getmtime(
                    self.phialpha_path
                ):  # phialpha and gevdm modified at the same time
                    self.phialpha_path = None
                    self.gevdm_path = None
                else:
                    self.vdp_path = None
            if self.vdp_path is not None:
                self.t_data["vdp"] = torch.tensor(
                    np.load(self.vdp_path).reshape(
                        raw_nframes, -1, self.nlocal, self.nlocal, self.natm, self.ndesc
                    )[conv]
                )
            elif self.phialpha_path is not None and self.gevdm_path is not None:
                phialpha = np.load(self.phialpha_path)
                nl = phialpha.shape[2]
                mmax = phialpha.shape[-1]
                self.t_data["phialpha"] = torch.tensor(
                    phialpha.reshape(raw_nframes, self.natm, nl, -1, self.nlocal, mmax)[conv]
                )  # -1 for nks
                self.t_data["gevdm"] = torch.tensor(
                    np.load(self.gevdm_path).reshape(raw_nframes, self.natm, nl, mmax, mmax, mmax)[conv]
                )

            # for phi labels and band labels
            if self.h_base_path is not None:
                self.t_data["h_base"] = torch.tensor(
                    np.load(self.h_base_path).reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv]
                )  # -1 for nks
            if self.h_ref_path is not None:
                h_ref = torch.tensor(np.load(self.h_ref_path))
                if self.read_overlap is True and self.overlap_path is not None:
                    overlap = torch.tensor(np.load(self.overlap_path))
                    L = torch.linalg.cholesky(overlap)
                    L_inv = torch.linalg.inv(L)
                    self.t_data["L_inv"] = L_inv.reshape(raw_nframes, -1, self.nlocal, self.nlocal)[
                        conv
                    ].clone()
                    band_ref, phi_ref = generalized_eigh(h_ref, L_inv)
                else:
                    band_ref, phi_ref = torch.linalg.eigh(h_ref, UPLO="U")  # U for upper triangle
                self.t_data["lb_band"] = band_ref.reshape(raw_nframes, -1, self.nlocal)[conv].clone()
                self.t_data["lb_phi"] = phi_ref.reshape(raw_nframes, -1, self.nlocal, self.nlocal)[
                    conv
                ].clone()
        # Hamiltonian in R space
        if self.hr_path is not None:
            self.t_data["lb_vdr"] = torch.tensor(np.load(self.hr_path)[conv])
            self.nlocal = self.t_data["lb_vdr"].shape[-1]
            if self.vdrp_path is not None and self.gevdm_path is not None:  # both file exist, choose newer ones
                if os.path.getmtime(self.vdrp_path) >= os.path.getmtime(
                    self.gevdm_path
                ):  # phialpha and gevdm modified at the same time
                    self.gevdm_path = None
                else:
                    self.vdrp_path = None
            if self.gevdm_path is not None:
                gevdm = np.load(self.gevdm_path)
                self.t_data["gevdm"] = torch.tensor(gevdm[conv])
                if (
                    self.orb_list is not None
                    and self.alpha_list is not None
                    and self.a_path is not None
                    and self.b_path is not None
                ):
                    types = torch.tensor(self.atom_info["elems"])
                    atoms = torch.tensor(self.atom_info["coords"])
                    box = torch.tensor(self.atom_info["lattice"])
                    orb, alpha, integrator = make_integrator(self.orb_list, self.alpha_list)
                    self.t_data["overlap"], self.t_data["iR_mat"], self.t_data["data_shape"] = cal_nb_overlap(
                        types, atoms, box, orb, alpha, integrator, self.nlocal
                    )
            elif self.vdrp_path is not None:
                self.t_data["vdrp"] = torch.tensor(np.load(self.vdrp_path)[conv])
        # Energy gradient
        if self.eg_path is not None and self.gveg_path is not None:
            self.t_data["eg0"] = torch.tensor(np.load(self.eg_path).reshape(raw_nframes, -1)[conv])
            self.t_data["gveg"] = torch.tensor(
                np.load(self.gveg_path).reshape(raw_nframes, self.natm, self.ndesc, -1)[conv]
            )
            self.neg = self.t_data["eg0"].shape[-1]
        # Density
        if self.gldv_path is not None:
            self.t_data["gldv"] = torch.tensor(
                np.load(self.gldv_path).reshape(raw_nframes, self.natm, self.ndesc)[conv]
            )

    def sample_train(self, index_list=None):
        """
        Sample a training batch from the reader in given order (index_list).
        If index_list is None, sample in random order.
        """
        if self.batch_size == self.nframes == 1:
            return self.sample_all()
        if len(self.idx_queue) < self.batch_size:
            if index_list is not None:
                self.idx_queue = np.array(index_list)
            else:
                self.idx_queue = np.random.choice(self.nframes, self.nframes, replace=False)
        sample_idx = self.idx_queue[: self.batch_size]
        self.idx_queue = self.idx_queue[self.batch_size :]
        out_dict = {}
        for k, v in self.t_data.items():
            if k in {"data_shape"}:
                out_dict[k] = v
            elif isinstance(v, torch.Tensor):
                out_dict[k] = v[sample_idx]
            elif isinstance(v, list):
                out_dict[k] = [v[i] for i in sample_idx]
        return out_dict

    def sample_all(self):
        return self.t_data

    def get_train_size(self):
        return self.nframes

    def get_batch_size(self):
        return self.batch_size

    def get_nframes(self):
        return self.nframes

    def collect_elems(self, elem_list):
        if "elem_list" in self.atom_info:
            assert list(elem_list) == list(self.atom_info["elem_list"])
            return self.atom_info["nelem"]
        elem_to_idx = np.zeros(200, dtype=int) + 200
        for ii, ee in enumerate(elem_list):
            elem_to_idx[ee] = ii
        idxs = elem_to_idx[self.atom_info["elems"]]
        nelem = np.zeros((self.nframes, len(elem_list)), int)
        np.add.at(nelem, (np.arange(nelem.shape[0]).reshape(-1, 1), idxs), 1)
        self.atom_info["nelem"] = nelem
        self.atom_info["elem_list"] = elem_list
        return nelem

    def subtract_elem_const(self, elem_const):
        econst = (self.atom_info["nelem"] @ elem_const).reshape(self.nframes, 1)
        self.data_ec -= econst
        self.t_data["lb_e"] -= econst
        self.atom_info["elem_const"] = elem_const

    def revert_elem_const(self):
        if "elem_const" not in self.atom_info:
            return
        elem_const = self.atom_info.pop("elem_const")
        econst = (self.atom_info["nelem"] @ elem_const).reshape(self.nframes, 1)
        self.data_ec += econst
        self.t_data["lb_e"] += econst
