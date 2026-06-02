import os
import sys

import numpy as np
import torch

from deepks.io.schemas.reader_fields import (
    DEFAULT_READER_FIELD_NAMES,
    ReaderFieldNames,
    resolve_reader_paths,
)
from deepks.io.task_batches import sample_to_task_batch
from .data_loading import build_reader_tensor_data, load_reader_raw_data


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
        eigh_method = 1,
        eg_name=DEFAULT_READER_FIELD_NAMES.eg_name,
        gveg_name=DEFAULT_READER_FIELD_NAMES.gveg_name,
        gldv_name=DEFAULT_READER_FIELD_NAMES.gldv_name,
        conv_name=DEFAULT_READER_FIELD_NAMES.conv_name,
        atom_name=DEFAULT_READER_FIELD_NAMES.atom_name,
        box_name=DEFAULT_READER_FIELD_NAMES.box_name,
        iR_mat_name=DEFAULT_READER_FIELD_NAMES.iR_mat_name,
        phialpha_r_name=DEFAULT_READER_FIELD_NAMES.phialpha_r_name,
        orb_list=None,
        alpha_list=None,
        hamiltonian_level_names=None,
        hamiltonian_name=None,
        csr_hr_name=None,
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
            iR_mat_name=iR_mat_name,
            phialpha_r_name=phialpha_r_name,
        )
        for path_name, path in resolve_reader_paths(self.data_path, field_names).items():
            setattr(self, path_name, path)

        self.system_raw_path = os.path.join(self.data_path, "system.raw")
        self.read_overlap = read_overlap
        self.eigh_method = eigh_method
        self.orb_list = ["../../" + orb for orb in orb_list] if orb_list is not None else None
        self.alpha_list = ["../../" + alpha for alpha in alpha_list] if alpha_list is not None else None
        self.hamiltonian_level_names = list(hamiltonian_level_names) if hamiltonian_level_names else None
        self.hamiltonian_name = hamiltonian_name
        self.csr_hr_name = csr_hr_name
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
        raw_data = load_reader_raw_data(
            data_path=self.data_path,
            e_path=self.e_path,
            d_path=self.d_path,
            c_path=self.c_path,
            a_path=self.a_path,
            b_path=self.b_path,
            natm=self.natm,
            ndesc=self.ndesc,
        )
        self.data_ec = raw_data["data_ec"]
        self.data_dm = raw_data["data_dm"]
        self.atom_info = raw_data["atom_info"]
        self.nframes = raw_data["conv"].sum()
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print("#", self.data_path, f"reset batch size to {self.batch_size}", file=sys.stderr)
        self.t_data, extra = build_reader_tensor_data(
            raw_data=raw_data,
            natm=self.natm,
            ndesc=self.ndesc,
            f_path=self.f_path,
            gvx_path=self.gvx_path,
            s_path=self.s_path,
            gvepsl_path=self.gvepsl_path,
            o_path=self.o_path,
            op_path=self.op_path,
            h_path=self.h_path,
            vdp_path=self.vdp_path,
            phialpha_path=self.phialpha_path,
            gevdm_path=self.gevdm_path,
            h_base_path=self.h_base_path,
            h_ref_path=self.h_ref_path,
            read_overlap=self.read_overlap,
            overlap_path=self.overlap_path,
            eigh_method=self.eigh_method,
            hr_path=self.hr_path,
            vdrp_path=self.vdrp_path,
            orb_list=self.orb_list,
            alpha_list=self.alpha_list,
            eg_path=self.eg_path,
            gveg_path=self.gveg_path,
            gldv_path=self.gldv_path,
            iR_mat_path=self.iR_mat_path,
            phialpha_r_path=self.phialpha_r_path,
            hamiltonian_level_names=self.hamiltonian_level_names,
            hamiltonian_name=self.hamiltonian_name,
            csr_hr_name=self.csr_hr_name,
        )
        if "nlocal" in extra:
            self.nlocal = extra["nlocal"]
        if "neg" in extra:
            self.neg = extra["neg"]

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

    def sample_train_task_batch(self, index_list=None):
        return sample_to_task_batch(self.sample_train(index_list=index_list))

    def sample_all_task_batch(self):
        return sample_to_task_batch(self.sample_all())

    def get_display_fields(self):
        return self.sample_all_task_batch().display_keys()

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
