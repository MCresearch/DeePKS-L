import os
import sys
import numpy as np
import torch

from deepks.io.schemas.reader_fields import (
    DEFAULT_READER_FIELD_NAMES,
    ReaderFieldNames,
    resolve_reader_paths,
)
from deepks.io.transforms.batch import concat_batch, split_batch
from deepks.io.transforms.linalg import generalized_eigh
from deepks.core.ml.utils import make_integrator, cal_nb_overlap

class Reader(object):
    def __init__(self, data_path, batch_size, 
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
                 orb_list=None, alpha_list=None, **kwargs):
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
        except:
            print('#', self.data_path, f"no system.raw, infer meta from data", file=sys.stderr)
            sys_shape = np.load(self.d_path).shape
            assert len(sys_shape) == 3, \
                f"descriptor has to be an order-3 array with shape [nframes, natom, nproj]"
            self.natm = sys_shape[1]
            self.nproj = sys_shape[2]
        self.ndesc = self.nproj

    def prepare(self):
        ## Load energy and check nframes
        data_ec = np.load(self.e_path).reshape(-1, 1) # energy
        raw_nframes = data_ec.shape[0] # number of total frames
        data_dm = np.load(self.d_path).reshape(raw_nframes, self.natm, self.ndesc) # descriptor
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
            print('#', self.data_path, 
                 f"reset batch size to {self.batch_size}", file=sys.stderr)

        ## Handle atom and element data
        self.atom_info = {}
        if self.a_path is not None:
            atoms = np.load(self.a_path).reshape(raw_nframes, self.natm, 4) # atom.npy
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
            self.t_data["lb_f"] = torch.tensor(
                np.load(self.f_path)\
                  .reshape(raw_nframes, -1, 3)[conv])
            self.t_data["gvx"] = torch.tensor(
                np.load(self.gvx_path)\
                  .reshape(raw_nframes, self.natm, 3, self.natm, self.ndesc)[conv])
        # Stress
        if self.s_path is not None and self.gvepsl_path is not None:
            self.t_data["lb_s"] = torch.tensor(
                np.load(self.s_path)\
                  .reshape(raw_nframes, 6)[conv])
            self.t_data["gvepsl"] = torch.tensor(
                np.load(self.gvepsl_path)\
                  .reshape(raw_nframes, 6, self.natm, self.ndesc)[conv])
        # Orbital
        if self.o_path is not None and self.op_path is not None:
            self.t_data["lb_o"] = torch.tensor(
                np.load(self.o_path)[conv])
            self.t_data["op"] = torch.tensor(
                np.load(self.op_path)[conv])
        # Hamiltonian in k space
        if self.h_path is not None and (self.vdp_path is not None or (self.phialpha_path is not None and self.gevdm_path is not None)):
            h_shape = np.load(self.h_path).shape
            assert h_shape[-1] == h_shape[-2], \
                f"The last two dimension of H must have the same size , which is nlocal"
            self.nlocal = h_shape[-1]
            self.t_data["lb_vd"] = torch.tensor(
                np.load(self.h_path)\
                  .reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv]) #-1 for nks
            
            # for v_delta_precalc
            if self.vdp_path is not None and (self.phialpha_path is not None and self.gevdm_path is not None): #both file exist, choose newer ones
                if os.path.getmtime(self.vdp_path) >= os.path.getmtime(self.phialpha_path):#phialpha and gevdm modified at the same time
                    self.phialpha_path=None
                    self.gevdm_path=None
                else:
                    self.vdp_path=None
            if self.vdp_path is not None:
                self.t_data["vdp"] = torch.tensor(
                    np.load(self.vdp_path)\
                        .reshape(raw_nframes, -1, self.nlocal, self.nlocal, self.natm, self.ndesc)[conv])
            elif self.phialpha_path is not None and self.gevdm_path is not None:
                phialpha=np.load(self.phialpha_path)
                nl=phialpha.shape[2]
                mmax=phialpha.shape[-1]
                self.t_data["phialpha"] = torch.tensor(
                    phialpha\
                        .reshape(raw_nframes, self.natm, nl, -1, self.nlocal, mmax)[conv])#-1 for nks
                self.t_data["gevdm"] = torch.tensor(
                    np.load(self.gevdm_path)\
                      .reshape(raw_nframes, self.natm, nl, mmax, mmax, mmax)[conv]) 
                # self.t_data["vdp"] = vdp\
                #         .reshape(raw_nframes, -1, self.nlocal, self.nlocal, self.natm, self.ndesc)[conv].clone()
            
            # for phi labels and band labels
            if self.h_base_path is not None:
                self.t_data["h_base"]=torch.tensor(
                    np.load(self.h_base_path)\
                    .reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv]) #-1 for nks
            if self.h_ref_path is not None:
                h_ref=torch.tensor(np.load(self.h_ref_path))
                if self.read_overlap is True and self.overlap_path is not None:
                    #print("use generalized eigh")
                    overlap=torch.tensor(np.load(self.overlap_path))
                    L=torch.linalg.cholesky(overlap)
                    L_inv=torch.linalg.inv(L)
                    self.t_data["L_inv"]=L_inv\
                            .reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv].clone()  
                    band_ref,phi_ref=generalized_eigh(h_ref,L_inv)    
                else:
                    band_ref,phi_ref=torch.linalg.eigh(h_ref,UPLO='U') # U for upper triangle
                self.t_data["lb_band"]=band_ref\
                    .reshape(raw_nframes, -1, self.nlocal)[conv].clone()             
                self.t_data["lb_phi"]=phi_ref\
                    .reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv].clone()      
        # Hamiltonian in R space
        if self.hr_path is not None:
            self.t_data["lb_vdr"] = torch.tensor(np.load(self.hr_path)[conv])
            self.nlocal = self.t_data["lb_vdr"].shape[-1]
            if self.vdrp_path is not None and self.gevdm_path is not None: #both file exist, choose newer ones
                if os.path.getmtime(self.vdrp_path) >= os.path.getmtime(self.gevdm_path):#phialpha and gevdm modified at the same time
                    self.gevdm_path=None
                else:
                    self.vdrp_path=None
            if self.gevdm_path is not None:
                gevdm = np.load(self.gevdm_path)
                self.t_data["gevdm"] = torch.tensor(gevdm[conv])
                if self.orb_list is not None and self.alpha_list is not None and self.a_path is not None and self.b_path is not None:
                    types = torch.tensor(self.atom_info["elems"])
                    atoms = torch.tensor(self.atom_info["coords"])
                    box = torch.tensor(self.atom_info["lattice"])
                    orb, alpha, integrator = make_integrator(self.orb_list, self.alpha_list)
                    self.t_data["overlap"], self.t_data["iR_mat"], self.t_data["data_shape"] = \
                        cal_nb_overlap(types, atoms, box, orb, alpha, integrator, self.nlocal)
            elif self.vdrp_path is not None:
                self.t_data["vdrp"] = torch.tensor(np.load(self.vdrp_path)[conv])
        # Energy gradient
        if self.eg_path is not None and self.gveg_path is not None:
            self.t_data['eg0'] = torch.tensor(
                np.load(self.eg_path)\
                  .reshape(raw_nframes, -1)[conv])
            self.t_data["gveg"] = torch.tensor(
                np.load(self.gveg_path)\
                  .reshape(raw_nframes, self.natm, self.ndesc, -1)[conv])
            self.neg = self.t_data['eg0'].shape[-1]
        # Density
        if self.gldv_path is not None:
            self.t_data["gldv"] = torch.tensor(
                np.load(self.gldv_path)\
                  .reshape(raw_nframes, self.natm, self.ndesc)[conv])

    def sample_train(self, index_list=None):
        '''
        Sample a training batch from the reader in given order (index_list).
        If index_list is None, sample in random order.
        '''
        if self.batch_size == self.nframes == 1:
            return self.sample_all()
        if len(self.idx_queue) < self.batch_size:
            if index_list is not None:
                self.idx_queue = np.array(index_list)
            else:
                self.idx_queue = np.random.choice(self.nframes, self.nframes, replace=False)
        sample_idx = self.idx_queue[:self.batch_size]
        self.idx_queue = self.idx_queue[self.batch_size:]
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
        np.add.at(nelem, (np.arange(nelem.shape[0]).reshape(-1,1), idxs), 1)
        self.atom_info["nelem"] = nelem
        self.atom_info["elem_list"] = elem_list
        return nelem
    
    def subtract_elem_const(self, elem_const):
        # assert "elem_const" not in self.atom_info, \
        #     "subtract_elem_const has been done. The method should not be executed twice."
        econst = (self.atom_info["nelem"] @ elem_const).reshape(self.nframes, 1)
        self.data_ec -= econst
        self.t_data["lb_e"] -= econst
        self.atom_info["elem_const"] = elem_const
    
    def revert_elem_const(self):
        # assert "elem_const" not in self.atom_info, \
        #     "subtract_elem_const has been done. The method should not be executed twice."
        if "elem_const" not in self.atom_info:
            return
        elem_const = self.atom_info.pop("elem_const")
        econst = (self.atom_info["nelem"] @ elem_const).reshape(self.nframes, 1)
        self.data_ec += econst
        self.t_data["lb_e"] += econst
        

class GroupReader(object):
    def __init__(self, path_list, batch_size=1, group_batch=1, extra_label=True, **kwargs):
        if isinstance(path_list, str):
            path_list = [path_list]
        self.path_list = path_list
        self.batch_size = batch_size
        # init system readers
        self.readers = []
        self.nframes = []
        for ipath in self.path_list :
            ireader = Reader(ipath, batch_size, **kwargs)
            if ireader.get_nframes() == 0:
                print('# ignore empty dataset:', ipath, file=sys.stderr)
                continue
            self.readers.append(ireader)
            self.nframes.append(ireader.get_nframes())
        if not self.readers:
            raise RuntimeError("No system is avaliable")
        self.nsystems = len(self.readers)
        data_keys = self.readers[0].sample_all().keys()
        print(f"# load {self.nsystems} systems with fields {list(dict.fromkeys(data_keys))}")
        # probability of each system
        self.ndesc = self.readers[0].ndesc
        self.sys_prob = np.asarray(self.nframes, dtype=float)
        self.sys_prob = self.sys_prob / self.sys_prob.sum()
        
        self.group_batch = max(group_batch, 1)
        if self.group_batch > 1:
            self._build_group_sampling_cache()

        self._sample_used = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._sample_used > self.get_train_size():
            self._sample_used = 0
            raise StopIteration
        sample = self.sample_train() if self.group_batch == 1 else self.sample_train_group()
        self._sample_used += sample["lb_e"].shape[0]
        return sample

    @staticmethod
    def _reader_shape(reader):
        return (reader.natm, getattr(reader, "neg", None))

    def _build_group_sampling_cache(self):
        """Cache grouped readers and sampling probabilities for grouped batches."""
        self.group_dict = {}
        for reader in self.readers:
            shape = self._reader_shape(reader)
            self.group_dict.setdefault(shape, []).append(reader)

        total_frames = float(np.sum(self.nframes))
        self.group_prob = {
            shape: sum(reader.nframes for reader in readers) / total_frames
            for shape, readers in self.group_dict.items()
        }
        self.batch_prob_raw = {
            shape: np.asarray([reader.nframes / reader.batch_size for reader in readers], dtype=float)
            for shape, readers in self.group_dict.items()
        }
        self.batch_prob = {
            shape: probs / probs.sum()
            for shape, probs in self.batch_prob_raw.items()
        }

        # Avoid rebuilding dict key/value lists on every sample.
        self._group_shapes = tuple(self.group_prob.keys())
        self._group_probs = np.asarray(list(self.group_prob.values()), dtype=float)

    def sample_idx(self) :
        '''
        Sample a system (group) index based on the system probability distribution.
        '''
        return np.random.choice(np.arange(self.nsystems), p=self.sys_prob)
        
    def sample_train(self, idx=None, index_list=None):
        '''
        Sample a training batch from a specific system reader (idx) in given order (index_list).
        If idx is None, sample from a random system reader.
        If index_list is None, sample in random order.
        '''
        if idx is None:
            idx = self.sample_idx()
        return self.readers[idx].sample_train(index_list=index_list)

    def sample_train_group(self):
        '''
        Sample a big batch from `group_batch` systems and `batch_size` frames from each system.
        The systems are sampled based on the group probability distribution.
        The batch size is `group_batch * batch_size`. 
        '''
        cidx = np.random.choice(len(self._group_shapes), p=self._group_probs)
        cshape = self._group_shapes[cidx]
        cgrp = self.group_dict[cshape]
        csys = np.random.choice(cgrp, self.group_batch, p=self.batch_prob[cshape])
        batch = concat_batch([s.sample_train() for s in csys], dim=0)
        return batch

    def sample_all(self, idx=None) :
        '''
        Sample all data from a specific system reader (idx).
        If idx is None, sample from a random system reader.
        This method is used to get all data for training or evaluation.
        '''
        if idx is None:
            idx = self.sample_idx()
        return self.readers[idx].sample_all()
    
    def sample_all_batch(self, idx=None):
        '''
        Sample all data from a specific system reader (idx) in batches.
        If idx is None, sample data from all systems in batches in system order.
        '''
        if idx is not None:
            all_data = self.sample_all(idx)
            size = self.batch_size * self.group_batch
            yield from split_batch(all_data, size, dim=0)
        else:
            for i in range(self.nsystems):
                yield from self.sample_all_batch(i)

    def get_train_size(self) :
        return np.sum(self.nframes)

    def get_batch_size(self) :
        return self.batch_size

    def compute_data_stat(self, symm_sections=None):
        all_dm = np.concatenate([r.data_dm.reshape(-1,r.ndesc) for r in self.readers])
        if symm_sections is None:
            all_mean, all_std = all_dm.mean(0), all_dm.std(0)
        else:
            assert sum(symm_sections) == all_dm.shape[-1]
            dm_shells = np.split(all_dm, np.cumsum(symm_sections)[:-1], axis=-1)
            mean_shells = [d.mean().repeat(s) for d, s in zip(dm_shells, symm_sections)]
            std_shells = [d.std().repeat(s) for d, s in zip(dm_shells, symm_sections)]
            all_mean = np.concatenate(mean_shells, axis=-1)
            all_std = np.concatenate(std_shells, axis=-1)
        return all_mean, all_std

    def compute_prefitting(self, shift=None, scale=None, ridge_alpha=1e-8, symm_sections=None):
        if shift is None or scale is None:
            all_mean, all_std = self.compute_data_stat(symm_sections=symm_sections)
            if shift is None:
                shift = all_mean
            if scale is None:
                scale = all_std
        all_sdm = np.concatenate([((r.data_dm - shift) / scale).sum(1) for r in self.readers])
        all_natm = np.concatenate([[float(r.data_dm.shape[1])]*r.data_dm.shape[0] for r in self.readers])
        if symm_sections is not None: # in this case ridge alpha cannot be 0
            assert sum(symm_sections) == all_sdm.shape[-1]
            sdm_shells = np.split(all_sdm, np.cumsum(symm_sections)[:-1], axis=-1)
            all_sdm = np.stack([d.sum(-1) for d in sdm_shells], axis=-1)
        # build feature matrix
        X = np.concatenate([all_sdm, all_natm.reshape(-1,1)], -1)
        y = np.concatenate([r.data_ec for r in self.readers])
        I = np.identity(X.shape[1])
        I[-1,-1] = 0 # do not punish the bias term
        # solve ridge reg
        coef = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ y).reshape(-1)
        weight, bias = coef[:-1], coef[-1]
        if symm_sections is not None:
            weight = np.concatenate([w.repeat(s) for w, s in zip(weight, symm_sections)], axis=-1)
        return weight, bias
    
    def collect_elems(self, elem_list=None):
        if elem_list is None:
            elem_list = np.array(sorted(set.union(*[
                set(r.atom_info["elems"].flatten()) for r in self.readers
            ])))
        for rd in self.readers:
            rd.collect_elems(elem_list)
        return elem_list

    def compute_elem_const(self, ridge_alpha=0.):
        elem_list = self.collect_elems()
        all_nelem = np.concatenate([r.atom_info["nelem"] for r in self.readers])
        all_ec = np.concatenate([r.data_ec for r in self.readers])
        # lex sort by nelem
        lexidx = np.lexsort(all_nelem.T)
        all_nelem = all_nelem[lexidx]
        all_ec = all_ec[lexidx]
        # group by nelem
        _, sidx = np.unique(all_nelem, return_index=True, axis=0)
        sidx = np.sort(sidx)
        grp_nelem = all_nelem[sidx]
        grp_ec = np.array(list(map(np.mean, np.split(all_ec, sidx[1:]))))
        if ridge_alpha <= 0:
            elem_const, _res, _rank, _sing = np.linalg.lstsq(grp_nelem, grp_ec, None)
        else:
            I = np.identity(grp_nelem.shape[1])
            elem_const = np.linalg.solve(
                grp_nelem.T @ grp_nelem + ridge_alpha * I, grp_nelem.T @ grp_ec)
        return elem_list.reshape(-1), elem_const.reshape(-1)
    
    def subtract_elem_const(self, elem_const):
        for rd in self.readers:
            rd.subtract_elem_const(elem_const)
    
    def revert_elem_const(self):
        for rd in self.readers:
            rd.revert_elem_const()


class SimpleReader(object):
    def __init__(self, data_path, batch_size, 
                 e_name=DEFAULT_READER_FIELD_NAMES.e_name,
                 d_name=DEFAULT_READER_FIELD_NAMES.d_name,
                 conv_filter=True,
                 conv_name=DEFAULT_READER_FIELD_NAMES.conv_name,
                 **kwargs):
        # copy from config
        self.data_path = data_path
        self.batch_size = batch_size
        self.e_name = e_name
        self.d_name = d_name if isinstance(d_name, (list, tuple)) else [d_name]
        self.c_filter = conv_filter
        self.c_name = conv_name
        self.load_meta()
        self.prepare()

    def load_meta(self):
        try:
            sys_meta = np.loadtxt(os.path.join(self.data_path,'system.raw'), converters = float).astype(int).reshape([-1])
            self.natm = sys_meta[0]
            self.nproj = sys_meta[-1]
        except:
            print('#', self.data_path, f"no system.raw, infer meta from data", file=sys.stderr)
            sys_shape = np.load(os.path.join(self.data_path, f'{self.d_name[0]}.npy')).shape
            assert len(sys_shape) == 3, \
                f"{self.d_name[0]} has to be an order-3 array with shape [nframes, natom, nproj]"
            self.natm = sys_shape[1]
            self.nproj = sys_shape[2]
    
    def prepare(self):
        self.index_count_all = 0
        data_ec = np.load(os.path.join(self.data_path,f'{self.e_name}.npy')).reshape([-1, 1])
        raw_nframes = data_ec.shape[0]
        data_dm = np.concatenate(
            [np.load(os.path.join(self.data_path,f'{dn}.npy'))\
               .reshape([raw_nframes, self.natm, -1])
            for dn in self.d_name], 
            axis=-1)
        if self.c_filter:
            conv = np.load(os.path.join(self.data_path,f'{self.c_name}.npy')).reshape(raw_nframes)
        else:
            conv = np.ones(raw_nframes, dtype=bool)
        self.data_ec = data_ec[conv]
        self.data_dm = data_dm[conv]
        self.nframes = conv.sum()
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print('#', self.data_path, f"reset batch size to {self.batch_size}", file=sys.stderr)
    
    def sample_train(self):
        if self.nframes == self.batch_size == 1:
            return self.sample_all()
        self.index_count_all += self.batch_size
        if self.index_count_all > self.nframes:
            # shuffle the data
            self.index_count_all = self.batch_size
            ind = np.random.choice(self.nframes, self.nframes, replace=False)
            self.data_ec = self.data_ec[ind]
            self.data_dm = self.data_dm[ind]
        ind = np.arange(self.index_count_all - self.batch_size, self.index_count_all)
        return {
            "lb_e": torch.from_numpy(self.data_ec[ind]), 
            "eig": torch.from_numpy(self.data_dm[ind])
        }

    def sample_all(self) :
        return {
            "lb_e": torch.from_numpy(self.data_ec), 
            "eig": torch.from_numpy(self.data_dm)
        }

    def get_train_size(self) :
        return self.nframes

    def get_batch_size(self) :
        return self.batch_size

    def get_nframes(self) :
        return self.nframes