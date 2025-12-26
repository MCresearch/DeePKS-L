import os,time,sys
import numpy as np
import torch
from deepks.model.utils import safe_eigh

def concat_batch(tdicts, dim=0):
    keys = tdicts[0].keys()
    assert all(d.keys() == keys for d in tdicts)
    return {
        k: torch.cat([d[k] for d in tdicts], dim) 
        for k in keys
    }

def split_batch(tdict, size, dim=0):
    dsplit = {k: torch.split(v, size, dim) for k,v in tdict.items()}
    nsecs = [len(v) for v in dsplit.values()]
    assert all(ns == nsecs[0] for ns in nsecs)
    return [
        {k: v[i] for k, v in dsplit.items()}
        for i in range(nsecs[0])
    ]

def eigh_wrapper(a, use_safe_eigh=False):
    """
    Wrapper for eigendecomposition that supports safe gradients for degenerate cases.
    Args:
        a: Symmetric/Hermitian matrix.
        use_safe_eigh: If True, uses SafeEigh to prevent NaN gradients.
    """
    if use_safe_eigh:
        return safe_eigh(a)
    else:
        return torch.linalg.eigh(a, UPLO='U')

def generalized_eigh(h,trans_matrix,use_safe_eigh=False):
    symm_h=trans_matrix.mT @ h @ trans_matrix
    e,v=eigh_wrapper(symm_h, use_safe_eigh=use_safe_eigh)
    phi=trans_matrix @ v 
    return e,phi

class Reader(object):
    def __init__(self, data_path, batch_size, 
                 e_name="l_e_delta", d_name="dm_eig", 
                 f_name="l_f_delta", gvx_name="grad_vx", 
                 s_name="l_s_delta", gvepsl_name="grad_vepsl", 
                 o_name="l_o_delta", op_name="orbital_precalc",
                 h_name="l_h_delta", vdp_name="v_delta_precalc",
                 phialpha_name="phialpha",gevdm_name="grad_evdm",
                 h_base_name="h_base",h_ref_name="hamiltonian",
                 read_overlap = False, overlap_name="overlap",
                 eigh_method = 1,
                 eg_name="eg_base", gveg_name="grad_veg", 
                 gldv_name="grad_ldv", conv_name="conv", 
                 atom_name="atom", **kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.e_path = self.check_exist(e_name+".npy")
        self.f_path = self.check_exist(f_name+".npy")
        self.s_path = self.check_exist(s_name+".npy")
        self.o_path = self.check_exist(o_name+".npy")
        self.h_path = self.check_exist(h_name+".npy")
        self.h_base_path = self.check_exist(h_base_name+".npy")
        self.h_ref_path = self.check_exist(h_ref_name+".npy")
        self.overlap_path = self.check_exist(overlap_name+".npy")
        self.phialpha_path = self.check_exist(phialpha_name+".npy")
        self.gevdm_path = self.check_exist(gevdm_name+".npy")
        self.d_path = self.check_exist(d_name+".npy")
        self.gvx_path = self.check_exist(gvx_name+".npy")
        self.gvepsl_path = self.check_exist(gvepsl_name+".npy")
        self.op_path = self.check_exist(op_name+".npy")
        self.vdp_path = self.check_exist(vdp_name+".npy")
        self.eg_path = self.check_exist(eg_name+".npy")
        self.gveg_path = self.check_exist(gveg_name+".npy")
        self.gldv_path = self.check_exist(gldv_name+".npy")
        self.c_path = self.check_exist(conv_name+".npy")
        self.a_path = self.check_exist(atom_name+".npy")
        self.read_overlap = read_overlap
        self.eigh_method = eigh_method
        # load data
        self.load_meta()
        self.prepare()
        # initialize sample index queue
        self.idx_queue = []

    def check_exist(self, fname):
        if fname is None:
            return None
        fpath = os.path.join(self.data_path, fname)
        if os.path.exists(fpath):
            return fpath

    def load_meta(self):
        try:
            sys_meta = np.loadtxt(self.check_exist('system.raw'), converters = float).astype(int).reshape([-1])
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
        # Energy and descriptor
        # load data in torch
        self.t_data = {}
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
                    self.t_data["overlap"]=overlap\
                        .reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv].clone()
                    # When overlap matrix is ill-conditioned, the eigenvalues (i.e. band) can suffer from significant roundoff errors.
                    if self.eigh_method == 1:
                        L=torch.linalg.cholesky(overlap)
                        trans_matrix=torch.linalg.inv(L).mT
                    # Substitute cholesky with eigen decomposition.
                    # This modification effectively reorders the entries of symm_h, placing larger values towards the upper left-hand corner, thereby enhancing the precision in computing smaller eigenvalues
                    elif self.eigh_method == 2:
                        overlap_eigenvalue,overlap_eigenvector=torch.linalg.eigh(overlap)
                        # replace small eigenvalue with epsilon, avoid numerical instability
                        epsilon = 1e-16
                        overlap_eigenvalue = torch.clamp(overlap_eigenvalue, min=epsilon)
                        sigma_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(overlap_eigenvalue))
                        trans_matrix=overlap_eigenvector @ sigma_inv_sqrt
                    self.t_data["trans_matrix"]=trans_matrix\
                            .reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv].clone()  
                    band_ref,phi_ref=generalized_eigh(h_ref,trans_matrix)    
                else:
                    band_ref,phi_ref=torch.linalg.eigh(h_ref,UPLO='U') # U for upper triangle
                self.t_data["lb_band"]=band_ref\
                    .reshape(raw_nframes, -1, self.nlocal)[conv].clone()             
                self.t_data["lb_phi"]=phi_ref\
                    .reshape(raw_nframes, -1, self.nlocal, self.nlocal)[conv].clone()      
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
        if self.batch_size == self.nframes == 1:
            return self.sample_all()
        if len(self.idx_queue) < self.batch_size:
            if index_list is not None:
                self.idx_queue = np.array(index_list)
            else:
                self.idx_queue = np.random.choice(self.nframes, self.nframes, replace=False)
        sample_idx = self.idx_queue[:self.batch_size]
        self.idx_queue = self.idx_queue[self.batch_size:]
        return {k: v[sample_idx] for k, v in self.t_data.items()}

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
        

class GroupReader(object) :
    def __init__ (self, path_list, batch_size=1, group_batch=1, extra_label=True, **kwargs):
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
        self.sys_prob = [float(ii) for ii in self.nframes] / np.sum(self.nframes)
        
        self.group_batch = max(group_batch, 1)
        if self.group_batch > 1:
            self.group_dict = {}
            # self.group_index = {}
            for idx, r in enumerate(self.readers):
                shape = (r.natm, getattr(r, "neg", None))
                if shape in self.group_dict:
                    self.group_dict[shape].append(r)
                    # self.group_index[shape].append(idx)
                else:
                    self.group_dict[shape] = [r]
                    # self.group_index[shape] = [idx]
            self.group_prob = {n: sum(r.nframes for r in r_list) / sum(self.nframes)
                                for n, r_list in self.group_dict.items()}
            self.batch_prob_raw = {n: [r.nframes / r.batch_size for r in r_list] 
                                for n, r_list in self.group_dict.items()}
            self.batch_prob = {n: p / np.sum(p) for n, p in self.batch_prob_raw.items()}

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

    def sample_idx(self) :
        return np.random.choice(np.arange(self.nsystems), p=self.sys_prob)
        
    def sample_train(self, idx=None, index_list=None):
        if idx is None:
            idx = self.sample_idx()
        return self.readers[idx].sample_train(index_list=index_list)

    def sample_train_group(self):
        cidx = np.random.choice(len(self.group_prob), p=list(self.group_prob.values()))
        cshape = list(self.group_prob.keys())[cidx]
        cgrp = self.group_dict[cshape]
        csys = np.random.choice(cgrp, self.group_batch, p=self.batch_prob[cshape])
        batch = concat_batch([s.sample_train() for s in csys], dim=0)
        return batch

    def sample_all(self, idx=None) :
        if idx is None:
            idx = self.sample_idx()
        return \
            self.readers[idx].sample_all()
    
    def sample_all_batch(self, idx=None):
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
                 e_name="l_e_delta", d_name="dm_eig", 
                 conv_filter=True, conv_name="conv", **kwargs):
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