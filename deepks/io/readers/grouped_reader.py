import sys

import numpy as np

from deepks.io.readers.reader import Reader
from deepks.io.transforms.batch import concat_batch, split_batch


class GroupReader(object):
    def __init__(self, path_list, batch_size=1, group_batch=1, extra_label=True, **kwargs):
        if isinstance(path_list, str):
            path_list = [path_list]
        self.path_list = path_list
        self.batch_size = batch_size
        # init system readers
        self.readers = []
        self.nframes = []
        for ipath in self.path_list:
            ireader = Reader(ipath, batch_size, **kwargs)
            if ireader.get_nframes() == 0:
                print("# ignore empty dataset:", ipath, file=sys.stderr)
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
        self.batch_prob = {shape: probs / probs.sum() for shape, probs in self.batch_prob_raw.items()}

        # Avoid rebuilding dict key/value lists on every sample.
        self._group_shapes = tuple(self.group_prob.keys())
        self._group_probs = np.asarray(list(self.group_prob.values()), dtype=float)

    def sample_idx(self):
        """
        Sample a system (group) index based on the system probability distribution.
        """
        return np.random.choice(np.arange(self.nsystems), p=self.sys_prob)

    def sample_train(self, idx=None, index_list=None):
        """
        Sample a training batch from a specific system reader (idx) in given order (index_list).
        If idx is None, sample from a random system reader.
        If index_list is None, sample in random order.
        """
        if idx is None:
            idx = self.sample_idx()
        return self.readers[idx].sample_train(index_list=index_list)

    def sample_train_group(self):
        """
        Sample a big batch from `group_batch` systems and `batch_size` frames from each system.
        The systems are sampled based on the group probability distribution.
        The batch size is `group_batch * batch_size`.
        """
        cidx = np.random.choice(len(self._group_shapes), p=self._group_probs)
        cshape = self._group_shapes[cidx]
        cgrp = self.group_dict[cshape]
        csys = np.random.choice(cgrp, self.group_batch, p=self.batch_prob[cshape])
        batch = concat_batch([s.sample_train() for s in csys], dim=0)
        return batch

    def sample_all(self, idx=None):
        """
        Sample all data from a specific system reader (idx).
        If idx is None, sample from a random system reader.
        This method is used to get all data for training or evaluation.
        """
        if idx is None:
            idx = self.sample_idx()
        return self.readers[idx].sample_all()

    def sample_all_batch(self, idx=None):
        """
        Sample all data from a specific system reader (idx) in batches.
        If idx is None, sample data from all systems in batches in system order.
        """
        if idx is not None:
            all_data = self.sample_all(idx)
            size = self.batch_size * self.group_batch
            yield from split_batch(all_data, size, dim=0)
        else:
            for i in range(self.nsystems):
                yield from self.sample_all_batch(i)

    def get_train_size(self):
        return np.sum(self.nframes)

    def get_batch_size(self):
        return self.batch_size

    def compute_data_stat(self, symm_sections=None):
        all_dm = np.concatenate([r.data_dm.reshape(-1, r.ndesc) for r in self.readers])
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
        all_natm = np.concatenate([[float(r.data_dm.shape[1])] * r.data_dm.shape[0] for r in self.readers])
        if symm_sections is not None:  # in this case ridge alpha cannot be 0
            assert sum(symm_sections) == all_sdm.shape[-1]
            sdm_shells = np.split(all_sdm, np.cumsum(symm_sections)[:-1], axis=-1)
            all_sdm = np.stack([d.sum(-1) for d in sdm_shells], axis=-1)
        # build feature matrix
        X = np.concatenate([all_sdm, all_natm.reshape(-1, 1)], -1)
        y = np.concatenate([r.data_ec for r in self.readers])
        I = np.identity(X.shape[1])
        I[-1, -1] = 0  # do not punish the bias term
        # solve ridge reg
        coef = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ y).reshape(-1)
        weight, bias = coef[:-1], coef[-1]
        if symm_sections is not None:
            weight = np.concatenate([w.repeat(s) for w, s in zip(weight, symm_sections)], axis=-1)
        return weight, bias

    def collect_elems(self, elem_list=None):
        if elem_list is None:
            elem_list = np.array(sorted(set.union(*[set(r.atom_info["elems"].flatten()) for r in self.readers])))
        for rd in self.readers:
            rd.collect_elems(elem_list)
        return elem_list

    def compute_elem_const(self, ridge_alpha=0.0):
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
            elem_const = np.linalg.solve(grp_nelem.T @ grp_nelem + ridge_alpha * I, grp_nelem.T @ grp_ec)
        return elem_list.reshape(-1), elem_const.reshape(-1)

    def subtract_elem_const(self, elem_const):
        for rd in self.readers:
            rd.subtract_elem_const(elem_const)

    def revert_elem_const(self):
        for rd in self.readers:
            rd.revert_elem_const()
