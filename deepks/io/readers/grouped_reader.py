import sys

import numpy as np

from deepks.io.readers.reader import Reader
from deepks.io.readers.sampling import build_group_sampling_cache, build_system_probabilities
from deepks.io.readers.stats import (
    collect_elems,
    compute_data_stat,
    compute_elem_const,
    compute_prefitting,
    revert_elem_const,
    subtract_elem_const,
)
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
        self.sys_prob = build_system_probabilities(self.nframes)

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

    def _build_group_sampling_cache(self):
        """Cache grouped readers and sampling probabilities for grouped batches."""
        cache = build_group_sampling_cache(self.readers)
        self.group_dict = cache["group_dict"]
        self.group_prob = cache["group_prob"]
        self.batch_prob_raw = cache["batch_prob_raw"]
        self.batch_prob = cache["batch_prob"]
        self._group_shapes = cache["group_shapes"]
        self._group_probs = cache["group_probs"]

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
        return compute_data_stat(self.readers, symm_sections=symm_sections)

    def compute_prefitting(self, shift=None, scale=None, ridge_alpha=1e-8, symm_sections=None):
        return compute_prefitting(
            self.readers,
            shift=shift,
            scale=scale,
            ridge_alpha=ridge_alpha,
            symm_sections=symm_sections,
        )

    def collect_elems(self, elem_list=None):
        return collect_elems(self.readers, elem_list=elem_list)

    def compute_elem_const(self, ridge_alpha=0.0):
        return compute_elem_const(self.readers, ridge_alpha=ridge_alpha)

    def subtract_elem_const(self, elem_const):
        subtract_elem_const(self.readers, elem_const)

    def revert_elem_const(self):
        revert_elem_const(self.readers)
