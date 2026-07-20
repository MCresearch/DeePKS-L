import os
import sys

import numpy as np
import torch

from deepks.io.schemas.reader_fields import DEFAULT_READER_FIELD_NAMES


class SimpleReader(object):
    def __init__(
        self,
        data_path,
        batch_size,
        e_name=DEFAULT_READER_FIELD_NAMES.e_name,
        d_name=DEFAULT_READER_FIELD_NAMES.d_name,
        conv_filter=True,
        conv_name=DEFAULT_READER_FIELD_NAMES.conv_name,
        **kwargs,
    ):
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
            sys_meta = np.loadtxt(os.path.join(self.data_path, "system.raw"), converters=float).astype(int).reshape([-1])
            self.natm = sys_meta[0]
            self.nproj = sys_meta[-1]
        except Exception:
            print("#", self.data_path, "no system.raw, infer meta from data", file=sys.stderr)
            sys_shape = np.load(os.path.join(self.data_path, f"{self.d_name[0]}.npy")).shape
            assert len(sys_shape) == 3, (
                f"{self.d_name[0]} has to be an order-3 array with shape [nframes, natom, nproj]"
            )
            self.natm = sys_shape[1]
            self.nproj = sys_shape[2]

    def prepare(self):
        self.index_count_all = 0
        data_ec = np.load(os.path.join(self.data_path, f"{self.e_name}.npy")).reshape([-1, 1])
        raw_nframes = data_ec.shape[0]
        data_dm = np.concatenate(
            [
                np.load(os.path.join(self.data_path, f"{dn}.npy")).reshape([raw_nframes, self.natm, -1])
                for dn in self.d_name
            ],
            axis=-1,
        )
        if self.c_filter:
            conv = np.load(os.path.join(self.data_path, f"{self.c_name}.npy")).reshape(raw_nframes)
        else:
            conv = np.ones(raw_nframes, dtype=bool)
        self.data_ec = data_ec[conv]
        self.data_dm = data_dm[conv]
        self.nframes = conv.sum()
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print("#", self.data_path, f"reset batch size to {self.batch_size}", file=sys.stderr)

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
            "eig": torch.from_numpy(self.data_dm[ind]),
        }

    def sample_all(self):
        return {
            "lb_e": torch.from_numpy(self.data_ec),
            "eig": torch.from_numpy(self.data_dm),
        }

    def get_train_size(self):
        return self.nframes

    def get_batch_size(self):
        return self.batch_size

    def get_nframes(self):
        return self.nframes
