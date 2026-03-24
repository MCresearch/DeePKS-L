"""ABACUS-related helpers for parsing geometry and sparse matrices."""

import numpy as np
import torch
from scipy.sparse import csr_matrix


def parse_xyz(filename):
    with open(filename) as fp:
        natom = int(fp.readline())
        comments = fp.readline().strip()
        atom_str = fp.readlines()
    atom_list = [a.split() for a in atom_str]
    elements = [a[0] for a in atom_list]
    coords = np.array([a[1:] for a in atom_list], dtype=float)
    return natom, comments, elements, coords


def R2iR(R):
    if isinstance(R, int):
        return 2 * R - 1 if R > 0 else -2 * R
    if isinstance(R, np.ndarray):
        return np.where(R > 0, 2 * R - 1, -2 * R)
    if torch.is_tensor(R):
        return torch.where(R > 0, 2 * R - 1, -2 * R)
    raise TypeError('R should be int, numpy.ndarray, or torch.Tensor')


def iR2R(iR):
    if isinstance(iR, int):
        assert iR >= 0, 'iR should be a non-negative integer'
        return -iR // 2 if iR % 2 == 0 else (iR + 1) // 2
    if isinstance(iR, np.ndarray):
        assert np.all(iR >= 0), 'iR should be non-negative'
        return np.where(iR % 2 == 0, -iR // 2, (iR + 1) // 2)
    if torch.is_tensor(iR):
        assert torch.all(iR >= 0), 'iR should be non-negative'
        return torch.where(iR % 2 == 0, -iR // 2, (iR + 1) // 2)
    raise TypeError('iR should be int, numpy.ndarray, or torch.Tensor')


def read_csr(file, dtype=torch.float64):
    all_indices = []
    all_values = []
    max_iR = 0
    dim = 0
    with open(file, 'r') as f:
        dim = int(f.readline().split()[-1])
        num = int(f.readline().split()[-1])
        for _ in range(num):
            nnz = 0
            while nnz == 0:
                r = f.readline().split()
                if len(r) == 0:
                    break
                Rx, Ry, Rz, nnz = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                iRx = R2iR(Rx)
                iRy = R2iR(Ry)
                iRz = R2iR(Rz)
                if max_iR < max(iRx, iRy, iRz):
                    max_iR = max(iRx, iRy, iRz)
            if nnz == 0:
                break
            data = [float(x) for x in f.readline().split()]
            indices = [int(x) for x in f.readline().split()]
            indptr = [int(x) for x in f.readline().split()]
            matrix = csr_matrix((data, indices, indptr), shape=(dim, dim))
            matrix_coo = matrix.tocoo()
            rows = matrix_coo.row
            cols = matrix_coo.col
            data = matrix_coo.data
            indices = np.vstack([
                np.full(nnz, iRx),
                np.full(nnz, iRy),
                np.full(nnz, iRz),
                rows,
                cols,
            ])
            all_indices.append(indices)
            all_values.append(data)
    final_indices = np.hstack(all_indices)
    final_values = np.concatenate(all_values)
    sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.tensor(final_indices, dtype=torch.long),
        values=torch.tensor(final_values, dtype=dtype),
        size=(max_iR + 1, max_iR + 1, max_iR + 1, dim, dim),
    )
    return sparse_tensor.coalesce()
