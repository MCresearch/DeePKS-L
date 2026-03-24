"""Basis helpers for the PySCF backend."""

import numpy as np

from deepks.physics.defaults import DEFAULT_BASIS, DEFAULT_SYMB
from deepks.io.utils import load_array


def load_basis(basis):
    if basis is None:
        return DEFAULT_BASIS
    if isinstance(basis, np.ndarray) and basis.ndim == 2:
        return [[ll, *basis.tolist()] for ll in range(3)]
    if not isinstance(basis, str):
        return basis
    if basis.endswith('.npy'):
        table = np.load(basis)
        return [[ll, *table.tolist()] for ll in range(3)]
    if basis.endswith('.npz'):
        all_tables = np.load(basis)
        return [
            [int(name.split('_L')[-1]) if '_L' in name else ii, *table.tolist()]
            for ii, (name, table) in enumerate(all_tables.items())
        ]
    from pyscf import gto

    symb = DEFAULT_SYMB
    if '@' in basis:
        basis, symb = basis.split('@')
    return gto.basis.load(basis, symb=symb)


def save_basis(file, basis):
    tables = {f'arr_{i}_L{l}': np.array(b) for i, (l, *b) in enumerate(basis)}
    np.savez(file, **tables)


def get_shell_sec(basis):
    if not isinstance(basis, (list, tuple)):
        basis = load_basis(basis)
    shell_sec = []
    for l, c0, *cr in basis:
        nb = c0 if isinstance(c0, int) else (len(c0) - 1)
        shell_sec.extend([2 * l + 1] * nb)
    return shell_sec
