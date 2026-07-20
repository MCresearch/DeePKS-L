"""PySCF backend fallback settings."""

import numpy as np

_zeta = 1.5 ** np.array([17, 13, 10, 7, 5, 3, 2, 1, 0, -1, -2, -3])
_coef = np.diag(np.ones(_zeta.size)) - np.diag(np.ones(_zeta.size - 1), k=1)
_table = np.concatenate([_zeta.reshape(-1, 1), _coef], axis=1)

DEFAULT_BASIS = [[0, *_table.tolist()], [1, *_table.tolist()], [2, *_table.tolist()]]
DEFAULT_SYMB = "Ne"
DEFAULT_UNIT = "Bohr"

DEFAULT_HF_ARGS = {
    "conv_tol": 1e-9,
}

DEFAULT_SCF_ARGS = {
    "conv_tol": 1e-7,
}

MOL_ATTRIBUTE = {"charge", "basis", "unit"}
