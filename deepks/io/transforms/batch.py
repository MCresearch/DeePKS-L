"""Batch-level tensor/list splitting and concatenation helpers."""

from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import torch


def concat_batch(tdicts: Iterable[Dict[str, Any]], dim: int = 0) -> Dict[str, Any]:
    tdicts = list(tdicts)
    keys = tdicts[0].keys()
    assert all(d.keys() == keys for d in tdicts)
    return {k: torch.cat([d[k] for d in tdicts], dim) for k in keys}


def split_batch(
    tdict: Dict[str, Any],
    size: int,
    dim: int = 0,
    global_keys: Optional[Set[str]] = None,
):
    if global_keys is None:
        global_keys = {"data_shape"}
    dsplit = {}
    for k, v in tdict.items():
        if k in global_keys:
            dsplit[k] = v
        elif isinstance(v, torch.Tensor):
            dsplit[k] = torch.split(v, size, dim)
        elif isinstance(v, np.ndarray):
            assert dim == 0, "numpy.ndarray supports only for dim=0 split"
            dsplit[k] = np.array_split(v, range(size, v.shape[0], size), axis=0)
        elif isinstance(v, list):
            assert dim == 0, "list supports only for dim=0 split"
            dsplit[k] = [v[i : i + size] for i in range(0, len(v), size)]
        else:
            raise TypeError(f"Unsupported type for split_batch: {type(v)}")

    nsecs = [len(v) for k, v in dsplit.items() if k not in global_keys]
    assert all(ns == nsecs[0] for ns in nsecs)
    return [
        {k: (v[i] if k not in global_keys else v) for k, v in dsplit.items()}
        for i in range(nsecs[0])
    ]
