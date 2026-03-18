import numpy as np


__all__ = [
    "collect_elems",
    "compute_data_stat",
    "compute_elem_const",
    "compute_prefitting",
    "revert_elem_const",
    "subtract_elem_const",
]


def _ensure_readers(readers):
    if not readers:
        raise ValueError("readers must not be empty")


def compute_data_stat(readers, symm_sections=None):
    _ensure_readers(readers)
    all_dm = np.concatenate([reader.data_dm.reshape(-1, reader.ndesc) for reader in readers])
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


def compute_prefitting(readers, shift=None, scale=None, ridge_alpha=1e-8, symm_sections=None):
    _ensure_readers(readers)
    if shift is None or scale is None:
        all_mean, all_std = compute_data_stat(readers, symm_sections=symm_sections)
        if shift is None:
            shift = all_mean
        if scale is None:
            scale = all_std
    all_sdm = np.concatenate([((reader.data_dm - shift) / scale).sum(1) for reader in readers])
    all_natm = np.concatenate([[float(reader.data_dm.shape[1])] * reader.data_dm.shape[0] for reader in readers])
    if symm_sections is not None:  # in this case ridge alpha cannot be 0
        assert sum(symm_sections) == all_sdm.shape[-1]
        sdm_shells = np.split(all_sdm, np.cumsum(symm_sections)[:-1], axis=-1)
        all_sdm = np.stack([d.sum(-1) for d in sdm_shells], axis=-1)
    # build feature matrix
    xmat = np.concatenate([all_sdm, all_natm.reshape(-1, 1)], -1)
    yvec = np.concatenate([reader.data_ec for reader in readers])
    eye = np.identity(xmat.shape[1])
    eye[-1, -1] = 0  # do not punish the bias term
    # solve ridge reg
    coef = np.linalg.solve(xmat.T @ xmat + ridge_alpha * eye, xmat.T @ yvec).reshape(-1)
    weight, bias = coef[:-1], coef[-1]
    if symm_sections is not None:
        weight = np.concatenate([w.repeat(s) for w, s in zip(weight, symm_sections)], axis=-1)
    return weight, bias


def collect_elems(readers, elem_list=None):
    _ensure_readers(readers)
    if elem_list is None:
        elem_list = np.array(
            sorted(set.union(*[set(reader.atom_info["elems"].flatten()) for reader in readers]))
        )
    for reader in readers:
        reader.collect_elems(elem_list)
    return elem_list


def compute_elem_const(readers, ridge_alpha=0.0):
    _ensure_readers(readers)
    elem_list = collect_elems(readers)
    all_nelem = np.concatenate([reader.atom_info["nelem"] for reader in readers])
    all_ec = np.concatenate([reader.data_ec for reader in readers])
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
        eye = np.identity(grp_nelem.shape[1])
        elem_const = np.linalg.solve(grp_nelem.T @ grp_nelem + ridge_alpha * eye, grp_nelem.T @ grp_ec)
    return elem_list.reshape(-1), elem_const.reshape(-1)


def subtract_elem_const(readers, elem_const):
    _ensure_readers(readers)
    for reader in readers:
        reader.subtract_elem_const(elem_const)


def revert_elem_const(readers):
    _ensure_readers(readers)
    for reader in readers:
        reader.revert_elem_const()
