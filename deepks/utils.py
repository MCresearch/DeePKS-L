import os
import shutil
from glob import glob
from pathlib import Path
from ruamel.yaml import YAML
import numpy as np
import torch
from collections.abc import Mapping
from itertools import chain
from scipy.sparse import csr_matrix
from deepks.default import DEFAULT_BASIS, DEFAULT_SYMB

def load_basis(basis):
    if basis is None:
        return DEFAULT_BASIS
    elif isinstance(basis, np.ndarray) and basis.ndim == 2:
        return [[ll, *basis.tolist()] for ll in range(3)]
    elif not isinstance(basis, str):
        return basis
    elif basis.endswith(".npy"):
        table = np.load(basis)
        return [[ll, *table.tolist()] for ll in range(3)]
    elif basis.endswith(".npz"):
        all_tables = np.load(basis)
        return [[int(name.split("_L")[-1]) if "_L" in name else ii, *table.tolist()] 
                for ii, (name, table) in enumerate(all_tables.items())]
    else:
        from pyscf import gto
        symb = DEFAULT_SYMB
        if "@" in basis:
            basis, symb = basis.split("@")
        return gto.basis.load(basis, symb=symb)


def save_basis(file, basis):
    """Save the basis to npz file from internal format of pyscf"""
    tables = {f"arr_{i}_L{l}":np.array(b) for i, (l,*b) in enumerate(basis)}
    np.savez(file, **tables)


def get_shell_sec(basis):
    if not isinstance(basis, (list, tuple)):
        basis = load_basis(basis)
    shell_sec = []
    for l, c0, *cr in basis:
        nb = c0 if isinstance(c0, int) else (len(c0)-1)
        shell_sec.extend([2*l+1] * nb)
    return shell_sec
    

# below are argument chekcing utils

def check_list(arg, nullable=True):
    # make sure the argument is a list
    if arg is None:
        if nullable:
            return []
        else:
            raise TypeError("arg cannot be None")
    if not isinstance(arg, (list, tuple, np.ndarray)):
        return [arg]
    return arg


def check_array(arr, nullable=True):
    if arr is None:
        if nullable:
            return arr
        else:
            raise TypeError("arg cannot be None")
    if isinstance(arr, str):
        return load_array(arr)
    else:
        return np.array(arr)


def flat_file_list(file_list, filter_func=lambda p: True, sort=True):
    # make sure file list contains desired files
    # flat all wildcards and files contains other files (once)
    # if no satisfied files, return empty list
    file_list = check_list(file_list)
    if sort:
        file_list = sorted(sum([glob(p) for p in file_list], []))
    else:
        file_list = sum([glob(p) for p in file_list], [])
    new_list = []
    for p in file_list:
        if filter_func(p):
            new_list.append(p)
        else:
            with open(p) as f:
                sub_list = f.read().splitlines()
                if sort:
                    sub_list = sorted(sum([glob(p) for p in sub_list], []))
                else: 
                    sub_list = sum([glob(p) for p in sub_list], [])
                new_list.extend(sub_list)
    return new_list


def load_dirs(path_list):
    return flat_file_list(path_list, os.path.isdir)

def load_xyz_files(file_list):
    return flat_file_list(file_list, is_xyz)

def load_sys_paths(sys_list):
    return flat_file_list(sys_list, lambda p: os.path.isdir(p) or is_xyz(p))

def is_xyz(p):
    return os.path.splitext(p)[1] == '.xyz'


def deep_update(o, u=(), **f):
    """Recursively update a dict.

    Subdict's won't be overwritten but also updated.
    """
    if not isinstance(o, Mapping):
        return u
    kvlst = chain(u.items() if isinstance(u, Mapping) else u, 
                  f.items())
    for k, v in kvlst:
        if isinstance(v, Mapping):
            o[k] = deep_update(o.get(k, {}), v)
        else:
            o[k] = v
    return o


# below are file loading utils

def load_yaml(file_path):
    with open(file_path, 'r') as fp:
        yaml = YAML(typ='safe', pure=True)
        res = yaml.load(fp)
    return res


def save_yaml(data, file_path):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(file_path, 'w') as fp:
        yaml = YAML(typ='safe', pure=True)
        yaml.dump(data, fp)


def load_array(file):
    ext = os.path.splitext(file)[-1]
    if "npy" in ext:
        return np.load(file)
    elif "npz" in ext:
        raise NotImplementedError
    else:
        try:
            arr = np.loadtxt(file)
        except ValueError:
            arr = np.loadtxt(file, dtype=str)
        return arr


def parse_xyz(filename):
    with open(filename) as fp:
        natom = int(fp.readline())
        comments = fp.readline().strip()
        atom_str = fp.readlines()
    atom_list = [a.split() for a in atom_str]
    elements = [a[0] for a in atom_list]
    coords = np.array([a[1:] for a in atom_list], dtype=float)
    return natom, comments, elements, coords


def load_elem_table(filename):
    elem_list, elem_const = np.loadtxt(filename).T
    elem_list = elem_list.round().astype(int)
    return elem_list, elem_const


def save_elem_table(filename, elem_table):
    np.savetxt(filename, np.stack(elem_table).T, fmt=["%i", "%.16f"])
    

# below are path related utils

def get_abs_path(p):
    if p is None:
        return None
    else:
        return Path(p).absolute()


def get_sys_name(p):
    if p.endswith(os.path.sep):
        return p.rstrip(os.path.sep)
    if p.endswith(".xyz"):
        return p[:-4]
    return p


def get_with_prefix(p, base=None, prefer=None, nullable=False):
    """
    Get file path by searching its prefix.
    If `base` is a directory, equals to get "base/p*".
    Otherwise, equals to get "base.p*".
    Only one result will be return. 
    If more than one match, give the first one with suffix in `prefer`.
    """
    if not base:
        base = "./"
    if os.path.isdir(base):
        pattern = os.path.join(base, p)
    else:
        pattern = f"{base.rstrip('.')}.{p}"
    matches = glob(pattern + "*")
    if len(matches) == 1:
        return matches[0]
    prefer = check_list(prefer)
    for suffix in prefer:
        if pattern+suffix in matches:
            return pattern+suffix
    if nullable:
        return None
    raise FileNotFoundError(f"{pattern}* not exists or has more than one matches")

    
def link_file(src, dst, use_abs=False):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    src_path = os.path.abspath(src) if use_abs else os.path.relpath(src, dst.parent)
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        os.symlink(src_path, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        os.symlink(src_path, dst)


def copy_file(src, dst):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        shutil.copy2(src, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        shutil.copy2(src, dst)


def create_dir(dirname, backup=False):
    dirname = Path(dirname)
    if not dirname.exists():
        os.makedirs(dirname)
    elif backup and dirname != Path('.'):
        os.makedirs(dirname.parent, exist_ok=True)
        counter = 0
        bckname = str(dirname) + f'.bck.{counter:03d}'
        while os.path.exists(bckname):
            counter += 1
            bckname = str(dirname) + f'.bck.{counter:03d}'
        dirname.rename(bckname)
        os.makedirs(dirname)
    else:
        assert dirname.is_dir(), f'{dirname} is not a dir'

def R2iR(R):
    """
    Convert real space coordinates to iR index (natural index).
    Supports int, numpy.ndarray, or torch.Tensor.
    """
    if isinstance(R, int):
        if R > 0:
            return 2 * R - 1
        else:
            return -2 * R
    elif isinstance(R, np.ndarray):
        out = np.where(R > 0, 2 * R - 1, -2 * R)
        return out
    elif torch.is_tensor(R):
        out = torch.where(R > 0, 2 * R - 1, -2 * R)
        return out
    else:
        raise TypeError("R should be int, numpy.ndarray, or torch.Tensor")

def iR2R(iR):
    """
    Convert iR index (natural index) to real space coordinates.
    Supports int, numpy.ndarray, or torch.Tensor.
    """
    if isinstance(iR, int):
        assert iR >= 0, "iR should be a non-negative integer"
        if iR % 2 == 0:
            return - iR // 2
        else:
            return (iR + 1) // 2
    elif isinstance(iR, np.ndarray):
        assert np.all(iR >= 0), "iR should be non-negative"
        out = np.where(iR % 2 == 0, -iR // 2, (iR + 1) // 2)
        return out
    elif torch.is_tensor(iR):
        assert torch.all(iR >= 0), "iR should be non-negative"
        out = torch.where(iR % 2 == 0, -iR // 2, (iR + 1) // 2)
        return out
    else:
        raise TypeError("iR should be int, numpy.ndarray, or torch.Tensor")

def read_csr(file, dtype=torch.float64):
    '''
    Read csr format file generated by ABACUS and return a torch sparse tensor
    The structure of the file is:
        The first two lines contains the dimension and number of matrices
        The following lines are grouped by 4/1 line(s), each group contains:
        For nnz != 0: Rx Ry Rz nnz (first line), data (second line), indices (third line), indptr (fourth line)
        For nnz == 0: Rx Ry Rz nnz (only one line)
    '''
    all_indices = []
    all_values = []
    max_iR = 0
    dim = 0
    with open(file, 'r') as f:
        dim = int(f.readline().split()[-1])
        num = int(f.readline().split()[-1])
        for ir in range(num):
            nnz = 0
            while nnz == 0:
                r = f.readline().split()
                if len(r) == 0:
                    break
                # get index of Bravais lattice and convert to iR index
                Rx, Ry, Rz, nnz = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                iRx = R2iR(Rx)
                iRy = R2iR(Ry)
                iRz = R2iR(Rz)
                if max_iR < max(iRx, iRy, iRz):
                    max_iR = max(iRx, iRy, iRz)
            if nnz == 0:
                break
            # following three lines are in csr format with nnz (number of non-zero) elements
            data = [float(x) for x in f.readline().split()]
            indices = [int(x) for x in f.readline().split()]
            indptr = [int(x) for x in f.readline().split()]
            # build csr matrix
            matrix = csr_matrix((data, indices, indptr), shape=(dim, dim))
            # convert to coo format to get row and col
            matrix_coo = matrix.tocoo()
            rows = matrix_coo.row
            cols = matrix_coo.col
            data = matrix_coo.data
            # combine with iR index
            indices = np.vstack([np.full(nnz, iRx), np.full(nnz, iRy), np.full(nnz, iRz), rows, cols])
            all_indices.append(indices)
            all_values.append(data)
    # convert to sparse tensor
    final_indices = np.hstack(all_indices)
    final_values = np.concatenate(all_values)
    sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.tensor(final_indices, dtype=torch.long),
        values=torch.tensor(final_values, dtype=dtype),
        size=(max_iR + 1, max_iR + 1, max_iR + 1, dim, dim)
    )
    return sparse_tensor.coalesce()
