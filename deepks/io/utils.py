"""I/O and path helper utilities shared across workflows and orchestration."""

import os
import shutil
from collections.abc import Mapping
from glob import glob
from itertools import chain
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


def check_list(arg, nullable=True):
    if arg is None:
        if nullable:
            return []
        raise TypeError("arg cannot be None")
    if not isinstance(arg, (list, tuple, np.ndarray)):
        return [arg]
    return arg


def check_array(arr, nullable=True):
    if arr is None:
        if nullable:
            return arr
        raise TypeError("arg cannot be None")
    if isinstance(arr, str):
        return load_array(arr)
    return np.array(arr)


def flat_file_list(file_list, filter_func=lambda p: True, sort=True):
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
    if not isinstance(o, Mapping):
        return u
    kvlst = chain(u.items() if isinstance(u, Mapping) else u, f.items())
    for k, v in kvlst:
        if isinstance(v, Mapping):
            o[k] = deep_update(o.get(k, {}), v)
        else:
            o[k] = v
    return o


def load_yaml(file_path):
    with open(file_path, 'r') as fp:
        yaml = YAML(typ='safe', pure=True)
        res = yaml.load(fp)
    return res


def save_yaml(data, file_path):
    dirname = os.path.dirname(file_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(file_path, 'w') as fp:
        yaml = YAML(typ='safe', pure=True)
        yaml.dump(data, fp)


def load_array(file):
    ext = os.path.splitext(file)[-1]
    if 'npy' in ext:
        return np.load(file)
    if 'npz' in ext:
        raise NotImplementedError
    try:
        arr = np.loadtxt(file)
    except ValueError:
        arr = np.loadtxt(file, dtype=str)
    return arr


def load_elem_table(filename):
    elem_list, elem_const = np.loadtxt(filename).T
    elem_list = elem_list.round().astype(int)
    return elem_list, elem_const


def save_elem_table(filename, elem_table):
    np.savetxt(filename, np.stack(elem_table).T, fmt=['%i', '%.16f'])


def get_abs_path(p):
    if p is None:
        return None
    return Path(p).absolute()


def get_sys_name(p):
    if p.endswith(os.path.sep):
        return p.rstrip(os.path.sep)
    if p.endswith('.xyz'):
        return p[:-4]
    return p


def get_with_prefix(p, base=None, prefer=None, nullable=False):
    if not base:
        base = './'
    if os.path.isdir(base):
        pattern = os.path.join(base, p)
    else:
        pattern = f"{base.rstrip('.')}.{p}"
    matches = glob(pattern + '*')
    if len(matches) == 1:
        return matches[0]
    prefer = check_list(prefer)
    for suffix in prefer:
        if pattern + suffix in matches:
            return pattern + suffix
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
