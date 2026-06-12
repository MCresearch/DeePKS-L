"""I/O and path helper utilities shared across workflows and orchestration."""

import os
import shutil
import warnings
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


def dump_yaml_str(data):
    """Serialize *data* to a YAML string (ruamel.yaml safe mode)."""
    import io
    buf = io.StringIO()
    yaml = YAML(typ='safe', pure=True)
    yaml.dump(data, buf)
    return buf.getvalue()


def check_share_folder(data, name, share_folder="share"):
    """Save *data* to ``share_folder/name`` (or verify it exists) and return ``name``.

    Used by both the iterate workflow (to materialize shared inputs once and
    reference them by relative name in each subtask) and by the abacus backend
    helpers (to resolve orb/pp/proj files against the shared layout). Lives in
    io so neither workflow nor physics need to import upward to share files.
    """

    if not data:
        return None

    dst_name = os.path.join(share_folder, name)
    if data is True:
        if not os.path.exists(dst_name):
            raise FileNotFoundError(f"No required file: {dst_name}")
        return name
    if isinstance(data, str) and os.path.exists(data):
        copy_file(data, dst_name)
        return name
    if isinstance(data, dict):
        save_yaml(data, dst_name)
        return name
    raise ValueError(f"Invalid argument: {data}")


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


# ---------------------------------------------------------------------------
# Array shape coercion helpers
# ---------------------------------------------------------------------------

def coerce_box(arr, nframes, fname="box.npy"):
    """Ensure box array has shape (nframes, 9); accept (nframes, 3, 3)."""
    if arr.shape == (nframes, 3, 3):
        warnings.warn(f"{fname}: got shape {arr.shape}, reshaping to ({nframes}, 9).")
        return arr.reshape(nframes, 9)
    if arr.shape != (nframes, 9):
        raise ValueError(f"{fname}: expected shape ({nframes}, 9), got {arr.shape}.")
    return arr


def coerce_energy(arr, nframes, fname="energy.npy"):
    """Ensure energy array has shape (nframes, 1); accept (nframes,)."""
    if arr.shape == (nframes,):
        warnings.warn(f"{fname}: got shape {arr.shape}, reshaping to ({nframes}, 1).")
        return arr.reshape(nframes, 1)
    if arr.shape != (nframes, 1):
        raise ValueError(f"{fname}: expected shape ({nframes}, 1), got {arr.shape}.")
    return arr


def coerce_stress(arr, nframes, fname="stress.npy"):
    """Ensure stress array has shape (nframes, 6) upper-triangle (xx,xy,xz,yy,yz,zz).

    Accepted input shapes:
      (nframes, 6)   -- already upper-triangle, returned as-is.
      (nframes, 3,3) -- full matrix, reshaped then upper-triangle sliced.
      (nframes, 9)   -- full flat, upper-triangle sliced.
    """
    if arr.shape == (nframes, 6):
        return arr
    if arr.shape == (nframes, 3, 3):
        warnings.warn(
            f"{fname}: got shape {arr.shape}, reshaping to ({nframes}, 9) "
            f"then taking upper-triangle to ({nframes}, 6)."
        )
        arr = arr.reshape(nframes, 9)
    if arr.shape == (nframes, 9):
        warnings.warn(f"{fname}: got shape {arr.shape}, taking upper-triangle to ({nframes}, 6).")
        return arr[:, [0, 1, 2, 4, 5, 8]]
    raise ValueError(
        f"{fname}: expected ({nframes},6), ({nframes},9), or ({nframes},3,3), got {arr.shape}."
    )
