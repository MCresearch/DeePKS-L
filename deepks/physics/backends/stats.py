"""Backend-agnostic SCF statistics utilities.

Functions for loading, aggregating, and printing SCF result statistics.
These are independent of the specific SCF backend (PySCF, ABACUS, etc.).
"""

import os
import sys
import glob
import numpy as np
import shutil

from deepks.io.utils import check_list, check_array
from deepks.io.utils import load_array, load_yaml
from deepks.io.utils import get_sys_name, get_with_prefix
from deepks.io.utils import coerce_energy, coerce_stress


def concat_data(systems=None, sys_dir=".", dump_dir=".", pattern="*"):
    if systems is None:
        systems = sorted(filter(os.path.isdir,
            map(os.path.abspath, glob.glob(f"{sys_dir}/{pattern}"))))
    npy_names = list(map(os.path.basename, glob.glob(f"{systems[0]}/*.npy")))
    os.makedirs(dump_dir, exist_ok=True)
    for nm in npy_names:
        tmp_array = np.concatenate([np.load(f"{sys}/{nm}") for sys in systems])
        np.save(f"{dump_dir}/{nm}", tmp_array)
    if os.path.exists(f'{systems[0]}/system.raw'):
        shutil.copy(f'{systems[0]}/system.raw', dump_dir)


def print_stats(systems=None, test_sys=None,
               dump_dir=None, test_dump=None, group=False,
               with_conv=True, with_e=True, e_name="e_tot",
               with_f=True, f_name="f_tot",
               with_s=True, s_name="s_tot",
               with_o=True, o_name="o_tot"):
    load_func = load_stat if not group else load_stat_grouped
    if dump_dir is None:
        dump_dir = "."
    if test_dump is None:
        test_dump = dump_dir
    shift = None
    if systems is not None:
        tr_c, tr_e, tr_f, tr_s, tr_o = load_func(systems, dump_dir, with_conv,
                                     with_e, e_name, with_f, f_name, with_s, s_name, with_o, o_name)
        print("Training:")
        if tr_c is not None:
            print_stats_conv(tr_c, indent=2)
        if tr_e is not None:
            shift = tr_e.mean()
            print_stats_e(tr_e, shift=shift, indent=2)
        if tr_f is not None:
            print_stats_f(tr_f, indent=2)
        if tr_s is not None:
            print_stats_s(tr_s, indent=2)
        if tr_o is not None:
            print_stats_o(tr_o, indent=2)
    if test_sys is not None:
        ts_c, ts_e, ts_f, ts_s, ts_o = load_func(test_sys, test_dump, with_conv,
                                     with_e, e_name, with_f, f_name, with_s, s_name, with_o, o_name)
        print("Testing:")
        if ts_c is not None:
            print_stats_conv(ts_c, indent=2)
        if ts_e is not None:
            print_stats_e(ts_e, shift=shift, indent=2)
        if ts_f is not None:
            print_stats_f(ts_f, indent=2)
        if ts_s is not None:
            print_stats_s(ts_s, indent=2)
        if ts_o is not None:
            print_stats_o(ts_o, indent=2)


def print_stats_conv(conv, indent=0):
    nsys = conv.shape[0]
    ind = " " * indent
    print(ind + 'Convergence:')
    print(ind + f'  {np.sum(conv)} / {nsys} = \t {np.mean(conv):.5f}')


def print_stats_e(e_err, shift=None, indent=0):
    ind = " " * indent
    print(ind + "Energy:")
    print(ind + f'  ME: \t {e_err.mean()}')
    print(ind + f'  MAE: \t {np.abs(e_err).mean()}')
    if shift:
        print(ind + f'  MARE: \t {np.abs(e_err - shift).mean()}')


def print_stats_f(f_err, indent=0):
    ind = " " * indent
    print(ind + "Force:")
    print(ind + f'  MAE: \t {np.abs(f_err).mean()}')


def print_stats_s(s_err, indent=0):
    ind = " " * indent
    print(ind + "Stress:")
    print(ind + f'  MAE: \t {np.abs(s_err).mean()}')


def print_stats_o(o_err, indent=0):
    ind = " " * indent
    print(ind + "Band gap:")
    print(ind + f'  MAE: \t {np.abs(o_err).mean()}')


def load_stat(systems, dump_dir,
              with_conv=True, with_e=True, e_name="e_tot",
              with_f=True, f_name="f_tot",
              with_s=True, s_name="s_tot",
              with_o=True, o_name="o_tot"):
    systems = check_list(systems)
    c_res = []
    e_err = []
    f_err = []
    s_err = []
    o_err = []
    for fl in systems:
        lbase = get_sys_name(fl)
        rbase = os.path.join(dump_dir, os.path.basename(lbase))
        if with_conv:
            try:
                c_res.append(load_array(get_with_prefix("conv", rbase, ".npy")))
            except FileNotFoundError as e:
                print("Warning! conv.npy not found:", e, file=sys.stderr)
        if with_e:
            try:
                re_raw = load_array(get_with_prefix(e_name, rbase, ".npy"))
                le_raw = load_array(get_with_prefix("energy", lbase, ".npy"))
                nframes = re_raw.shape[0]
                re = coerce_energy(re_raw, nframes, e_name + ".npy")
                le = coerce_energy(le_raw, nframes, "energy.npy")
                e_err.append(le - re)
            except FileNotFoundError as e:
                print("Warning! energy file not found:", e, file=sys.stderr)
        if with_f:
            try:
                rf = load_array(get_with_prefix(f_name, rbase, ".npy"))
                lf = load_array(get_with_prefix("force", lbase, ".npy")).reshape(rf.shape)
                f_err.append(np.abs(lf - rf).mean((-1, -2)))
            except FileNotFoundError as e:
                print("Warning! force file not found:", e, file=sys.stderr)
        if with_s:
            try:
                rs = load_array(get_with_prefix(s_name, rbase, ".npy"))
                nframes = rs.shape[0]
                ls_raw = load_array(get_with_prefix("stress", lbase, ".npy"))
                ls = coerce_stress(ls_raw, nframes, "stress.npy").reshape(rs.shape)
                s_err.append(np.abs(ls - rs))
            except FileNotFoundError as e:
                print("Warning! stress file not found:", e, file=sys.stderr)
        if with_o:
            try:
                ro = load_array(get_with_prefix(o_name, rbase, ".npy"))
                lo = load_array(get_with_prefix("orbital", lbase, ".npy")).reshape(ro.shape)
                o_err.append(np.abs(lo - ro).mean((-1, -2)))
            except FileNotFoundError as e:
                print("Warning! orbital file not found:", e, file=sys.stderr)
    return (np.concatenate(c_res, 0) if c_res else None,
            np.concatenate(e_err, 0) if e_err else None,
            np.concatenate(f_err, 0) if f_err else None,
            np.concatenate(s_err, 0) if s_err else None,
            np.concatenate(o_err, 0) if o_err else None)


def load_stat_grouped(systems, dump_dir=".",
                      with_conv=True, with_e=True, e_name="e_tot",
                      with_f=True, f_name="f_tot",
                      with_s=True, s_name="s_tot",
                      with_o=True, o_name="o_tot"):
    systems = check_list(systems)
    lbases = [get_sys_name(fl) for fl in systems]
    c_res = e_err = f_err = s_err = o_err = None
    if with_conv:
        c_res = load_array(get_with_prefix("conv", dump_dir, ".npy"))
    if with_e:
        e_res = load_array(get_with_prefix(e_name, dump_dir, ".npy"))
        nframes = e_res.shape[0]
        e_lbl_raw = np.concatenate([
            load_array(get_with_prefix("energy", lb, ".npy")) for lb in lbases
        ], 0)
        e_lbl = coerce_energy(e_lbl_raw, nframes, "energy.npy")
        e_res = coerce_energy(e_res, nframes, e_name + ".npy")
        e_err = e_lbl - e_res
    if with_f:
        f_res = load_array(get_with_prefix(f_name, dump_dir, ".npy"))
        f_lbl = np.concatenate([
            load_array(get_with_prefix("force", lb, ".npy")) for lb in lbases
        ], 0).reshape(f_res.shape)
        f_err = f_lbl - f_res
    if with_s:
        s_res = load_array(get_with_prefix(s_name, dump_dir, ".npy"))
        nframes = s_res.shape[0]
        s_lbl_raw = np.concatenate([
            load_array(get_with_prefix("stress", lb, ".npy")) for lb in lbases
        ], 0)
        s_lbl = coerce_stress(s_lbl_raw, nframes, "stress.npy").reshape(s_res.shape)
        s_err = s_lbl - s_res
    if with_o:
        o_res = load_array(get_with_prefix(o_name, dump_dir, ".npy"))
        o_lbl = np.concatenate([
            load_array(get_with_prefix("orbital", lb, ".npy")) for lb in lbases
        ], 0).reshape(o_res.shape)
        o_err = o_lbl - o_res
    return c_res, e_err, f_err, s_err, o_err


# ---------------------------------------------------------------------------
# Legacy tools, kept for old examples
# ---------------------------------------------------------------------------

def print_stats_per_sys(err, conv=None, train_idx=None, test_idx=None):
    err = np.array(err).reshape(-1)
    nsys = err.shape[0]
    if conv is not None:
        assert len(conv) == nsys
        print(f'converged calculation: {np.sum(conv)} / {nsys} = {np.mean(conv):.3f}')
    print(f'mean error: {err.mean()}')
    print(f'mean absolute error: {np.abs(err).mean()}')
    if train_idx is not None:
        if test_idx is None:
            test_idx = np.setdiff1d(np.arange(nsys), train_idx, assume_unique=True)
        print(f'  training: {np.abs(err[train_idx]).mean()}')
        print(f'  testing: {np.abs(err[test_idx]).mean()}')
        print(f'mean absolute error after shift: {np.abs(err - err[train_idx].mean()).mean()}')
        print(f'  training: {np.abs(err[train_idx] - err[train_idx].mean()).mean()}')
        print(f'  testing: {np.abs(err[test_idx] - err[train_idx].mean()).mean()}')
