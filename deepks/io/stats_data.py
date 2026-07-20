"""Stats dataset loading and concatenation helpers owned by the io layer."""

import glob
import os
import shutil
import sys

import numpy as np

from deepks.io.utils import check_list, coerce_energy, coerce_stress, get_sys_name, get_with_prefix, load_array


def concat_data(systems=None, sys_dir=".", dump_dir=".", pattern="*"):
    if systems is None:
        systems = sorted(filter(os.path.isdir, map(os.path.abspath, glob.glob(f"{sys_dir}/{pattern}"))))
    npy_names = list(map(os.path.basename, glob.glob(f"{systems[0]}/*.npy")))
    os.makedirs(dump_dir, exist_ok=True)
    for nm in npy_names:
        tmp_array = np.concatenate([np.load(f"{sys}/{nm}") for sys in systems])
        np.save(f"{dump_dir}/{nm}", tmp_array)
    if os.path.exists(f"{systems[0]}/system.raw"):
        shutil.copy(f"{systems[0]}/system.raw", dump_dir)


def load_stat(
    systems,
    dump_dir,
    with_conv=True,
    with_e=True,
    e_name="e_tot",
    with_f=True,
    f_name="f_tot",
    with_s=True,
    s_name="s_tot",
    with_o=True,
    o_name="o_tot",
):
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
    return (
        np.concatenate(c_res, 0) if c_res else None,
        np.concatenate(e_err, 0) if e_err else None,
        np.concatenate(f_err, 0) if f_err else None,
        np.concatenate(s_err, 0) if s_err else None,
        np.concatenate(o_err, 0) if o_err else None,
    )


def load_stat_grouped(
    systems,
    dump_dir=".",
    with_conv=True,
    with_e=True,
    e_name="e_tot",
    with_f=True,
    f_name="f_tot",
    with_s=True,
    s_name="s_tot",
    with_o=True,
    o_name="o_tot",
):
    systems = check_list(systems)
    lbases = [get_sys_name(fl) for fl in systems]
    c_res = e_err = f_err = s_err = o_err = None
    if with_conv:
        c_res = load_array(get_with_prefix("conv", dump_dir, ".npy"))
    if with_e:
        e_res = load_array(get_with_prefix(e_name, dump_dir, ".npy"))
        nframes = e_res.shape[0]
        e_lbl_raw = np.concatenate([load_array(get_with_prefix("energy", lb, ".npy")) for lb in lbases], 0)
        e_lbl = coerce_energy(e_lbl_raw, nframes, "energy.npy")
        e_res = coerce_energy(e_res, nframes, e_name + ".npy")
        e_err = e_lbl - e_res
    if with_f:
        f_res = load_array(get_with_prefix(f_name, dump_dir, ".npy"))
        f_lbl = np.concatenate([load_array(get_with_prefix("force", lb, ".npy")) for lb in lbases], 0).reshape(f_res.shape)
        f_err = f_lbl - f_res
    if with_s:
        s_res = load_array(get_with_prefix(s_name, dump_dir, ".npy"))
        nframes = s_res.shape[0]
        s_lbl_raw = np.concatenate([load_array(get_with_prefix("stress", lb, ".npy")) for lb in lbases], 0)
        s_lbl = coerce_stress(s_lbl_raw, nframes, "stress.npy").reshape(s_res.shape)
        s_err = s_lbl - s_res
    if with_o:
        o_res = load_array(get_with_prefix(o_name, dump_dir, ".npy"))
        o_lbl = np.concatenate([load_array(get_with_prefix("orbital", lb, ".npy")) for lb in lbases], 0).reshape(o_res.shape)
        o_err = o_lbl - o_res
    return c_res, e_err, f_err, s_err, o_err


__all__ = ["concat_data", "load_stat", "load_stat_grouped"]
