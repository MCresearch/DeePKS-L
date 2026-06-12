"""Backend-agnostic SCF statistics utilities.

These functions compare dumped results against labels and generate reports.
They belong to the interface/report layer rather than the physics layer.
"""

import numpy as np

from deepks.io.stats_data import concat_data, load_stat, load_stat_grouped

def print_stats(
    systems=None,
    test_sys=None,
    dump_dir=None,
    test_dump=None,
    group=False,
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
    load_func = load_stat if not group else load_stat_grouped
    if dump_dir is None:
        dump_dir = "."
    if test_dump is None:
        test_dump = dump_dir
    shift = None
    if systems is not None:
        tr_c, tr_e, tr_f, tr_s, tr_o = load_func(
            systems, dump_dir, with_conv, with_e, e_name, with_f, f_name, with_s, s_name, with_o, o_name
        )
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
        ts_c, ts_e, ts_f, ts_s, ts_o = load_func(
            test_sys, test_dump, with_conv, with_e, e_name, with_f, f_name, with_s, s_name, with_o, o_name
        )
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
    ind = " " * indent
    print(ind + "Convergence:")
    print(ind + f"  {np.sum(conv)} / {conv.shape[0]} = \t {np.mean(conv):.5f}")


def print_stats_e(e_err, shift=None, indent=0):
    ind = " " * indent
    print(ind + "Energy:")
    print(ind + f"  ME: \t {e_err.mean()}")
    print(ind + f"  MAE: \t {np.abs(e_err).mean()}")
    if shift is not None:
        print(ind + f"  MARE: \t {np.abs(e_err - shift).mean()}")


def print_stats_f(f_err, indent=0):
    ind = " " * indent
    print(ind + "Force:")
    print(ind + f"  MAE: \t {np.abs(f_err).mean()}")


def print_stats_s(s_err, indent=0):
    ind = " " * indent
    print(ind + "Stress:")
    print(ind + f"  MAE: \t {np.abs(s_err).mean()}")


def print_stats_o(o_err, indent=0):
    ind = " " * indent
    print(ind + "Band gap:")
    print(ind + f"  MAE: \t {np.abs(o_err).mean()}")
def print_stats_per_sys(err, conv=None, train_idx=None, test_idx=None):
    err = np.array(err).reshape(-1)
    nsys = err.shape[0]
    if conv is not None:
        assert len(conv) == nsys
        print(f"converged calculation: {np.sum(conv)} / {nsys} = {np.mean(conv):.3f}")
    print(f"mean error: {err.mean()}")
    print(f"mean absolute error: {np.abs(err).mean()}")
    if train_idx is not None:
        if test_idx is None:
            test_idx = np.setdiff1d(np.arange(nsys), train_idx, assume_unique=True)
        print(f"  training: {np.abs(err[train_idx]).mean()}")
        print(f"  testing: {np.abs(err[test_idx]).mean()}")
        print(f"mean absolute error after shift: {np.abs(err - err[train_idx].mean()).mean()}")
        print(f"  training: {np.abs(err[train_idx] - err[train_idx].mean()).mean()}")
        print(f"  testing: {np.abs(err[test_idx] - err[train_idx].mean()).mean()}")
