"""Stats adapter.

Translate packed stats parameters into report-layer statistics calls.
"""

from contextlib import redirect_stdout


def _build_stats_kwargs(config):
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_input = backend.get("input") if isinstance(backend.get("input"), dict) else {}
    backend_output = backend.get("output") if isinstance(backend.get("output"), dict) else {}
    dump_fields = list(backend_output.get("dump_fields", []))
    loader_cfg = data.get("loader") if isinstance(data.get("loader"), dict) else {}

    has_dump_fields = bool(dump_fields)
    return {
        "group": bool(loader_cfg.get("group", False)),
        "dump_dir": backend_output.get("dump_dir", "."),
        "test_dump": backend_output.get("test_dump", backend_output.get("dump_dir", ".")),
        "with_conv": True,
        "with_e": (not has_dump_fields) or any(field in dump_fields for field in ("e_tot", "e_base")),
        "e_name": "e_tot",
        "with_f": bool(backend_input.get("cal_force")) and (
            (not has_dump_fields) or any(field in dump_fields for field in ("f_tot", "f_base"))
        ),
        "f_name": "f_tot",
        "with_s": bool(backend_input.get("cal_stress")) and (
            (not has_dump_fields) or any(field in dump_fields for field in ("s_tot", "s_base"))
        ),
        "s_name": "s_tot",
        "with_o": bool(backend_input.get("deepks_bandgap")) and (
            (not has_dump_fields) or "bandgap" in dump_fields
        ),
        "o_name": "o_tot",
    }


def _compute_summary(data_tuple):
    if data_tuple is None:
        return None
    conv, e_err, f_err, s_err, o_err = data_tuple
    result = {}
    if conv is not None:
        result["conv_frac"] = float(conv.mean())
        result["conv_count"] = int(conv.sum())
        result["conv_total"] = int(conv.shape[0])
    if e_err is not None:
        result["e_me"] = float(e_err.mean())
        result["e_mae"] = float(abs(e_err).mean())
    if f_err is not None:
        result["f_mae"] = float(abs(f_err).mean())
    if s_err is not None:
        result["s_mae"] = float(abs(s_err).mean())
    if o_err is not None:
        result["o_mae"] = float(abs(o_err).mean())
    return result


def run_stats(config, log_file):
    """Run stats collection/reporting from packed stats config."""
    from deepks.io.reporting import load_stat, load_stat_grouped, print_stats

    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    stats_kwargs = _build_stats_kwargs(config)
    load_func = load_stat_grouped if stats_kwargs["group"] else load_stat

    systems = data.get("systems")
    test_sys = data.get("test")
    tr_data = load_func(
        systems,
        stats_kwargs["dump_dir"],
        stats_kwargs["with_conv"],
        stats_kwargs["with_e"],
        stats_kwargs["e_name"],
        stats_kwargs["with_f"],
        stats_kwargs["f_name"],
        stats_kwargs["with_s"],
        stats_kwargs["s_name"],
        stats_kwargs["with_o"],
        stats_kwargs["o_name"],
    ) if systems is not None else None
    ts_data = load_func(
        test_sys,
        stats_kwargs["test_dump"],
        stats_kwargs["with_conv"],
        stats_kwargs["with_e"],
        stats_kwargs["e_name"],
        stats_kwargs["with_f"],
        stats_kwargs["f_name"],
        stats_kwargs["with_s"],
        stats_kwargs["s_name"],
        stats_kwargs["with_o"],
        stats_kwargs["o_name"],
    ) if test_sys is not None else None

    with open(log_file, "w", 1) as f_stats, redirect_stdout(f_stats):
        print_stats(
            systems=systems,
            test_sys=test_sys,
            dump_dir=stats_kwargs["dump_dir"],
            test_dump=stats_kwargs["test_dump"],
            group=stats_kwargs["group"],
            with_conv=stats_kwargs["with_conv"],
            with_e=stats_kwargs["with_e"],
            e_name=stats_kwargs["e_name"],
            with_f=stats_kwargs["with_f"],
            f_name=stats_kwargs["f_name"],
            with_s=stats_kwargs["with_s"],
            s_name=stats_kwargs["s_name"],
            with_o=stats_kwargs["with_o"],
            o_name=stats_kwargs["o_name"],
        )

    return {
        "stats_log": log_file,
        "backend": config.get("physics", {}).get("backend", {}).get("name", "abacus"),
        "summary": {
            "training": _compute_summary(tr_data),
            "testing": _compute_summary(ts_data),
        },
    }
