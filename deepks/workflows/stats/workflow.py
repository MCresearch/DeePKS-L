"""Stats workflow - main orchestration.

Implements the DeePKS stats workflow as a 3-stage pipeline:
  Stage 1 (collect): Load SCF result data from dump_dir / system paths.
  Stage 2 (compute): Compute summary statistics from the loaded data.
  Stage 3 (report):  Write the stats log.
"""

from contextlib import redirect_stdout
from typing import Any, Dict, Optional


_STATS_COMPAT_KEYS = {
    'systems',
    'test_sys',
    'dump_dir',
    'test_dump',
    'group',
    'with_conv',
    'with_e',
    'e_name',
    'with_f',
    'f_name',
    'with_s',
    's_name',
    'with_o',
    'o_name',
}


def _build_stats_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stats-relevant keys from config, preserving defaults."""
    return {key: config[key] for key in _STATS_COMPAT_KEYS if key in config}


# ---------------------------------------------------------------------------
# Stage 1: collect
# ---------------------------------------------------------------------------

def _collect(config: Dict[str, Any]):
    """Stage 1: load stat data for training and test sets.

    Returns a tuple (tr_data, ts_data, group) where each *_data is either
    None (no systems given) or a 5-tuple (conv, e_err, f_err, s_err, o_err)
    as returned by load_stat / load_stat_grouped.
    """
    from deepks.physics.backends.stats import load_stat, load_stat_grouped

    group = config.get('group', False)
    load_func = load_stat_grouped if group else load_stat

    dump_dir = config.get('dump_dir', '.')
    test_dump = config.get('test_dump', dump_dir)

    # shared keyword arguments forwarded to load_func
    load_kwargs = {
        'with_conv': config.get('with_conv', True),
        'with_e':    config.get('with_e', True),
        'e_name':    config.get('e_name', 'e_tot'),
        'with_f':    config.get('with_f', True),
        'f_name':    config.get('f_name', 'f_tot'),
        'with_s':    config.get('with_s', True),
        's_name':    config.get('s_name', 's_tot'),
        'with_o':    config.get('with_o', True),
        'o_name':    config.get('o_name', 'o_tot'),
    }

    systems = config.get('systems')
    test_sys = config.get('test_sys')

    tr_data = load_func(systems, dump_dir, **load_kwargs) if systems is not None else None
    ts_data = load_func(test_sys, test_dump, **load_kwargs) if test_sys is not None else None

    return tr_data, ts_data, load_kwargs


# ---------------------------------------------------------------------------
# Stage 2: compute
# ---------------------------------------------------------------------------

def _compute(tr_data, ts_data, load_kwargs):
    """Stage 2: compute summary statistics from loaded data.

    Returns a dict with keys 'training' and 'testing', each mapping to a
    sub-dict of computed statistics (or None when that split is absent).
    """
    def _stats_for(data):
        if data is None:
            return None
        import numpy as np
        conv, e_err, f_err, s_err, o_err = data
        result = {}
        if conv is not None:
            result['conv_frac'] = float(conv.mean())
            result['conv_count'] = int(conv.sum())
            result['conv_total'] = int(conv.shape[0])
        if e_err is not None:
            result['e_me']  = float(e_err.mean())
            result['e_mae'] = float(abs(e_err).mean())
        if f_err is not None:
            import numpy as np
            result['f_mae'] = float(abs(f_err).mean())
        if s_err is not None:
            result['s_mae'] = float(abs(s_err).mean())
        if o_err is not None:
            result['o_mae'] = float(abs(o_err).mean())
        return result

    return {
        'training': _stats_for(tr_data),
        'testing':  _stats_for(ts_data),
    }


# ---------------------------------------------------------------------------
# Stage 3: report
# ---------------------------------------------------------------------------

def _report(config: Dict[str, Any], stats_kwargs: Dict[str, Any], log_file: str):
    """Stage 3: write the stats log via print_stats."""
    from deepks.physics.backends.stats import print_stats

    with open(log_file, 'w', 1) as f_stats, redirect_stdout(f_stats):
        print_stats(**stats_kwargs)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_stats_workflow(config):
    """Run SCF statistics workflow.

    Internally structured as three stages:
      1. Collect  - load stat arrays from dump_dir / system paths
      2. Compute  - compute summary metrics from loaded arrays
      3. Report   - write the stats log file

    Args:
        config: Configuration dictionary containing stats runtime keys:
            systems, dump_dir, test_sys, test_dump, group,
            with_conv, with_e, e_name, with_f, f_name,
            with_s, s_name, with_o, o_name, stats_log / log_file.

    Returns:
        dict: Metadata including stats_log path and computed summary.
    """
    log_file = config.get('stats_log', config.get('log_file', 'log.stats'))
    stats_kwargs = _build_stats_kwargs(config)

    # Stage 1: collect
    tr_data, ts_data, load_kwargs = _collect(config)

    # Stage 2: compute
    summary = _compute(tr_data, ts_data, load_kwargs)

    # Stage 3: report
    _report(config, stats_kwargs, log_file)

    return {
        'stats_log': log_file,
        'backend': config.get('scf_soft', 'abacus'),
        'summary': summary,
    }
