import os
import pytest
import contextlib
from deepks.utils import load_yaml
from deepks.pipelines.scf.stats import print_stats as stats_main

current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(current_dir, "stats_input.yaml")
log_dir = os.path.join(current_dir, "log.stats")
err_dir = os.path.join(current_dir, "err")

def run_stats():
    argdict = load_yaml(yaml_path)
    with open(log_dir, "w") as f_out, open(err_dir, "w") as f_err:
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            stats_main(**argdict)

def test_result():
    run_stats()
    with open(log_dir, "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == "Training:"
    assert lines[1].strip() == "Convergence:"
    assert lines[2].split()[-1] == "1.00000"
    assert lines[3].strip() == "Energy:"
    assert lines[4].split()[-1] == "0.0008121852979900979"
    assert lines[5].split()[-1] == "0.0008121852979900979"
    assert lines[6].split()[-1] == "0.0003160297875860844"