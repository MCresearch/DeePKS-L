import os
import pytest
import contextlib
from deepks.utils import load_yaml
from deepks.iterate.iterate import main as iter_main

current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(current_dir, "iter_input.yaml")
log_dir = os.path.join(current_dir, "log.iter")
err_dir = os.path.join(current_dir, "err")

def run_iter():
    argdict = load_yaml(yaml_path)
    with open(log_dir, "w") as f_out, open(err_dir, "w") as f_err:
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            iter_main(**argdict)

def test_result():
    run_iter()
    with open(log_dir, "r") as f:
        lines = f.readlines()
        final_line = lines[-1].split()
        assert final_line[-1] == "(1,)"
        assert final_line[-3] == "FINISH"
