import os
import pytest
import contextlib
from deepks.io.utils import load_yaml
from deepks.ml.eval.test import main as tst_main

current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(current_dir, "test_input.yaml")
log_dir = os.path.join(current_dir, "log.test")
err_dir = os.path.join(current_dir, "err")

def run_tst():
    argdict = load_yaml(yaml_path)
    with open(log_dir, "w") as f_out, open(err_dir, "w") as f_err:
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            tst_main(**argdict)

def test_result():
    run_tst()
    with open(log_dir, "r") as f:
        lines = f.readlines()
    assert lines[-2].split()[-1] == "0.01701651108918445"
    assert lines[-1].split()[-1] == "0.02092359948705435"