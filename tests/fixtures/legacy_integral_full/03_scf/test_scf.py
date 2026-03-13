import os
import pytest
import contextlib
from deepks.utils import load_yaml
# from deepks.scf.run import main as scf_main

current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(current_dir, "scf_input.yaml")
log_dir = os.path.join(current_dir, "log.scf")
err_dir = os.path.join(current_dir, "err")

def run_scf():
    argdict = load_yaml(yaml_path)
    with open(log_dir, "w") as f_out, open(err_dir, "w") as f_err:
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            pass # scf_main(**argdict)

def test_result():
    # run_scf()
    # with open(log_dir, "r") as f:
    #     lines = f.readlines()
    return