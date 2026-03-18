import os
import pytest
import contextlib
from deepks.utils import load_yaml
from deepks.pipelines.train.train import main as train_main

current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(current_dir, "train_input.yaml")
log_dir = os.path.join(current_dir, "log.train")
err_dir = os.path.join(current_dir, "err")

def run_train():
    argdict = load_yaml(yaml_path)
    with open(log_dir, "w") as f_out, open(err_dir, "w") as f_err:
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            train_main(**argdict)

def test_result():
    run_train()
    with open(log_dir, "r") as f:
        lines = f.readlines()
    assert lines[-3].split()[-1] == "4.6620e-04"
    assert lines[-2].split()[-1] == "3.0762e-04"
    assert lines[-1].split()[-1] == "3.0845e-04"