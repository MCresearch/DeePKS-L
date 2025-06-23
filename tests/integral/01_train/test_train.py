import pytest
import contextlib
from deepks.utils import load_yaml
from deepks.model.train import main as train_main

def run_train():
    argdict = load_yaml("./train_input.yaml")
    with open("log.train", "w") as f_out, open("err", "w") as f_err:
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            train_main(**argdict)

def test_result():
    run_train()
    with open("log.train", "r") as f:
        lines = f.readlines()
    train_0 = lines[-3].split()[-1]
    train_1 = lines[-2].split()[-1]
    train_2 = lines[-1].split()[-1]
    assert train_0 == "4.6620e-04"
    assert train_1 == "3.0762e-04"
    assert train_2 == "3.0845e-04"