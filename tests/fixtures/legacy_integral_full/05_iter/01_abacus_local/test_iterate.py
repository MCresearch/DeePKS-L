import contextlib
import shutil
from pathlib import Path

import deepks.pipelines.iterate.iterate as iterate_mod
import pytest

from deepks.iterate.iterate import main as iter_main
from deepks.orchestration.workflow.task import PythonTask
from deepks.utils import load_yaml


CURRENT_DIR = Path(__file__).resolve().parent
YAML_PATH = CURRENT_DIR / "iter_input.yaml"
LOG_PATH = CURRENT_DIR / "log.iter"
ERR_PATH = CURRENT_DIR / "err"

GENERATED_NAMES = ("share", "iter.init", "iter.00", "RECORD", "log.iter", "err")


def _remove_generated() -> None:
    for name in GENERATED_NAMES:
        target = CURRENT_DIR / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)


def _fake_make_scf_abacus(*args, **kwargs):
    workdir = kwargs.get("workdir", "00.scf")

    def _run_mock_scf():
        Path("data_train").mkdir(exist_ok=True)
        Path("data_test").mkdir(exist_ok=True)

    return PythonTask(_run_mock_scf, workdir=workdir, outlog="log.scf", errlog="err")


def _fake_make_train(*args, **kwargs):
    workdir = kwargs.get("workdir", "01.train")

    def _run_mock_train():
        Path("model.pth").write_text("mock-model\n", encoding="utf-8")

    return PythonTask(_run_mock_train, workdir=workdir, outlog="log.train", errlog="err")


@pytest.fixture(autouse=True)
def _prepare_runtime_tree(monkeypatch):
    _remove_generated()
    monkeypatch.setattr(iterate_mod, "make_scf_abacus", _fake_make_scf_abacus)
    monkeypatch.setattr(iterate_mod, "make_train", _fake_make_train)
    yield
    _remove_generated()


def run_iter():
    argdict = load_yaml(str(YAML_PATH))
    with open(LOG_PATH, "w", encoding="utf-8") as f_out, open(ERR_PATH, "w", encoding="utf-8") as f_err:
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            iter_main(**argdict)


def test_result():
    for name in GENERATED_NAMES:
        assert not (CURRENT_DIR / name).exists()

    run_iter()

    for name in ("share", "iter.init", "iter.00", "RECORD", "log.iter", "err"):
        assert (CURRENT_DIR / name).exists()

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    final_line = lines[-1].split()
    assert final_line[-1] == "(1,)"
    assert final_line[-3] == "FINISH"
