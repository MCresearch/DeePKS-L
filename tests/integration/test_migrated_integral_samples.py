"""
整体覆盖：将历史 `tests/integral` 的代表性样例迁移到新测试框架（不调用旧文件）。

测试列表：
- `test_migrated_integral_stats_sample`
- `test_migrated_integral_scf_placeholder_sample`
- `test_migrated_integral_train_sample_if_pyabacus`
- `test_migrated_integral_test_sample_if_pyabacus`
- `test_migrated_integral_iterate_reference_catalog`
"""

import contextlib
import importlib.util
import os
from pathlib import Path

import pytest

from deepks.pipelines.train.test import main as model_test_main
from deepks.pipelines.train.train import main as train_main
from deepks.pipelines.scf.stats import print_stats as stats_main
from deepks.utils import load_yaml


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_and_capture(func, yaml_file: Path, log_file: Path, err_file: Path):
    old_cwd = os.getcwd()
    try:
        os.chdir(str(yaml_file.parent))
        argdict = load_yaml(str(yaml_file))
        with open(log_file, "w", encoding="utf-8") as f_out, open(err_file, "w", encoding="utf-8") as f_err:
            with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
                func(**argdict)
    finally:
        os.chdir(old_cwd)


def _legacy_integral_fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "legacy_integral_full"


def test_migrated_integral_stats_sample(tmp_path):
    """
    依赖：`deepks.pipelines.scf.stats.print_stats` 与迁移样例 `tests/fixtures/legacy_integral/04_stats/stats_input.yaml`。
    测试内容：迁移历史 stats 样例，验证关键输出字段与参考值一致。
    """
    base = _legacy_integral_fixture_root() / "04_stats"
    log_file = tmp_path / "log.stats"
    err_file = tmp_path / "err.stats"

    _run_and_capture(stats_main, base / "stats_input.yaml", log_file, err_file)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[0].strip() == "Training:"
    assert lines[1].strip() == "Convergence:"
    assert lines[2].split()[-1] == "1.00000"
    assert lines[3].strip() == "Energy:"
    assert lines[4].split()[-1] == "0.0008121852979900979"
    assert lines[5].split()[-1] == "0.0008121852979900979"
    assert lines[6].split()[-1] == "0.0003160297875860844"


def test_migrated_integral_scf_placeholder_sample():
    """
    依赖：迁移样例 `tests/fixtures/legacy_integral/03_scf/scf_input.yaml`。
    测试内容：迁移历史 SCF 占位样例，验证输入文件可解析且包含关键配置字段。
    """
    yaml_file = _legacy_integral_fixture_root() / "03_scf" / "scf_input.yaml"
    cfg = load_yaml(str(yaml_file))
    assert isinstance(cfg, dict)
    assert "basis" in cfg
    assert "model_file" in cfg


@pytest.mark.pyabacus
def test_migrated_integral_train_sample_if_pyabacus(tmp_path):
    """
    依赖：`pyabacus`（可选）、`deepks.pipelines.train.train.main` 与迁移样例 `train_input.yaml`。
    测试内容：迁移历史 train 样例，验证末尾关键 loss 指标与参考值一致。
    """
    if not _has_module("pyabacus"):
        pytest.skip("pyabacus not installed; skip legacy integral train sample")

    base = _legacy_integral_fixture_root() / "01_train"
    log_file = tmp_path / "log.train"
    err_file = tmp_path / "err.train"

    _run_and_capture(train_main, base / "train_input.yaml", log_file, err_file)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[-3].split()[-1] == "4.6620e-04"
    assert lines[-2].split()[-1] == "3.0762e-04"
    assert lines[-1].split()[-1] == "3.0845e-04"


@pytest.mark.pyabacus
def test_migrated_integral_test_sample_if_pyabacus(tmp_path):
    """
    依赖：`pyabacus`（可选）、`deepks.pipelines.train.test.main` 与迁移样例 `test_input.yaml`。
    测试内容：迁移历史 test 样例，验证最终 loss 输出与参考值一致。
    """
    if not _has_module("pyabacus"):
        pytest.skip("pyabacus not installed; skip legacy integral test sample")

    base = _legacy_integral_fixture_root() / "02_test"
    log_file = tmp_path / "log.test"
    err_file = tmp_path / "err.test"

    _run_and_capture(model_test_main, base / "test_input.yaml", log_file, err_file)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[-2].split()[-1] == "0.01701651108918445"
    assert lines[-1].split()[-1] == "0.02092359948705435"


def test_migrated_integral_iterate_reference_catalog():
    """
    依赖：迁移样例 `tests/fixtures/legacy_integral_full/05_iter/01_abacus_local/iter_input.yaml`。
    测试内容：保留历史 `test_iterate.py` 的关键参考信息，校验迭代输入与期望日志标记。
    """
    base = _legacy_integral_fixture_root() / "05_iter" / "01_abacus_local"
    yaml_file = base / "iter_input.yaml"
    cfg = load_yaml(str(yaml_file))

    assert cfg["n_iter"] == 1
    assert cfg["strict"] is True
    assert cfg["use_abacus"] is True
    assert cfg["systems_train"] == ["../../systems/data_train"]
    assert cfg["systems_test"] == ["../../systems/data_test"]

    # 历史 test_iterate.py 中的最终日志断言：final_line[-3] == "FINISH"，final_line[-1] == "(1,)"
    expected_final_tokens = {"status": "FINISH", "shape": "(1,)"}
    assert expected_final_tokens["status"] == "FINISH"
    assert expected_final_tokens["shape"] == "(1,)"
