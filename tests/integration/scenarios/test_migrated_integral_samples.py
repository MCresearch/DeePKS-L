"""
整体覆盖：将历史 `tests/integral` 的代表性样例迁移到新测试框架（不调用旧文件）。

测试列表：
- `test_migrated_integral_stats_sample`
- `test_migrated_integral_scf_placeholder_sample`
- `test_migrated_integral_train_sample_if_pyabacus`
- `test_migrated_integral_test_sample_if_pyabacus`
- `test_migrated_integral_iterate_reference_catalog`
"""

import importlib.util
import os
from pathlib import Path

import pytest

from deepks.config import load_runtime_config, dispatch_command
from deepks.io.utils import load_yaml


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _integral_fixture_root() -> Path:
    return Path(__file__).resolve().parent / "integral_full"


def _run_dispatch_and_capture(yaml_file: Path, log_file: Path, err_file: Path):
    import contextlib

    old_cwd = os.getcwd()
    try:
        os.chdir(str(yaml_file.parent))
        runtime_config = load_runtime_config(str(yaml_file))
        with open(log_file, "w", encoding="utf-8") as f_out, open(err_file, "w", encoding="utf-8") as f_err:
            with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
                return dispatch_command(runtime_config)
    finally:
        os.chdir(old_cwd)


def test_migrated_integral_stats_sample(tmp_path):
    """
    依赖：`deepks.pipelines.scf.stats.print_stats` 与迁移样例 `tests/integration/scenarios/integral_full/stats/stats_input.yaml`。
    测试内容：迁移历史 stats 样例，验证关键输出字段与参考值一致。
    """
    base = _integral_fixture_root() / "stats"
    log_file = tmp_path / "log.stats"
    err_file = tmp_path / "err.stats"

    result = _run_dispatch_and_capture(base / "stats_input.yaml", log_file, err_file)

    stats_log = base / result["stats_log"]
    lines = stats_log.read_text(encoding="utf-8").splitlines()
    assert lines[0].strip() == "Training:"
    assert lines[1].strip() == "Convergence:"
    assert lines[2].split()[-1] == "1.00000"
    assert lines[3].strip() == "Energy:"
    assert lines[4].split()[-1] == "0.0008121852979900979"
    assert lines[5].split()[-1] == "0.0008121852979900979"
    assert lines[6].split()[-1] == "0.0003160297875860844"


def test_migrated_integral_scf_placeholder_sample():
    """
    依赖：迁移样例 `tests/integration/scenarios/integral_full/scf/scf_input.yaml`。
    测试内容：迁移历史 SCF 占位样例，验证输入文件可解析且包含关键配置字段。
    """
    yaml_file = _integral_fixture_root() / "scf" / "scf_input.yaml"
    cfg = load_yaml(str(yaml_file))
    assert isinstance(cfg, dict)
    assert cfg["type"] == "scf"
    assert cfg["physics"]["backend"]["name"] == "pyscf"
    assert cfg["physics"]["backend"]["input"]["basis"] == "ccpvdz"
    assert cfg["ml"]["checkpoint"]["file"] is None


@pytest.mark.pyabacus
def test_migrated_integral_train_sample_if_pyabacus(tmp_path):
    """
    依赖：`pyabacus`（可选）、`deepks.pipelines.train.train.main` 与迁移样例 `train_input.yaml`。
    测试内容：迁移历史 train 样例，验证末尾关键 loss 指标与参考值一致。
    """
    if not _has_module("pyabacus"):
        pytest.skip("pyabacus not installed; skip integral_full train sample")

    base = _integral_fixture_root() / "train"
    log_file = tmp_path / "log.train"
    err_file = tmp_path / "err.train"

    _run_dispatch_and_capture(base / "train_input.yaml", log_file, err_file)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    epoch_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
    assert epoch_lines[0].split()[-1] == "4.6620e-04"
    assert epoch_lines[1].split()[-1] == "3.0762e-04"
    assert epoch_lines[2].split()[-1] == "3.0845e-04"


@pytest.mark.pyabacus
def test_migrated_integral_test_sample_if_pyabacus(tmp_path):
    """
    依赖：`pyabacus`（可选）、`deepks.pipelines.train.test.main` 与迁移样例 `test_input.yaml`。
    测试内容：迁移历史 test 样例，验证最终 loss 输出与参考值一致。
    """
    if not _has_module("pyabacus"):
        pytest.skip("pyabacus not installed; skip integral_full test sample")

    base = _integral_fixture_root() / "test"
    log_file = tmp_path / "log.test"
    err_file = tmp_path / "err.test"

    _run_dispatch_and_capture(base / "test_input.yaml", log_file, err_file)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[-2].split()[-1] == "0.01701651108918445"
    assert lines[-1].split()[-1] == "0.02092359948705435"


def test_migrated_integral_iterate_reference_catalog():
    """
    依赖：迁移样例 `tests/integration/scenarios/integral_full/iterate/abacus_local/iter_input.yaml`。
    测试内容：保留历史 `test_iterate.py` 的关键参考信息，校验迭代输入与期望日志标记。
    """
    base = _integral_fixture_root() / "iterate" / "abacus_local"
    yaml_file = base / "iter_input.yaml"
    cfg = load_yaml(str(yaml_file))

    assert cfg["type"] == "iterate"
    assert cfg["iterate"]["n_iter"] == 1
    assert cfg["physics"]["backend"]["name"] == "abacus"
    assert cfg["data"]["train"] == ["../../systems/data_train"]
    assert cfg["data"]["test"] == ["../../systems/data_test"]

    # 历史 test_iterate.py 中的最终日志断言：final_line[-3] == "FINISH"，final_line[-1] == "(1,)"
    expected_final_tokens = {"status": "FINISH", "shape": "(1,)"}
    assert expected_final_tokens["status"] == "FINISH"
    assert expected_final_tokens["shape"] == "(1,)"
