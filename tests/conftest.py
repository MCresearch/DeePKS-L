"""
全局测试配置：
- 固定随机种子，降低非确定性波动；
- 本地默认开启 `pyabacus` 相关测试，CI 默认关闭（可通过环境变量覆盖）；
- 作为后续共享 fixture 的统一入口。
"""

import importlib.util
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest


def _safe_remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    path.unlink(missing_ok=True)


def _cleanup_legacy_integral_full_artifacts() -> None:
    """删除 legacy integral 样例在运行中产生的状态文件，避免历史状态污染。"""
    root = Path(__file__).resolve().parent / "integration" / "scenarios" / "legacy_integral_full"
    if not root.exists():
        return

    # Common logs/stderr under sample cases.
    cleanup_patterns = {
        "train": ("log.train", "err"),
        "test": ("log.test", "err"),
        "scf": ("log.scf", "err"),
        "stats": ("log.stats", "err"),
    }
    for rel_dir, patterns in cleanup_patterns.items():
        base = root / rel_dir
        if not base.exists():
            continue
        for pattern in patterns:
            for target in base.glob(pattern):
                _safe_remove(target)

    iter_root = root / "iterate" / "abacus_local"
    if iter_root.exists():
        for fname in ("log.iter", "err", "RECORD", "share"):
            _safe_remove(iter_root / fname)

        for target in iter_root.glob("iter.*"):
            _safe_remove(target)

    for cache_dir in root.rglob("__pycache__"):
        _safe_remove(cache_dir)


@pytest.fixture(autouse=True)
def _fixed_seed():
    """所有测试自动启用固定随机种子。"""
    random.seed(20260313)
    np.random.seed(20260313)
    try:
        import torch

        torch.manual_seed(20260313)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _preclean_generated_artifacts(request):
    """仅在可能写入 legacy fixture 目录的测试中执行预清理。"""
    node_path = Path(str(request.fspath)).as_posix()
    needs_cleanup = (
        "tests/integration/scenarios/legacy_integral_full/" in node_path
        or node_path.endswith("tests/integration/scenarios/test_migrated_integral_samples.py")
    )
    if needs_cleanup:
        _cleanup_legacy_integral_full_artifacts()


def pytest_collection_modifyitems(config, items):
    """本地默认运行 pyabacus 测试；CI 默认跳过，可通过环境变量覆盖。"""
    env_override = os.getenv("ENABLE_PYABACUS_TESTS")
    if env_override is not None:
        enabled = env_override == "1"
    else:
        enabled = (os.getenv("CI", "false").lower() != "true") and (importlib.util.find_spec("pyabacus") is not None)
    if enabled:
        return
    skip_marker = pytest.mark.skip(reason="pyabacus tests are disabled by default in this repository")
    for item in items:
        if item.get_closest_marker("pyabacus"):
            item.add_marker(skip_marker)
