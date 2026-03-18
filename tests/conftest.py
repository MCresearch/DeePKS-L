"""
全局测试配置：
- 固定随机种子，降低非确定性波动；
- 本地默认开启 `pyabacus` 相关测试，CI 默认关闭（可通过环境变量覆盖）；
- 作为后续共享 fixture 的统一入口。
"""

import importlib.util
import os
import random

import numpy as np
import pytest


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
