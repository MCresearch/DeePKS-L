"""
全局测试配置：
- 固定随机种子，降低非确定性波动；
- 默认关闭 `pyabacus` 相关测试（可通过环境变量显式开启）；
- 作为后续共享 fixture 的统一入口。
"""

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
    """默认跳过 pyabacus 相关测试，除非显式开启。"""
    if os.getenv("ENABLE_PYABACUS_TESTS", "0") == "1":
        return
    skip_marker = pytest.mark.skip(reason="pyabacus tests are disabled by default in this repository")
    for item in items:
        if item.get_closest_marker("pyabacus"):
            item.add_marker(skip_marker)
