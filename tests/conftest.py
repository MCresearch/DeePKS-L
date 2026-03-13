"""
全局测试配置：
- 固定随机种子，降低非确定性波动；
- 作为后续共享 fixture 的统一入口。
"""

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
