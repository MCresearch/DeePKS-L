"""
整体覆盖：关键模块导入可用性。

测试列表：
- `test_core_imports_smoke`
"""

import pytest


def test_core_imports_smoke():
    """
    依赖：`deepks` 各核心模块；可选依赖 `pyscf`。
    测试内容：验证常用导入路径可解析；若无 `pyscf` 则跳过对应分支。
    """
    import deepks
    import deepks.__main__
    import deepks.ml.models.corrnet
    import deepks.io.readers
    import deepks.ml.eval.evaluator
    import deepks.orchestration.workflow.workflow

    pytest.importorskip("pyscf")
    import deepks.physics.backends.pyscf.scf

    assert deepks is not None
