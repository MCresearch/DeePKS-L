"""
整体覆盖：`deepks test` 命令入口可用性。

测试列表：
- `test_test_cli_help_exit_zero`
"""

import pytest

from deepks.main import test_cli as dks_test_cli


def test_test_cli_help_exit_zero():
    """依赖：`deepks.main.test_cli`。测试内容：`test -h` 返回码为 0。"""
    with pytest.raises(SystemExit) as ex:
        dks_test_cli(["-h"])
    assert ex.value.code == 0
