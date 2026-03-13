"""
整体覆盖：`deepks train` 命令入口可用性。

测试列表：
- `test_train_cli_help_exit_zero`
"""

import pytest

from deepks.main import train_cli


def test_train_cli_help_exit_zero():
    """依赖：`deepks.main.train_cli`。测试内容：`train -h` 返回码为 0。"""
    with pytest.raises(SystemExit) as ex:
        train_cli(["-h"])
    assert ex.value.code == 0
