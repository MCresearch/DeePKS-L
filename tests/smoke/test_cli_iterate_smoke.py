"""
整体覆盖：`deepks iterate` 命令入口可用性。

测试列表：
- `test_iterate_cli_help_exit_zero`
"""

import pytest

from deepks.main import iter_cli


def test_iterate_cli_help_exit_zero():
    """依赖：`deepks.main.iter_cli`。测试内容：`iterate -h` 返回码为 0。"""
    with pytest.raises(SystemExit) as ex:
        iter_cli(["-h"])
    assert ex.value.code == 0
