"""
整体覆盖：`deepks scf` 命令入口可用性。

测试列表：
- `test_scf_cli_help_exit_zero`
"""

import pytest

from deepks.cli.main import scf_cli


def test_scf_cli_help_exit_zero():
    """依赖：`deepks.cli.main.scf_cli`。测试内容：`scf -h` 返回码为 0。"""
    with pytest.raises(SystemExit) as ex:
        scf_cli(["-h"])
    assert ex.value.code == 0
