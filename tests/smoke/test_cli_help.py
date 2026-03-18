"""
Smoke 测试：验证统一 CLI 入口是否可调用。

整体覆盖：
- `deepks` 统一入口帮助页可调用性。

测试列表：
- `test_main_help`
"""

import pytest
import sys


def test_main_help():
    """测试统一 CLI 帮助信息。"""
    from deepks.cli.main import main

    # 保存原始 argv
    original_argv = sys.argv
    try:
        # 模拟 --help 参数
        sys.argv = ['deepks', '--help']
        with pytest.raises(SystemExit) as ex:
            main()
        # --help 应该返回 0
        assert ex.value.code == 0
    finally:
        sys.argv = original_argv
