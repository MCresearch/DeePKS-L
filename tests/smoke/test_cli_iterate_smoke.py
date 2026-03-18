"""
Smoke 测试：验证统一 CLI 的 iterate 功能。
"""

import pytest
import sys


def test_iterate_cli_help_exit_zero():
    """测试统一 CLI 帮助信息。"""
    from deepks.cli.main import main

    original_argv = sys.argv
    try:
        sys.argv = ['deepks', '--help']
        with pytest.raises(SystemExit) as ex:
            main()
        assert ex.value.code == 0
    finally:
        sys.argv = original_argv
