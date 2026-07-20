"""
整体覆盖：统一 CLI 行为。

测试列表：
- `test_unified_cli_with_config`
- `test_unified_cli_missing_command`
- `test_unified_cli_unknown_command`
"""

import pytest
import tempfile
import os
import sys


def test_unified_cli_with_config():
    """测试统一 CLI 加载配置文件。"""
    from deepks.__main__ import main

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('command: scf\n')
        f.write('scf_soft: pyscf\n')
        f.write('systems:\n  - sys1\n')
        f.flush()
        config_path = f.name

    try:
        original_argv = sys.argv
        sys.argv = ['deepks', config_path]

        # 应该尝试运行但会因为缺少 pyscf 而失败
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = original_argv
        os.unlink(config_path)


def test_unified_cli_missing_command():
    """测试配置文件缺少 command 字段。"""
    from deepks.__main__ import main

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('systems:\n  - sys1\n')
        f.flush()
        config_path = f.name

    try:
        original_argv = sys.argv
        sys.argv = ['deepks', config_path]

        with pytest.raises(SystemExit) as ex:
            main()
        assert ex.value.code == 1
    finally:
        sys.argv = original_argv
        os.unlink(config_path)


def test_unified_cli_unknown_command():
    """测试未知命令。"""
    from deepks.__main__ import main

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('command: unknown\n')
        f.flush()
        config_path = f.name

    try:
        original_argv = sys.argv
        sys.argv = ['deepks', config_path]

        with pytest.raises(SystemExit) as ex:
            main()
        assert ex.value.code == 1
    finally:
        sys.argv = original_argv
        os.unlink(config_path)
