"""
Smoke 测试：只验证 CLI 帮助入口是否可调用。

整体覆盖：
- `deepks` 主入口与各子命令 `-h` 帮助页可调用性。

测试列表：
- `test_main_help`
- `test_subcommand_help_train`
- `test_subcommand_help_test`
- `test_subcommand_help_scf`
- `test_subcommand_help_stats`
- `test_subcommand_help_iterate`
"""

import pytest

from deepks.main import (
    iter_cli,
    main_cli,
    scf_cli,
    stats_cli,
    test_cli as dks_test_cli,
    train_cli,
)


def _assert_help_ok(func):
    with pytest.raises(SystemExit) as ex:
        func(["-h"])
    assert ex.value.code == 0


def test_main_help():
    """依赖：`deepks.main.main_cli`。测试内容：主命令 `-h` 返回码为 0。"""
    _assert_help_ok(main_cli)


def test_subcommand_help_train():
    """依赖：`deepks.main.train_cli`。测试内容：`train -h` 正常退出。"""
    _assert_help_ok(train_cli)


def test_subcommand_help_test():
    """依赖：`deepks.main.test_cli`。测试内容：`test -h` 正常退出。"""
    _assert_help_ok(dks_test_cli)


def test_subcommand_help_scf():
    """依赖：`deepks.main.scf_cli`。测试内容：`scf -h` 正常退出。"""
    _assert_help_ok(scf_cli)


def test_subcommand_help_stats():
    """依赖：`deepks.main.stats_cli`。测试内容：`stats -h` 正常退出。"""
    _assert_help_ok(stats_cli)


def test_subcommand_help_iterate():
    """依赖：`deepks.main.iter_cli`。测试内容：`iterate -h` 正常退出。"""
    _assert_help_ok(iter_cli)
