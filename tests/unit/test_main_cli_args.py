"""
整体覆盖：CLI 分发层行为。

测试列表：
- `test_main_cli_dispatch_train`
- `test_main_cli_dispatch_iterate_alias`
- `test_main_cli_unknown_command_returns_valueerror_obj`
"""

import deepks.cli.main as m


def test_main_cli_dispatch_train(monkeypatch):
	"""
	依赖：`deepks.cli.main.main_cli` 与 `monkeypatch`。
	测试内容：输入 `train` 时，命令被分发至 `train_cli` 且参数保持原样。
	"""
	captured = {}

	def fake_train_cli(args=None):
		captured["name"] = "train"
		captured["args"] = args

	monkeypatch.setattr(m, "train_cli", fake_train_cli)
	m.main_cli(["train", "--foo", "bar"])

	assert captured["name"] == "train"
	assert captured["args"] == ["--foo", "bar"]


def test_main_cli_dispatch_iterate_alias(monkeypatch):
	"""
	依赖：`deepks.cli.main.main_cli` 与 `monkeypatch`。
	测试内容：`iter` 别名应分发到 `iter_cli`。
	"""
	captured = {}

	def fake_iter_cli(args=None):
		captured["name"] = "iterate"
		captured["args"] = args

	monkeypatch.setattr(m, "iter_cli", fake_iter_cli)
	m.main_cli(["iter", "args.yaml"])

	assert captured["name"] == "iterate"
	assert captured["args"] == ["args.yaml"]


def test_main_cli_unknown_command_returns_valueerror_obj():
	"""
	依赖：`deepks.cli.main.main_cli`。
	测试内容：未知命令保持当前兼容行为（返回 `ValueError` 对象）。
	"""
	res = m.main_cli(["unknown-cmd"])
	assert isinstance(res, ValueError)


