"""
整体覆盖：`deepks.io.utils` 与 `deepks.physics.backends.abacus.utils` 中 I/O/路径与 xyz 工具函数的核心行为。

测试列表：
- `test_yaml_roundtrip`
- `test_deep_update_recursive`
- `test_get_with_prefix_dir_and_base_modes`
- `test_flat_file_list_with_nested_listfile`
- `test_parse_xyz_basic`
"""

from pathlib import Path

import numpy as np

from deepks.io.utils import deep_update, flat_file_list, get_with_prefix, load_yaml, save_yaml
from deepks.physics.backends.abacus.utils import parse_xyz


def test_yaml_roundtrip(tmp_path):
	"""
	依赖：`deepks.io.utils.save_yaml/load_yaml`。
	测试内容：YAML 保存与读取应保持数据结构一致。
	"""
	fp = tmp_path / "cfg" / "a.yaml"
	data = {"a": 1, "b": {"x": [1, 2, 3], "y": "ok"}}
	save_yaml(data, str(fp))
	out = load_yaml(str(fp))
	assert out == data


def test_deep_update_recursive():
	"""
	依赖：`deepks.io.utils.deep_update`。
	测试内容：递归更新时子字典应 merge 而非整体覆盖。
	"""
	base = {"a": 1, "b": {"x": 1, "y": 2}}
	upd = {"b": {"y": 9, "z": 10}, "c": 3}
	res = deep_update(base, upd)
	assert res == {"a": 1, "b": {"x": 1, "y": 9, "z": 10}, "c": 3}


def test_get_with_prefix_dir_and_base_modes(tmp_path):
	"""
	依赖：`deepks.io.utils.get_with_prefix`。
	测试内容：验证目录模式与 base 前缀模式都能正确解析；nullable 行为正确。
	"""
	d = tmp_path / "d"
	d.mkdir()
	p1 = d / "energy.npy"
	p1.write_text("x", encoding="utf-8")
	assert get_with_prefix("energy", str(d), prefer=".npy") == str(p1)

	base = tmp_path / "frame001"
	p2 = tmp_path / "frame001.energy.npy"
	p2.write_text("x", encoding="utf-8")
	assert get_with_prefix("energy", str(base), prefer=".npy") == str(p2)

	missing = get_with_prefix("notfound", str(base), prefer=".npy", nullable=True)
	assert missing is None


def test_flat_file_list_with_nested_listfile(tmp_path):
	"""
	依赖：`deepks.io.utils.flat_file_list`。
	测试内容：验证支持“文件列表文件”二级展开，并按 filter 过滤目标文件。
	"""
	a = tmp_path / "a.xyz"
	b = tmp_path / "b.xyz"
	c = tmp_path / "c.txt"
	a.write_text("", encoding="utf-8")
	b.write_text("", encoding="utf-8")
	c.write_text("", encoding="utf-8")

	lst = tmp_path / "list.raw"
	lst.write_text(f"{tmp_path}/*.xyz\n", encoding="utf-8")

	out = flat_file_list([str(lst)], filter_func=lambda p: p.endswith(".xyz"), sort=True)
	assert out == [str(a), str(b)]


def test_parse_xyz_basic(tmp_path):
	"""
	依赖：`deepks.physics.backends.abacus.utils.parse_xyz`。
	测试内容：验证 xyz 解析出的原子数、注释、元素与坐标形状/数值。
	"""
	fp = tmp_path / "m.xyz"
	fp.write_text(
		"2\ncomment line\nH 0.0 0.1 0.2\nO 1.0 1.1 1.2\n",
		encoding="utf-8",
	)
	natom, comment, elems, coords = parse_xyz(str(fp))
	assert natom == 2
	assert comment == "comment line"
	assert elems == ["H", "O"]
	assert coords.shape == (2, 3)
	assert np.allclose(coords[1], [1.0, 1.1, 1.2])


