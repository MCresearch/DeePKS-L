"""
整体覆盖：ABACUS 接口模板转换与任务对象构建。

测试列表：
- `test_coord_to_atom_basic`
- `test_coord_to_atom_missing_coord_raises`
- `test_make_convert_scf_abacus_returns_pythontask`
"""

from pathlib import Path

import numpy as np
import pytest

from deepks.iterate.template_abacus import coord_to_atom, make_convert_scf_abacus
from deepks.task.task import PythonTask


def test_coord_to_atom_basic(tmp_path):
	"""
	依赖：`deepks.iterate.template_abacus.coord_to_atom`。
	测试内容：验证 `coord.npy + type_map.raw + type.raw` 正确合成为 `atom` 三维数组。
	"""
	p = tmp_path / "sys"
	p.mkdir()
	coords = np.array([[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]])
	np.save(p / "coord.npy", coords)
	(p / "type_map.raw").write_text("H O\n", encoding="utf-8")
	np.savetxt(p / "type.raw", np.array([1, 2], dtype=int), fmt="%d")

	atom = coord_to_atom(str(p))
	assert atom.shape == (1, 2, 4)
	# 第一列是元素序号，后 3 列是坐标
	assert np.allclose(atom[0, 0, 1:], [0.0, 0.1, 0.2])
	assert np.allclose(atom[0, 1, 1:], [1.0, 1.1, 1.2])


def test_coord_to_atom_missing_coord_raises(tmp_path):
	"""
	依赖：`deepks.iterate.template_abacus.coord_to_atom`。
	测试内容：当缺少 `coord.npy` 时应抛出 `FileNotFoundError`。
	"""
	p = tmp_path / "bad"
	p.mkdir()
	with pytest.raises(FileNotFoundError):
		coord_to_atom(str(p))


def test_make_convert_scf_abacus_returns_pythontask(tmp_path):
	"""
	依赖：`deepks.iterate.template_abacus.make_convert_scf_abacus`。
	测试内容：在本地无集群环境下，验证该函数能构建可执行 `PythonTask` 且参数完整。
	"""
	trn = tmp_path / "group.00"
	tst = tmp_path / "group.01"
	trn.mkdir()
	tst.mkdir()

	task = make_convert_scf_abacus(
		systems_train=[str(trn)],
		systems_test=[str(tst)],
		no_model=True,
		resources={"task_per_node": 1},
		orb_files=["x.orb"],
		pp_files=["x.upf"],
		proj_file=["jle.orb"],
		run_cmd="mpirun",
		abacus_path="/bin/echo",
	)
	assert isinstance(task, PythonTask)
	assert "systems_train" in task.call_kwargs
	assert "systems_test" in task.call_kwargs
	assert task.call_kwargs["no_model"] is True


