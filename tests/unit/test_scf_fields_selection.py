"""
整体覆盖：PySCF 接口侧 `deepks.core.physics.pyscf.fields` 的字段选择与单位辅助函数。

说明：
- 本文件属于 PySCF 接口测试；
- 当前环境未安装 PySCF，因此通过 `importorskip("pyscf")` 自动跳过；
- 与 ABACUS 接口相关测试位于独立文件：
	- `tests/unit/test_abacus_generator_input.py`
	- `tests/unit/test_abacus_template_convert.py`
	- `tests/integration/test_abacus_stats_gather_minimal.py`

测试列表：
- `test_select_fields_by_name_and_alias`
- `test_select_fields_case_insensitive`
- `test_isinbohr_and_lunit`
- `test_atom_data_filters_ghost_atoms`
"""

import numpy as np
import pytest

pytest.importorskip("pyscf")
from deepks.default import BOHR2ANG
from deepks.core.physics.pyscf.fields import _Lunit, atom_data, isinbohr, select_fields


def test_select_fields_by_name_and_alias():
	"""
	依赖：`deepks.core.physics.pyscf.fields.select_fields`。
	测试内容：支持按字段名与别名选择，并正确分流到 scf/grad 两类。
	"""
	sel = select_fields(["e_tot", "gvx", "force"])
	scf_names = {f.name for f in sel["scf"]}
	grad_names = {f.name for f in sel["grad"]}
	assert "e_tot" in scf_names
	assert "grad_vx" in grad_names  # gvx alias
	assert "f_tot" in grad_names    # force alias


def test_select_fields_case_insensitive():
	"""
	依赖：`deepks.core.physics.pyscf.fields.select_fields`。
	测试内容：字段选择应大小写不敏感。
	"""
	sel = select_fields(["E_TOT", "ConVerGed"])
	names = {f.name for f in sel["scf"]}
	assert "e_tot" in names
	assert "conv" in names


def test_isinbohr_and_lunit():
	"""
	依赖：`deepks.core.physics.pyscf.fields.isinbohr/_Lunit`。
	测试内容：Bohr/AU 单位返回 1；Angstrom 返回 `BOHR2ANG`。
	"""
	class Mol:
		def __init__(self, unit):
			self.unit = unit

	m1 = Mol("Bohr")
	m2 = Mol("AU")
	m3 = Mol("Angstrom")

	assert isinbohr(m1)
	assert isinbohr(m2)
	assert not isinbohr(m3)
	assert _Lunit(m1) == 1.0
	assert _Lunit(m3) == BOHR2ANG


def test_atom_data_filters_ghost_atoms():
	"""
	依赖：`deepks.core.physics.pyscf.fields.atom_data`。
	测试内容：应过滤掉元素名以 `X` 开头的 ghost atom。
	"""
	class Mol:
		natm = 3
		elements = ["H", "X-He", "O"]

		@staticmethod
		def atom_charges():
			return np.array([1.0, 2.0, 8.0])

		@staticmethod
		def atom_coords(unit="Bohr"):
			return np.array(
				[
					[0.0, 0.0, 0.0],
					[1.0, 1.0, 1.0],
					[2.0, 2.0, 2.0],
				]
			)

	out = atom_data(Mol())
	assert out.shape == (2, 4)
	assert np.allclose(out[:, 0], [1.0, 8.0])


