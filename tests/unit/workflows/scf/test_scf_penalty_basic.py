"""
整体覆盖：PySCF 接口侧 penalty 组件的基础行为。

说明：
- 本文件属于 PySCF 接口测试；
- 当前环境未安装 PySCF 时会自动跳过。

测试列表：
- `test_select_penalty_basic`
- `test_select_penalty_invalid`
- `test_density_penalty_init_and_defaults`
- `test_coulomb_penalty_init_and_defaults`
"""

import numpy as np
import pytest

pytest.importorskip("pyscf")

from deepks.physics.backends.pyscf.penalty import CoulombPenalty, DensityPenalty, select_penalty


def test_select_penalty_basic():
	"""
	依赖：`deepks.physics.backends.pyscf.penalty.select_penalty`。
	测试内容：验证已注册类型名能映射到正确 penalty 类。
	"""
	assert select_penalty("density") is DensityPenalty
	assert select_penalty("coulomb") is CoulombPenalty


def test_select_penalty_invalid():
	"""
	依赖：`deepks.physics.backends.pyscf.penalty.select_penalty`。
	测试内容：未知 penalty 类型应抛出 `ValueError`。
	"""
	with pytest.raises(ValueError):
		select_penalty("not-exist")


def test_density_penalty_init_and_defaults(tmp_path):
	"""
	依赖：`deepks.physics.backends.pyscf.penalty.DensityPenalty`。
	测试内容：验证目标密度加载、参数初始化与默认字段状态。
	"""
	dm = np.eye(2)
	fn = tmp_path / "dm.npy"
	np.save(fn, dm)
	p = DensityPenalty(str(fn), strength=2.5, random=False, start_cycle=3)
	assert np.allclose(p.dm_t, dm)
	assert p.init_strength == 2.5
	assert p.strength == 2.5
	assert p.start_cycle == 3
	assert p.grids is None
	assert p.ao_value is None


def test_coulomb_penalty_init_and_defaults(tmp_path):
	"""
	依赖：`deepks.physics.backends.pyscf.penalty.CoulombPenalty`。
	测试内容：验证目标密度加载与初始化参数正确。
	"""
	dm = np.eye(3)
	fn = tmp_path / "dm.npy"
	np.save(fn, dm)
	p = CoulombPenalty(str(fn), strength=1.2, random=False, start_cycle=5)
	assert np.allclose(p.dm_t, dm)
	assert p.init_strength == 1.2
	assert p.strength == 1.2
	assert p.start_cycle == 5


