"""
整体覆盖：`deepks.physics.backends.pyscf.basis` 与 `deepks.io.utils` 中 basis/元素常数表工具函数。

测试列表：
- `test_load_basis_from_npy_and_npz`
- `test_save_basis_and_get_shell_sec`
- `test_elem_table_roundtrip`
"""

import numpy as np

from deepks.physics.backends.pyscf.basis import get_shell_sec, load_basis, save_basis
from deepks.io.utils import load_elem_table, save_elem_table


def test_load_basis_from_npy_and_npz(tmp_path):
	"""
	依赖：`deepks.physics.backends.pyscf.basis.load_basis`。
	测试内容：验证从 `.npy` 与 `.npz` 两种文件格式加载 basis 的行为。
	"""
	table = np.array([[1.0, 2.0], [3.0, 4.0]])
	fn_npy = tmp_path / "b.npy"
	np.save(fn_npy, table)
	b_npy = load_basis(str(fn_npy))
	assert len(b_npy) == 3
	assert b_npy[0][0] == 0 and b_npy[1][0] == 1 and b_npy[2][0] == 2

	fn_npz = tmp_path / "b.npz"
	np.savez(fn_npz, arr_0_L0=table, arr_1_L2=table)
	b_npz = load_basis(str(fn_npz))
	ls = [x[0] for x in b_npz]
	assert 0 in ls and 2 in ls


def test_save_basis_and_get_shell_sec(tmp_path):
	"""
	依赖：`deepks.physics.backends.pyscf.basis.save_basis/get_shell_sec/load_basis`。
	测试内容：验证 basis 保存后可读回，且 `get_shell_sec` 结果正确。
	"""
	basis = [
		[0, [1.0, 0.0]],        # s shell, nb=1 -> 1
		[1, [1.0, 0.0]],        # p shell, nb=1 -> 3
		[2, [1.0, 0.0], [0.5, 0.2]],  # d shell, nb=2 -> 5,5
	]
	fn = tmp_path / "basis.npz"
	save_basis(str(fn), basis)
	loaded = load_basis(str(fn))
	sec = get_shell_sec(loaded)
	# 当前实现中 nb 从 c0 推导（len(c0)-1），因此此例 d-shell 贡献一个 5
	assert sec == [1, 3, 5]


def test_elem_table_roundtrip(tmp_path):
	"""
	依赖：`deepks.io.utils.save_elem_table/load_elem_table`。
	测试内容：验证元素常数表文件写入与读取回环一致。
	"""
	fn = tmp_path / "elem.tab"
	elem = (np.array([1, 6, 8]), np.array([-0.1, -0.2, -0.3]))
	save_elem_table(str(fn), elem)
	out_e, out_c = load_elem_table(str(fn))
	assert np.array_equal(out_e, elem[0])
	assert np.allclose(out_c, elem[1])


