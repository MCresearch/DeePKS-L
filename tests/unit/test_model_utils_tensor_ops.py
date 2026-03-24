"""
整体覆盖：`deepks.physics.backends.abacus.utils` 中张量与索引工具函数。

测试列表：
- `test_r2ir_ir2r_for_int_numpy_torch`
- `test_read_csr_minimal`
"""

import numpy as np
import torch

from deepks.physics.backends.abacus.utils import R2iR, iR2R, read_csr


def test_r2ir_ir2r_for_int_numpy_torch():
	"""
	依赖：`deepks.physics.backends.abacus.utils.R2iR/iR2R`。
	测试内容：验证 int / numpy / torch 三类输入的双向转换一致性。
	"""
	assert R2iR(0) == 0
	assert R2iR(1) == 1
	assert R2iR(-1) == 2
	assert iR2R(0) == 0
	assert iR2R(1) == 1
	assert iR2R(2) == -1

	arr = np.array([-2, -1, 0, 1, 2])
	ir = R2iR(arr)
	back = iR2R(ir)
	assert np.array_equal(back, arr)

	t = torch.tensor([-2, -1, 0, 1, 2])
	ir_t = R2iR(t)
	back_t = iR2R(ir_t)
	assert torch.equal(back_t, t)


def test_read_csr_minimal(tmp_path):
	"""
	依赖：`deepks.physics.backends.abacus.utils.read_csr`。
	测试内容：构造最小 CSR 文本，验证稀疏张量维度与非零元素正确读取。
	"""
	fp = tmp_path / "m.csr"
	fp.write_text(
		"dim 2\n"
		"num 1\n"
		"0 0 0 1\n"   # Rx Ry Rz nnz
		"1.0\n"        # data
		"1\n"          # indices
		"0 1 1\n",     # indptr (dim+1)
		encoding="utf-8",
	)
	st = read_csr(str(fp), dtype=torch.float64)
	assert tuple(st.size()) == (1, 1, 1, 2, 2)
	dense = st.to_dense()
	assert torch.isclose(dense[0, 0, 0, 0, 1], torch.tensor(1.0, dtype=torch.float64))


