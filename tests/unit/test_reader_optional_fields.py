"""
整体覆盖：`deepks.model.reader.Reader` 在可选字段缺失时的行为。

测试列表：
- `test_reader_minimal_fields_only`
- `test_reader_with_conv_filter`
"""

import numpy as np

from deepks.model.reader import Reader


def test_reader_minimal_fields_only(tmp_path):
	"""
	依赖：`deepks.model.reader.Reader`。
	测试内容：仅提供最小字段 `l_e_delta.npy/dm_eig.npy` 时应正常读取且不强依赖其他标签。
	"""
	nframe, natm, ndesc = 3, 2, 4
	np.save(tmp_path / "l_e_delta.npy", np.arange(nframe).reshape(-1, 1))
	np.save(tmp_path / "dm_eig.npy", np.random.randn(nframe, natm, ndesc))

	rd = Reader(str(tmp_path), batch_size=2)
	sample = rd.sample_all()
	assert set(sample.keys()) == {"lb_e", "eig"}
	assert sample["lb_e"].shape == (nframe, 1)
	assert sample["eig"].shape == (nframe, natm, ndesc)


def test_reader_with_conv_filter(tmp_path):
	"""
	依赖：`deepks.model.reader.Reader`。
	测试内容：存在 `conv.npy` 时应只保留收敛帧。
	"""
	nframe, natm, ndesc = 4, 2, 3
	e = np.arange(nframe).reshape(-1, 1)
	d = np.random.randn(nframe, natm, ndesc)
	conv = np.array([True, False, True, False])
	np.save(tmp_path / "l_e_delta.npy", e)
	np.save(tmp_path / "dm_eig.npy", d)
	np.save(tmp_path / "conv.npy", conv)

	rd = Reader(str(tmp_path), batch_size=3)
	sample = rd.sample_all()
	assert rd.get_nframes() == 2
	assert sample["lb_e"].shape[0] == 2


