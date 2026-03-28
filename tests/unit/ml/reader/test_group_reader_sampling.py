"""
整体覆盖：`deepks.io.readers.GroupReader` 抽样与批处理行为。

测试列表：
- `test_groupreader_prob_and_grouping`
- `test_groupreader_sample_all_batch`
"""

import numpy as np

from deepks.io.readers import GroupReader


def _make_sys(path, nframe, natm, ndesc):
	path.mkdir(parents=True, exist_ok=True)
	np.save(path / "l_e_delta.npy", np.random.randn(nframe, 1))
	np.save(path / "dm_eig.npy", np.random.randn(nframe, natm, ndesc))


def test_groupreader_prob_and_grouping(tmp_path):
	"""
	依赖：`deepks.io.readers.GroupReader`。
	测试内容：验证系统概率与按 shape 分组逻辑正确。
	"""
	s1 = tmp_path / "g1"
	s2 = tmp_path / "g2"
	s3 = tmp_path / "g3"
	_make_sys(s1, nframe=4, natm=2, ndesc=3)
	_make_sys(s2, nframe=2, natm=3, ndesc=3)
	_make_sys(s3, nframe=2, natm=3, ndesc=3)

	gr = GroupReader([str(s1), str(s2), str(s3)], batch_size=2, group_batch=2)
	assert gr.nsystems == 3
	assert np.isclose(sum(gr.sys_prob), 1.0)
	assert (2, None) in gr.group_dict
	assert (3, None) in gr.group_dict


def test_groupreader_sample_all_batch(tmp_path):
	"""
	依赖：`deepks.io.readers.GroupReader.sample_all_batch`。
	测试内容：验证可按 batch 迭代返回全量数据，且字段形状正确。
	"""
	s1 = tmp_path / "g1"
	s2 = tmp_path / "g2"
	_make_sys(s1, nframe=3, natm=2, ndesc=4)
	_make_sys(s2, nframe=3, natm=2, ndesc=4)

	gr = GroupReader([str(s1), str(s2)], batch_size=2, group_batch=1)
	batches = list(gr.sample_all_batch())
	# 两个系统各 3 帧，batch=2 => 每系统 2 个 batch（2 + 1），总计 4
	assert len(batches) == 4
	for b in batches:
		assert "lb_e" in b and "eig" in b
		assert b["lb_e"].ndim == 2
		assert b["eig"].ndim == 3


