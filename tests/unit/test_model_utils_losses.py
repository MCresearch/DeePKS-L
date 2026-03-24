"""
整体覆盖：`deepks.ml.utils` 中损失函数行为。

测试列表：
- `test_make_loss_mean_sum_none_batch`
- `test_make_loss_with_shrink_and_cap`
- `test_make_loss_invalid_reduction_raises`
- `test_loss_hr_formula`
"""

import pytest
import torch

pytest.importorskip("pyabacus")

from deepks.ml.utils import loss_hr, make_loss


def test_make_loss_mean_sum_none_batch():
	"""
	依赖：`deepks.ml.utils.make_loss`。
	测试内容：验证 reduction=`mean/sum/none/batch` 的数值语义。
	"""
	inp = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
	tgt = torch.tensor([[2.0, 0.0], [3.0, 1.0]])
	sq = (tgt - inp).abs() ** 2

	assert torch.isclose(make_loss(reduction="mean")(inp, tgt), sq.mean())
	assert torch.isclose(make_loss(reduction="sum")(inp, tgt), sq.sum())
	assert torch.allclose(make_loss(reduction="none")(inp, tgt), sq)
	assert torch.isclose(make_loss(reduction="batch")(inp, tgt), sq.sum() / inp.shape[0])


def test_make_loss_with_shrink_and_cap():
	"""
	依赖：`deepks.ml.utils.make_loss`。
	测试内容：验证 `shrink` 与 `cap` 同时启用时，输出符合实现定义（softshrink + smooth cap）。
	"""
	inp = torch.tensor([0.0, 0.0, 0.0])
	tgt = torch.tensor([0.1, 1.0, 3.0])
	fn = make_loss(shrink=0.5, cap=1.0, reduction="none")

	# shrink 后 diff: [0, 0.5, 2.5]
	# sqdf: [0, 0.25, 6.25]
	# cap=1 对 2.5 走线性段: 1*(2*2.5-1)=4
	ref = torch.tensor([0.0, 0.25, 4.0])
	out = fn(inp, tgt)
	assert torch.allclose(out, ref)


def test_make_loss_invalid_reduction_raises():
	"""
	依赖：`deepks.ml.utils.make_loss`。
	测试内容：非法 reduction 应抛出 `ValueError`。
	"""
	fn = make_loss(reduction="bad")
	with pytest.raises(ValueError):
		fn(torch.zeros(1), torch.ones(1))


def test_loss_hr_formula():
	"""
	依赖：`deepks.ml.utils.loss_hr`。
	测试内容：验证 `loss_hr = sum(|diff|^2) / R_range / nlocal / nframe`。
	"""
	# shape: [nframe=2, R=2, R=2, R=2, nlocal=2, nlocal=2]
	inp = torch.zeros((2, 2, 2, 2, 2, 2), dtype=torch.float64)
	tgt = torch.ones_like(inp)
	# sum(|1-0|^2)=元素总数=2*2*2*2*2*2=64
	# R_range=2, nlocal=2, nframe=2 => 64/2/2/2 = 8
	out = loss_hr(inp, tgt)
	assert torch.isclose(out, torch.tensor(8.0, dtype=torch.float64))


