"""
整体覆盖：`deepks.interface.objectives.DescriptorPropertyObjectiveAdapter` 的多损失开关组合。

测试列表：
- `test_evaluator_energy_only`
- `test_evaluator_energy_force_stress_switches`
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("pyabacus")
pytestmark = pytest.mark.pyabacus

from deepks.interface.objectives import build_descriptor_property_objective


class SumModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.w = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

	def forward(self, x):
		return x.sum(dim=(1, 2)).unsqueeze(-1) * self.w


def test_evaluator_energy_only():
	"""
	依赖：`deepks.interface.objectives.DescriptorPropertyObjectiveAdapter`。
	测试内容：仅能量损失开启时，返回 `[e_loss, tot_loss]` 两项且二者相等。
	"""
	model = SumModel().double()
	sample = {
		"lb_e": torch.zeros((2, 1), dtype=torch.float64),
		"eig": torch.ones((2, 3, 4), dtype=torch.float64),
	}
	ev = build_descriptor_property_objective(
		{"energy_factor": 1.0, "force_factor": 0.0, "stress_factor": 0.0},
		property_scheme="energy_descriptor",
	)
	out = ev.compute_losses(model, sample)
	assert len(out) == 2
	assert torch.isclose(out[0], out[-1])


def test_evaluator_energy_force_stress_switches():
	"""
	依赖：`deepks.interface.objectives.DescriptorPropertyObjectiveAdapter`。
	测试内容：能量+力+应力开启时，输出应包含对应三项及总损失。
	"""
	b, natm, ndesc = 2, 2, 3
	model = SumModel().double()
	sample = {
		"lb_e": torch.zeros((b, 1), dtype=torch.float64),
		"eig": torch.ones((b, natm, ndesc), dtype=torch.float64),
		"lb_f": torch.zeros((b, natm, 3), dtype=torch.float64),
		"gvx": torch.zeros((b, natm, 3, natm, ndesc), dtype=torch.float64),
		"lb_s": torch.zeros((b, 6), dtype=torch.float64),
		"gvepsl": torch.zeros((b, 6, natm, ndesc), dtype=torch.float64),
	}
	ev = build_descriptor_property_objective(
		{"energy_factor": 1.0, "force_factor": 1.0, "stress_factor": 1.0},
		property_scheme="energy_descriptor",
	)
	out = ev.compute_losses(model, sample)
	# e + f + s + total
	assert len(out) == 4
	assert torch.isfinite(out[-1])


def test_evaluator_none_energy_per_atom_defaults_to_zero():
	"""
	依赖：`deepks.interface.objectives.DescriptorPropertyObjectiveAdapter`。
	测试内容：`energy_per_atom=None` 时应按 0 处理，不应保留为 None。
	"""
	ev = build_descriptor_property_objective(
		{"energy_per_atom": None},
		property_scheme="energy_descriptor",
	)
	assert ev.energy_per_atom == 0
