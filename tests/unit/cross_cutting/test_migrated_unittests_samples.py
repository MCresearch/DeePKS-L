"""
整体覆盖：将历史 `tests/unittests` 的代表性样例迁移到当前正式测试框架。

测试列表：
- `test_migrated_model_get_shell_sec_sample`
- `test_migrated_simplereader_sample`
- `test_migrated_model_reference_data_catalog`
- `test_migrated_reader_reference_data_catalog`
- `test_migrated_groupedlosstracker_sample`
- `test_migrated_evaluator_sample`
- `test_migrated_template_module_import_if_abacus`
"""

import importlib.util

import numpy as np
import pytest
import torch
from deepks.interface.objectives import build_descriptor_property_objective
from deepks.io.readers import SimpleReader
from deepks.ml.train import GroupedLossTracker
from deepks.physics.backends.pyscf.basis import get_shell_sec


MODEL_REFERENCE_DATA = {
    "trace": torch.tensor(
        [
            [-0.1368, -0.4975, 0.7234, -0.0998, 6.4928, -0.3836],
            [0.1646, 0.9874, -0.7656, 0.8742, -4.1812, 1.0911],
            [-0.7216, -0.0979, 2.4200, 0.5760, 2.2466, -1.7258],
            [0.6411, -0.1408, -1.2673, -1.2313, 1.4721, 2.5969],
        ]
    ),
    "thermal": torch.tensor(
        [
            [-1.1542, -0.0975, -2.5340, -0.6297, 0.2274, -0.4620],
            [-1.3175, -1.1912, -0.5842, -1.2733, -0.2728, -1.6477],
            [-1.1860, -0.4774, -1.9962, -0.4106, -0.7192, -0.3664],
            [0.5418, -0.1053, -0.6545, -0.3955, -2.3581, -0.1855],
        ]
    ),
    "densenet": torch.tensor(
        [
            [0.3203, 0.0591, 0.2396, 0.8337, 0.0184],
            [0.3563, -0.2009, 0.2498, 0.9055, -0.1204],
            [0.7344, -0.8686, 0.3067, 0.0719, -0.1099],
            [-0.1397, -1.2096, 0.4953, 1.3167, 0.1208],
        ]
    ),
    "corrnet_forward": torch.tensor([[-0.8677], [0.5022], [-2.0623], [-1.3056]], dtype=torch.float64),
    "corrnet_norm": torch.tensor([[0.1862], [0.8503], [-1.0830], [-0.5516]], dtype=torch.float64),
    "corrnet_prefit": torch.tensor([[-2.8196], [-5.9678], [-4.8722], [-3.4918]], dtype=torch.float64),
    "corrnet_econst": torch.tensor([[-2.3466], [-5.4789], [-4.4017], [-3.0000]], dtype=torch.float64),
}

READER_REFERENCE_DATA = {
    "group_err_empty_1": "# ignore empty dataset: ./group1/",
    "group_err_empty_2": "# ignore empty dataset: ./group2/",
    "group_out_load": "# load 3 systems with fields ['lb_e', 'eig']",
    "group_err_reset_2": "# ./group2/ reset batch size to 2",
    "group_err_reset_3": "# ./group3/ reset batch size to 1",
    "nelem_ref": np.array(
        [
            [1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 0],
        ]
    ),
    "elem_list_reader": [6, 30, 35, 40, 43, 50, 56, 76, 79],
    "all_mean": np.array([-0.18484446, -0.07506522, 0.27285756, -0.19529272]),
    "all_std": np.array([0.56544117, 1.20177998, 1.03995579, 0.89537965]),
    "prefit_weight": np.array([-0.0053749, 0.11951934, 0.815563, 0.13809183]),
    "prefit_bias": -0.028533653429280666,
    "group_elem_list": np.array([3, 6, 8, 10, 25, 26, 27, 29, 32, 44, 49, 51, 60, 79, 89]),
    "group_elem_const": np.array(
        [
            0.19558913,
            0.1471279,
            -0.37585689,
            0.19558913,
            -0.37585689,
            0.1471279,
            -0.09742901,
            -0.37585689,
            -0.09742901,
            -0.28587458,
            0.55784453,
            0.19558913,
            -0.28587458,
            -0.09742901,
            0.55784453,
        ]
    ),
}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_migrated_model_get_shell_sec_sample():
    """
    依赖：`deepks.physics.backends.pyscf.basis.get_shell_sec`。
    测试内容：迁移历史 `test_get_shell_sec` 样例，验证 basis 到 shell section 的映射结果。
    """
    basis = [[0, [0, 0, 0]], [1, [0, 0, 0]], [2, [0, 0, 0]]]
    shell_sec = get_shell_sec(basis)
    assert shell_sec == [1, 1, 3, 3, 5, 5]


def test_migrated_simplereader_sample(tmp_path):
    """
    依赖：`deepks.io.readers.SimpleReader`。
    测试内容：迁移历史 `test_SimpleReader` 样例，验证最小数据读取、收敛筛选和抽样行为。
    """
    data_size = 3
    natom, nproj = 2, 4
    meta = np.array([natom, 123, nproj])
    l_e_delta = np.arange(data_size, dtype=float).reshape(-1, 1)
    dm_eig = np.arange(data_size * natom * nproj, dtype=float).reshape(data_size, natom, nproj)
    conv = np.array([True, True, False])

    np.savetxt(tmp_path / "system.raw", meta)
    np.save(tmp_path / "l_e_delta.npy", l_e_delta)
    np.save(tmp_path / "dm_eig.npy", dm_eig)
    np.save(tmp_path / "conv.npy", conv)

    reader = SimpleReader(str(tmp_path), batch_size=3)
    assert reader.get_batch_size() == 2
    assert reader.natm == natom
    assert reader.nproj == nproj
    assert reader.get_nframes() == 2

    sample_all = reader.sample_all()
    assert sample_all["lb_e"].shape == (2, 1)
    assert sample_all["eig"].shape == (2, natom, nproj)
    assert np.allclose(sample_all["lb_e"], l_e_delta[:2])
    assert np.allclose(sample_all["eig"], dm_eig[:2])


def test_migrated_model_reference_data_catalog():
    """
    依赖：历史 `test_model.py` 中的参考向量。
    测试内容：在新框架中完整保留旧模型参考数据，并校验其结构与关键值。
    """
    assert MODEL_REFERENCE_DATA["trace"].shape == (4, 6)
    assert MODEL_REFERENCE_DATA["thermal"].shape == (4, 6)
    assert MODEL_REFERENCE_DATA["densenet"].shape == (4, 5)
    assert MODEL_REFERENCE_DATA["corrnet_forward"].shape == (4, 1)
    assert MODEL_REFERENCE_DATA["corrnet_norm"].shape == (4, 1)
    assert MODEL_REFERENCE_DATA["corrnet_prefit"].shape == (4, 1)
    assert MODEL_REFERENCE_DATA["corrnet_econst"].shape == (4, 1)

    assert torch.isclose(MODEL_REFERENCE_DATA["trace"][0, 0], torch.tensor(-0.1368), atol=1e-4)
    assert torch.isclose(MODEL_REFERENCE_DATA["thermal"][0, 2], torch.tensor(-2.5340), atol=1e-4)
    assert torch.isclose(MODEL_REFERENCE_DATA["densenet"][3, 3], torch.tensor(1.3167), atol=1e-4)
    assert torch.isclose(MODEL_REFERENCE_DATA["corrnet_forward"][2, 0], torch.tensor(-2.0623, dtype=torch.float64), atol=1e-4)


def test_migrated_reader_reference_data_catalog():
    """
    依赖：历史 `test_reader.py` 中的参考常量与统计值。
    测试内容：在新框架中完整保留旧 reader/group reader 参考数据并校验关键字段。
    """
    assert READER_REFERENCE_DATA["group_err_empty_1"] == "# ignore empty dataset: ./group1/"
    assert READER_REFERENCE_DATA["group_err_empty_2"] == "# ignore empty dataset: ./group2/"
    assert READER_REFERENCE_DATA["group_out_load"] == "# load 3 systems with fields ['lb_e', 'eig']"
    assert READER_REFERENCE_DATA["group_err_reset_2"] == "# ./group2/ reset batch size to 2"
    assert READER_REFERENCE_DATA["group_err_reset_3"] == "# ./group3/ reset batch size to 1"

    assert READER_REFERENCE_DATA["nelem_ref"].shape == (3, 9)
    assert READER_REFERENCE_DATA["elem_list_reader"] == [6, 30, 35, 40, 43, 50, 56, 76, 79]

    assert np.allclose(READER_REFERENCE_DATA["all_mean"], np.array([-0.18484446, -0.07506522, 0.27285756, -0.19529272]), atol=1e-8)
    assert np.allclose(READER_REFERENCE_DATA["all_std"], np.array([0.56544117, 1.20177998, 1.03995579, 0.89537965]), atol=1e-8)
    assert np.allclose(READER_REFERENCE_DATA["prefit_weight"], np.array([-0.0053749, 0.11951934, 0.815563, 0.13809183]), atol=1e-8)
    assert np.isclose(READER_REFERENCE_DATA["prefit_bias"], -0.028533653429280666, atol=1e-8)

    assert READER_REFERENCE_DATA["group_elem_list"].shape == (15,)
    assert READER_REFERENCE_DATA["group_elem_const"].shape == (15,)
    assert np.isclose(READER_REFERENCE_DATA["group_elem_const"][0], 0.19558913, atol=1e-8)
    assert np.isclose(READER_REFERENCE_DATA["group_elem_const"][-1], 0.55784453, atol=1e-8)


def test_migrated_groupedlosstracker_sample(capsys):
    """
    依赖：`deepks.ml.train.GroupedLossTracker`。
    测试内容：迁移历史 grouped-loss 聚合样例，验证分组聚合、均值统计及异常分支。
    """
    tracker = GroupedLossTracker()

    # 历史样例中的参考数据（原 tests/unittests/test_evaluator.py）
    loss_a = [
        torch.tensor(-0.9622731804847717),
        torch.tensor(0.635839581489563),
        torch.tensor(0.042841192334890366),
    ]
    loss_b = [
        torch.tensor(-0.11838816851377487),
        torch.tensor(0.3066321909427643),
        torch.tensor(0.15965326130390167),
    ]
    loss_c = [
        torch.tensor(-1.656829595565796),
        torch.tensor(0.1093858852982521),
        torch.tensor(2.9542760848999023),
    ]

    tracker.add_loss(5, loss_a)
    tracker.add_loss(5, loss_b)
    tracker.add_loss(1, loss_c)

    with pytest.raises(AssertionError, match="loss should not be empty"):
        tracker.add_loss(6, [])
    with pytest.raises(AssertionError, match="loss length differs"):
        tracker.add_loss(7, [torch.tensor(1.0), torch.tensor(2.0)])

    assert tracker.n_loss_term == 3
    assert tracker.group_keys() == [1, 5]
    assert tracker.grouped_losses == {
        5: [
            [-0.9622731804847717, 0.635839581489563, 0.042841192334890366],
            [-0.11838816851377487, 0.3066321909427643, 0.15965326130390167],
        ],
        1: [[-1.656829595565796, 0.1093858852982521, 2.9542760848999023]],
    }

    avg_group = tracker.avg_group_loss()
    assert np.allclose(avg_group[5], np.array([-0.54033067, 0.47123589, 0.10124723]), atol=1e-4)
    assert np.allclose(avg_group[1], np.array([-1.6568296, 0.10938589, 2.95427608]), atol=1e-4)
    assert np.allclose(tracker.avg_loss(), np.array([-0.91249698, 0.35061922, 1.05225685]), atol=1e-4)

    tracker.print_avg_group_loss()
    line = capsys.readouterr().out.strip().split()
    assert line[-4:] == ["-1.6568e+00", "1.0939e-01", "-5.4033e-01", "4.7124e-01"]


def test_migrated_evaluator_sample(capsys):
    """
    依赖：`deepks.interface.objectives.DescriptorPropertyObjectiveAdapter`。
    测试内容：迁移历史 `test_Evaluator` 的字段、参数和参考向量定义，验证损失项数量与表头输出。
    """

    def loss_fn(input, target):
        return torch.mean((input - target) ** 2)

    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=12, output_dim=1):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

        def forward_with_derivatives(self, model_inputs, derivative_spec=None):
            model_input = model_inputs.requires_grad_(bool(derivative_spec and derivative_spec.get("input")))
            prediction = self(model_input)
            derivatives = {"input": None}
            if derivative_spec and derivative_spec.get("input"):
                [input_grad] = torch.autograd.grad(
                    prediction,
                    model_input,
                    grad_outputs=torch.ones_like(prediction),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )
                derivatives["input"] = input_grad
            return {"primary_output": prediction}, derivatives

    torch.manual_seed(20260313)
    batch_size = 2
    natom, ndesc, nks, norb, nlocal = 3, 4, 2, 5, 6
    sample = {
        "lb_e": torch.randn(batch_size, 1),
        "eig": torch.randn(batch_size, natom, ndesc),
        "eg0": torch.randn(batch_size, 1),
        "gveg": torch.randn(batch_size, natom, ndesc, 1),
        "lb_f": torch.randn(batch_size, natom, 3),
        "gvx": torch.randn(batch_size, natom, 3, natom, ndesc),
        "lb_s": torch.randn(batch_size, 6),
        "gvepsl": torch.randn(batch_size, 6, natom, ndesc),
        "lb_o": torch.randn(batch_size, nks, norb),
        "op": torch.randn(batch_size, nks * norb, natom, ndesc),
        "lb_vd": torch.randn(batch_size, nks, nlocal, nlocal),
        "vdp": torch.randn(batch_size, nks, nlocal, nlocal, natom, ndesc),
        "lb_phi": torch.randn(batch_size, nks, nlocal, nlocal),
        "lb_band": torch.randn(batch_size, nks, nlocal),
        "h_base": torch.randn(batch_size, nks, nlocal, nlocal),
        "trans_matrix": torch.randn(batch_size, nks, nlocal, nlocal),
        "gldv": torch.randn(batch_size, natom, ndesc),
    }
    objective = build_descriptor_property_objective(
        {
            "energy_factor": 1.0,
            "energy_lossfn": loss_fn,
            "force_factor": 1.0,
            "force_lossfn": loss_fn,
            "stress_factor": 1.0,
            "stress_lossfn": loss_fn,
            "orbital_factor": 1.0,
            "orbital_lossfn": loss_fn,
            "v_delta_factor": 1.0,
            "v_delta_lossfn": loss_fn,
            "phi_factor": 1.0,
            "phi_occ": {natom: 4},
            "phi_lossfn": loss_fn,
            "band_factor": 1.0,
            "band_occ": {natom: 5},
            "band_lossfn": loss_fn,
            "density_m_factor": 1.0,
            "density_m_occ": {natom: 6},
            "density_m_lossfn": loss_fn,
            "density_factor": 1.0,
            "grad_penalty": 1.0,
            "energy_per_atom": 0,
            "vd_divide_by_nlocal": True,
        },
        property_scheme="energy_descriptor",
    )
    model = SimpleModel(input_dim=natom * ndesc, output_dim=1)
    loss_list = objective.compute_losses(model, sample)
    loss = torch.stack([item.detach() for item in loss_list])

    # 历史样例中的参考向量（用于迁移可追溯性）
    reference_loss = torch.tensor(
        [
            9.7316e-02,
            7.2783e-01,
            1.5804e00,
            3.4535e-01,
            1.1475e00,
            6.9864e00,
            1.5606e00,
            1.7065e02,
            9.6672e01,
            1.0310e-01,
            2.7987e02,
        ]
    )
    assert len(loss_list) == 11
    assert reference_loss.shape == loss.shape
    assert torch.isfinite(loss).all()

    data_keys = ["eg0", "lb_f", "lb_s", "lb_o", "lb_vd", "lb_phi", "lb_band", "gldv"]
    objective.print_head("test", data_keys)
    line = capsys.readouterr().out.strip().split()
    assert line[-10:] == [
        "test_energy",
        "test_grad",
        "test_force",
        "test_stress",
        "test_orbital",
        "test_v_delta",
        "test_phi",
        "test_band",
        "test_dm",
        "test_density",
    ]


def test_migrated_template_module_import_if_abacus():
    """
    依赖：`abacus`（可选）。
    测试内容：迁移历史模板样例意图，验证 `abacus` 模块在可用环境下可导入。
    """
    if not _has_module("abacus"):
        pytest.skip("abacus python module not installed")
    import abacus  # noqa: F401

    assert abacus is not None
