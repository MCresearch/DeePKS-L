import os
import pytest
import numpy as np
import torch
from deepks.utils import get_shell_sec
from deepks.model.model import TraceEmbedding, ThermalEmbedding, DenseNet, CorrNet

basis = [[0,[0,0,0]], # s
         [1,[0,0,0]], # p
         [2,[0,0,0]]] # d
shell_sec = get_shell_sec(basis)
elem_table = ([1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Example element table

def test_get_shell_sec():
    assert shell_sec == [1,1,3,3,5,5]

def test_TraceEmbedding():
    emb = TraceEmbedding(shell_sec)
    x = torch.randn(4, 18) # 18 = 2 * s + 2 * p + 2 * d
    y = emb(x)
    ref = torch.tensor([
        [-0.1368, -0.4975,  0.7234, -0.0998,  6.4928, -0.3836],
        [ 0.1646,  0.9874, -0.7656,  0.8742, -4.1812,  1.0911],
        [-0.7216, -0.0979,  2.4200,  0.5760,  2.2466, -1.7258],
        [ 0.6411, -0.1408, -1.2673, -1.2313,  1.4721,  2.5969]
    ])
    assert y.shape == (4, 6)  # 6 = 2 * 3 (s, p ,d)
    assert torch.allclose(y, ref, atol=1e-4)

def test_ThermalEmbedding():
    embd_sizes = [1] * 6  # Embedding size of 1 for each shell
    emb = ThermalEmbedding(shell_sec, embd_sizes)
    x = torch.randn(4, 18)
    y = emb(x)
    ref = torch.tensor([
        [-1.1542, -0.0975, -2.5340, -0.6297,  0.2274, -0.4620],
        [-1.3175, -1.1912, -0.5842, -1.2733, -0.2728, -1.6477],
        [-1.1860, -0.4774, -1.9962, -0.4106, -0.7192, -0.3664],
        [ 0.5418, -0.1053, -0.6545, -0.3955, -2.3581, -0.1855]
    ])
    assert y.shape == (4, sum(embd_sizes))
    assert torch.allclose(y, ref, atol=1e-4)

def test_DenseNet():
    input_size, hidden_sizes, output_size = 16, 64, 5
    sizes = [input_size, hidden_sizes, hidden_sizes, hidden_sizes, output_size]
    model = DenseNet(sizes, with_dt=True, layer_norm=True)
    x = torch.randn(4, input_size)
    y = model(x)
    ref = torch.tensor([
        [ 0.3203,  0.0591,  0.2396,  0.8337,  0.0184],
        [ 0.3563, -0.2009,  0.2498,  0.9055, -0.1204],
        [ 0.7344, -0.8686,  0.3067,  0.0719, -0.1099],
        [-0.1397, -1.2096,  0.4953,  1.3167,  0.1208]
    ])
    assert y.shape == (4, output_size)
    assert torch.allclose(y, ref, atol=1e-4)

def test_CorrNet():
    hidden_sizes = (32,16)
    model = CorrNet(18,
                    hidden_sizes,
                    embedding={"type": "thermal"},
                    layer_norm=True,
                    proj_basis=basis,
                    elem_table=elem_table)
    x = torch.randn(4, 6, 18)
    # test forward
    y = model(x)
    ref = torch.tensor([[-0.8677], [ 0.5022], [-2.0623], [-1.3056]], dtype=torch.float64)
    assert torch.allclose(y, ref, atol=1e-4)

    # test get_elem_const
    assert model.get_elem_const([1, 3, 6]) == 1.0

    # test set_normalization
    model.set_normalization(shift=0.1, scale=2)
    y = model(x)
    ref = torch.tensor([[ 0.1862], [ 0.8503], [-1.0830], [-0.5516]], dtype=torch.float64)
    assert torch.allclose(y, ref, atol=1e-4)

    # test set_prefitting
    model.set_prefitting(weight=0.5, bias=0.1)
    y = model(x)
    ref = torch.tensor([[-2.8196], [-5.9678], [-4.8722], [-3.4918]], dtype=torch.float64)
    assert torch.allclose(y, ref, atol=1e-4)

    # test set_energy_const
    model.set_energy_const(0.5)
    y = model(x)
    ref = torch.tensor([[-2.3466], [-5.4789], [-4.4017], [-3.0000]], dtype=torch.float64)
    assert torch.allclose(y, ref, atol=1e-4)

    # test save_dict and load from dict
    extra_info = {"testA": 1, "testB": 'abc'}
    dump_dict = model.save_dict(**extra_info)
    assert list(dump_dict.keys()) == ['state_dict', 'init_args', 'extra_info']
    assert dump_dict['extra_info'] == extra_info
    assert dump_dict['init_args']['embedding']['type'] == 'thermal'
    load_model = CorrNet.load_dict(dump_dict)
    assert isinstance(load_model, CorrNet)
    y = model(x)
    y_load = load_model(x)
    assert torch.allclose(y, y_load, atol=1e-4)

    # test compile
    jit_model = model.compile()
    y_jit = jit_model(x)
    assert torch.equal(y, y_jit), "Compiled model output does not match original model output"
    assert isinstance(jit_model, torch.jit.ScriptModule)

    # test save and load
    # basic model save and load
    basic_model_path = "test_model.pth"
    model.save(basic_model_path)
    assert os.path.exists(basic_model_path), "Model file was not created"
    load_basic = CorrNet.load(basic_model_path)
    assert isinstance(load_basic, CorrNet)
    y = model(x)
    y_load_basic = load_basic(x)
    assert torch.allclose(y, y_load_basic, atol=1e-4), "Loaded model output does not match original model output"
    os.remove(basic_model_path)  # Clean up the test file
    # JIT model save and load
    jit_model_path = "test_model_jit.pth"
    element_table_path = jit_model_path + ".elemtab"
    model.compile_save(jit_model_path)
    assert os.path.exists(jit_model_path), "JIT model file was not created"
    load_jit = CorrNet.load(jit_model_path)
    assert isinstance(load_jit, torch.jit.ScriptModule)
    y_jit_load = load_jit(x)
    assert torch.allclose(y, y_jit_load, atol=1e-4), "Loaded JIT model output does not match original model output"
    element_table = np.loadtxt(element_table_path, dtype=float)
    assert np.array_equal(element_table[:,0], np.array(elem_table[0]))
    assert np.array_equal(element_table[:,1], np.array(elem_table[1]))
    os.remove(jit_model_path)  # Clean up the test file
    os.remove(element_table_path)  # Clean up the element table file