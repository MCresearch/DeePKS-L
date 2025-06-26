import os
import pytest

import torch
import numpy as np
from deepks.model.evaluator import Evaluator, NatomLossList

def test_NatomLossList(capsys):
    nll = NatomLossList()
    # test add_loss
    loss_list = [[torch.randn(1) for _ in range(3)] for _ in range(3)]
    loss_list_float = [[loss_list[i][j].item() for j in range(3)] for i in range(3)]
    nll.add_loss(5, loss_list[0])
    nll.add_loss(5, loss_list[1])
    nll.add_loss(1, loss_list[2])
    with pytest.raises(AssertionError, match="loss should not be empty"):
        nll.add_loss(6, [])
    with pytest.raises(AssertionError, match="loss length are different for newly added natom 7, expected 3, got 2"):
        nll.add_loss(7, [torch.randn(1) for _ in range(2)])
    assert nll.n_loss_term == 3
    assert nll.natom_loss_list == {
        5: [[-0.9622731804847717, 0.635839581489563, 0.042841192334890366],
            [-0.11838816851377487, 0.3066321909427643, 0.15965326130390167]],
        1: [[-1.656829595565796, 0.1093858852982521, 2.9542760848999023]]
    }
    # test natoms
    assert nll.natoms() == [1, 5]
    # test avg_atom_loss
    avg_loss = nll.avg_atom_loss()
    assert (avg_loss[5] - np.array([-0.54033067,  0.47123589,  0.10124723]) < 1e-4).all()
    assert (avg_loss[1] - np.array([-1.6568296 ,  0.10938589,  2.95427608]) < 1e-4).all()
    # test avg_loss
    assert (nll.avg_loss() - np.array([-0.91249698, 0.35061922, 1.05225685]) < 1e-4).all()
    # test print_avg_atom_loss
    nll.print_avg_atom_loss()
    captured = capsys.readouterr()
    line = captured.out.strip().split()
    assert line[-4:] == ['-1.6568e+00', '1.0939e-01', '-5.4033e-01', '4.7124e-01']
    
def test_Evaluator(capsys):
    # test loss calculation
    # make sample loss function and model
    def loss_fn(input, target):
        return torch.mean((input - target) ** 2)
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=12, output_dim=1):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
    # make sample data
    batch_size = 2
    natom, ndesc, nks, norb, nlocal = 3, 4, 2, 5, 6
    sample = {
        'lb_e': torch.randn(batch_size, 1), # model output
        'eig': torch.randn(batch_size, natom, ndesc), # model input
        # for energy gradient
        'eg0': torch.randn(batch_size, 1),
        'gveg': torch.randn(batch_size, natom, ndesc, 1),
        # for force
        'lb_f': torch.randn(batch_size, natom, 3),
        'gvx': torch.randn(batch_size, natom, 3, natom, ndesc),
        # for stress
        'lb_s': torch.randn(batch_size, 6),
        'gvepsl': torch.randn(batch_size, 6, natom, ndesc),
        # for orbital
        'lb_o': torch.randn(batch_size, nks, norb),
        'op': torch.randn(batch_size, nks*norb, natom, ndesc),
        # for v_delta
        'lb_vd': torch.randn(batch_size, nks, nlocal, nlocal),
        'vdp': torch.randn(batch_size, nks, nlocal, nlocal, natom, ndesc),
        # for phi and band
        'lb_phi': torch.randn(batch_size, nks, nlocal, nlocal),
        'lb_band': torch.randn(batch_size, nks, nlocal),
        'h_base': torch.randn(batch_size, nks, nlocal, nlocal),
        'L_inv': torch.randn(batch_size, nks, nlocal, nlocal),
        # for density
        'gldv': torch.randn(batch_size, natom, ndesc),
    }
    evaluator = Evaluator(energy_factor=1., energy_lossfn=loss_fn,
                          force_factor=1., force_lossfn=loss_fn,
                          stress_factor=1., stress_lossfn=loss_fn,
                          orbital_factor=1., orbital_lossfn=loss_fn,
                          v_delta_factor=1., v_delta_lossfn=loss_fn,
                          phi_factor=1., phi_occ={natom:4}, phi_lossfn=loss_fn,
                          band_factor=1., band_occ={natom:5}, band_lossfn=loss_fn,
                          density_m_factor=1., density_m_occ={natom:6}, density_m_lossfn=loss_fn,
                          density_factor=1., grad_penalty=1.,
                          energy_per_atom=0, vd_divide_by_nlocal=True)
    model = SimpleModel(input_dim=natom*ndesc, output_dim=1)
    loss_list = evaluator(model, sample)
    loss = torch.tensor(loss_list)
    ref = torch.tensor([9.7316e-02, 7.2783e-01, 1.5804e+00, 3.4535e-01, 1.1475e+00,
                        6.9864e+00, 1.5606e+00, 1.7065e+02, 9.6672e+01, 1.0310e-01,
                        2.7987e+02])
    assert len(loss_list) == 11
    assert torch.allclose(loss, ref, atol=1e-4)  # total loss
    # test print_head
    data_keys = ['eg0', 'lb_f', 'lb_s', 'lb_o', 'lb_vd', 'lb_phi', 'lb_band', 'gldv']
    evaluator.print_head("test", data_keys)
    captured = capsys.readouterr()
    line = captured.out.strip().split()
    assert line[-10:] == ['test_energy', 'test_grad', 'test_force', 'test_stress',
                          'test_bandgap', 'test_v_delta', 'test_phi', 'test_band',
                          'test_dm', 'test_density']
