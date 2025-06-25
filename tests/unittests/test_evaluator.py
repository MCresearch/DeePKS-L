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
    evaluator = Evaluator(energy_factor=1., force_factor=1., stress_factor=1., 
                          orbital_factor=1., v_delta_factor=1.,
                          phi_factor=1., phi_occ={3:4, 6:8}, band_factor=1.,band_occ={3:8, 6:16},
                          density_m_factor=1.,density_m_occ=0, density_factor=1.,
                          grad_penalty=1., energy_per_atom=0, vd_divide_by_nlocal=True)
    # test loss calculation
    # TO BE IMPLEMENTED
    # test print_head
    data_keys = ['eg0', 'lb_f', 'lb_s', 'lb_o', 'lb_vd', 'lb_phi', 'lb_band', 'gldv']
    evaluator.print_head("test", data_keys)
    captured = capsys.readouterr()
    line = captured.out.strip().split()
    assert line[-10:] == ['test_energy', 'test_grad', 'test_force', 'test_stress',
                          'test_bandgap', 'test_v_delta', 'test_phi', 'test_band',
                          'test_dm', 'test_density']
