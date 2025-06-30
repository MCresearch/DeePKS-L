import os
import shutil
import pytest
import numpy as np
import torch
from deepks.model.reader import Reader, GroupReader, SimpleReader

def test_SimpleReader(capsys):
    # Create files for testing
    data_size = 3
    natom, nproj = 2, 4
    meta = np.array([natom, 123, nproj])
    l_e_delta = np.random.randn(data_size, 1)
    dm_eig = np.random.randn(data_size, natom, nproj)
    conv = np.array([True, True, False])
    np.savetxt("system.raw", meta)
    np.save("l_e_delta.npy", l_e_delta)
    np.save("dm_eig.npy", dm_eig)
    np.save("conv.npy", conv)
    # Test load_meta and prepare using __init__
    # Test case 1: with system.raw and conv.npy
    reader = SimpleReader("./", batch_size=3)
    assert reader.get_batch_size() == 2
    assert reader.natm == natom
    assert reader.nproj == nproj
    assert reader.get_nframes() == 2
    assert (reader.data_ec == l_e_delta[:2]).all()
    assert (reader.data_dm == dm_eig[:2]).all()
    capture = capsys.readouterr()
    assert capture.err.strip() == "# ./ reset batch size to 2"
    del reader
    # Clean up test files
    os.remove("system.raw")
    os.remove("conv.npy")
    # Test case 2: without system.raw and conv.npy
    reader = SimpleReader("./", batch_size=1, d_name=["dm_eig"], conv_filter=False)
    assert reader.get_batch_size() == 1
    assert reader.natm == natom
    assert reader.nproj == nproj
    assert reader.get_nframes() == 3
    assert (reader.data_ec == l_e_delta).all()
    assert (reader.data_dm == dm_eig).all()
    capture = capsys.readouterr()
    assert capture.err.strip() == "# ./ no system.raw, infer meta from data"
    # Test sampling
    for idx in range(3):
        sample = reader.sample_train()
        assert sample['lb_e'].shape == (1, 1)
        assert sample['lb_e'].item() == l_e_delta[idx]
        assert sample['eig'].shape == (1, natom, nproj)
        assert (sample['eig'] == dm_eig[idx:idx+1]).all()
    sample_all = reader.sample_all()
    assert sample_all['lb_e'].shape == (3, 1)
    assert (sample_all['lb_e'] == l_e_delta).all()
    assert sample_all['eig'].shape == (3, natom, nproj)
    assert (sample_all['eig'] == dm_eig).all()
    # Remove test files
    os.remove("l_e_delta.npy")
    os.remove("dm_eig.npy")

def test_Reader(capsys):
    # Create files for testing
    data_size = 3
    natom, ndesc, nks, norb, nlocal = 3, 4, 2, 5, 6
    meta = np.array([natom, 123, ndesc])
    np.savetxt("system.raw", meta)
    data_dict = {
        "l_e_delta": ("lb_e", np.random.randn(data_size, 1)),
        "dm_eig": ("eig", np.random.randn(data_size, natom, ndesc)),
        "l_f_delta": ("lb_f", np.random.randn(data_size, natom, 3)),
        "grad_vx": ("gvx", np.random.randn(data_size, natom, 3, natom, ndesc)),
        "l_s_delta": ("lb_s", np.random.randn(data_size, 6)),
        "grad_vepsl": ("gvepsl", np.random.randn(data_size, 6, natom, ndesc)),
        "l_o_delta": ("lb_o", np.random.randn(data_size, nks, norb)),
        "orbital_precalc": ("op", np.random.randn(data_size, nks*norb, natom, ndesc)),
        "l_h_delta": ("lb_vd", np.random.randn(data_size, nks, nlocal, nlocal)),
        "v_delta_precalc": ("vdp", np.random.randn(data_size, nks, nlocal, nlocal, natom, ndesc)),
        "h_base": ("h_base", np.random.randn(data_size, nks, nlocal, nlocal)),
        "eg_base": ("eg0", np.random.randn(data_size, 1)),
        "grad_veg": ("gveg", np.random.randn(data_size, natom, ndesc, 1)),
        "grad_ldv": ("gldv", np.random.randn(data_size, natom, ndesc)),
    }
    hamiltonian = np.random.randn(data_size, nks, nlocal, nlocal)
    # make sure overlap is positive definite
    o_tmp = np.random.randn(data_size, nks, nlocal, nlocal)
    overlap = np.matmul(o_tmp, np.transpose(o_tmp, (0,1,3,2))) + 1e-3 * np.eye(nlocal)[None, None, :, :]
    conv = np.array([True, True, False])
    atom = np.random.randint(1, 100, (data_size, natom, 4))
    for key, value in data_dict.items():
        np.save(f"{key}.npy", value[1])
    np.save("hamiltonian.npy", hamiltonian)
    np.save("overlap.npy", overlap)
    np.save("conv.npy", conv)
    np.save("atom.npy", atom)
    # Test check_exist, load_meta and prepare using __init__
    # Test case 1: with system.raw and conv.npy
    reader = Reader("./", batch_size=3, read_overlap=True)
    assert reader.get_batch_size() == 2
    assert reader.natm == natom
    assert reader.nproj == ndesc
    assert reader.get_nframes() == 2
    assert (reader.data_ec == data_dict['l_e_delta'][1][:2]).all()
    assert (reader.data_dm == data_dict['dm_eig'][1][:2]).all()
    capture = capsys.readouterr()
    assert capture.err.strip() == "# ./ reset batch size to 2"
    for key, value in data_dict.items():
        assert (reader.t_data[value[0]] == value[1][:2]).all(), f"Mismatch in {key}"
    assert reader.t_data["L_inv"].shape == (2, nks, nlocal, nlocal)
    assert reader.t_data["lb_band"].shape == (2, nks, nlocal)
    assert reader.t_data["lb_phi"].shape == (2, nks, nlocal, nlocal)
    assert (reader.atom_info["elems"] == atom[:2, :, 0]).all()
    assert (reader.atom_info["coords"] == atom[:2, :, 1:]).all()
    del reader
    # Clean up test files
    os.remove("system.raw")
    os.remove("hamiltonian.npy")
    os.remove("overlap.npy")
    os.remove("conv.npy")
    # Test case 2: without system.raw, conv.npy and read_overlap
    reader = Reader("./", batch_size=1)
    assert reader.get_batch_size() == 1
    assert reader.natm == natom
    assert reader.nproj == ndesc
    assert reader.get_nframes() == 3
    assert (reader.data_ec == data_dict['l_e_delta'][1]).all()
    assert (reader.data_dm == data_dict['dm_eig'][1]).all()
    capture = capsys.readouterr()
    assert capture.err.strip() == "# ./ no system.raw, infer meta from data"
    for key, value in data_dict.items():
        assert (reader.t_data[value[0]] == value[1]).all(), f"Mismatch in {key}"
    assert (reader.atom_info["elems"] == atom[:, :, 0]).all()
    assert (reader.atom_info["coords"] == atom[:, :, 1:]).all()
    # Test sampling
    index_list = [0, 1, 2] # Fix the shuffle order for testing
    for idx in index_list:
        sample = reader.sample_train(index_list)
        for key, value in data_dict.items():
            assert sample[value[0]].shape == value[1][idx:idx+1].shape, f"Mismatch in {key} shape"
            assert (sample[value[0]] == value[1][idx:idx+1]).all(), f"Mismatch in {key} values"
    sample_all = reader.sample_all()
    for key, value in data_dict.items():
        assert sample_all[value[0]].shape == value[1].shape, f"Mismatch in {key} shape"
        assert (sample_all[value[0]] == value[1]).all(), f"Mismatch in {key} values"
    # Test element table
    elem_list = sorted(set(atom[:, :, 0].flatten()))
    elem_const = np.random.randn(len(elem_list))
    nelem = reader.collect_elems(elem_list)
    nelem_ref = [[1, 0, 1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 0, 0, 1, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 1, 1, 0, 0]]
    assert nelem.shape == (data_size, len(elem_list))
    assert (nelem == nelem_ref).all()
    assert reader.atom_info["elem_list"] == [6, 30, 35, 40, 43, 50, 56, 76, 79]
    reader.subtract_elem_const(elem_const)
    assert (reader.atom_info["elem_const"] == elem_const).all()
    e_diff = (nelem_ref @ elem_const).reshape(-1, 1)
    assert (reader.data_ec == data_dict['l_e_delta'][1] - e_diff).all()
    assert (reader.t_data["lb_e"] == data_dict['l_e_delta'][1] - e_diff).all()
    reader.revert_elem_const()
    assert (reader.data_ec == data_dict['l_e_delta'][1]).all()
    assert (reader.data_dm == data_dict['dm_eig'][1]).all()
    assert "elem_const" not in reader.atom_info
    # Remove test files
    os.remove("atom.npy")
    for key in data_dict.keys():
        os.remove(f"{key}.npy")

def test_GroupReader(capsys):
    # Create files for testing
    data_size = [3, 2, 1]
    natom = [2, 3, 3]
    nproj = 4
    meta, l_e_delta, dm_eig = [], [], []
    for i in range(3):
        os.makedirs(f"./group{i+1}", exist_ok=True)
        meta_i = np.array([natom[i], 123 + i, nproj])
        l_e_delta_i = np.random.randn(data_size[i], 1)
        dm_eig_i = np.random.randn(data_size[i], natom[i], nproj)
        conv_i = np.array([False] * data_size[i])
        atom_i = np.random.randint(1, 100, (data_size[i], natom[i], 4))
        np.savetxt(f"./group{i+1}/system.raw", meta_i)
        np.save(f"./group{i+1}/l_e_delta.npy", l_e_delta_i)
        np.save(f"./group{i+1}/dm_eig.npy", dm_eig_i)
        np.save(f"./group{i+1}/conv.npy", conv_i)
        np.save(f"./group{i+1}/atom.npy", atom_i)
        meta.append(meta_i)
        l_e_delta.append(l_e_delta_i)
        dm_eig.append(dm_eig_i)
    # Test empty case
    with pytest.raises(RuntimeError, match="No system is avaliable"):
        gr = GroupReader(["./group1/", "./group2/"], batch_size=3)
    capture = capsys.readouterr()
    assert capture.err.strip().splitlines()[-3] == '# ignore empty dataset: ./group1/'
    assert capture.err.strip().splitlines()[-1] == '# ignore empty dataset: ./group2/'
    os.remove("group1/conv.npy")
    os.remove("group2/conv.npy")
    os.remove("group3/conv.npy")
    # Test case with valid data
    gr = GroupReader(["./group1/", "./group2/", "./group3/"], batch_size=3, group_batch=2)
    capture = capsys.readouterr()
    assert capture.out.strip().splitlines()[0] == '# load 3 systems with fields [\'lb_e\', \'eig\']'
    assert capture.err.strip().splitlines()[0] == '# ./group2/ reset batch size to 2'
    assert capture.err.strip().splitlines()[1] == '# ./group3/ reset batch size to 1'
    assert gr.ndesc == nproj
    assert gr.get_batch_size() == 3
    assert gr.get_train_size() == 6
    assert (gr.sys_prob == [1/2, 1/3, 1/6]).all()
    assert gr.group_batch == 2
    assert list(gr.group_dict.keys()) == [(2, None), (3, None)]
    for value in gr.group_dict.values():
        for reader in value:
            assert isinstance(reader, Reader)
    assert gr.group_prob == {(2, None): 0.5, (3, None): 0.5}
    batch_prop_ref = {(2, None): np.array([1]), (3, None): np.array([0.5, 0.5])}
    for k in batch_prop_ref:
        assert k in gr.batch_prob
        assert (gr.batch_prob[k] == batch_prop_ref[k]).all()
    # Test sampling
    # test sample_idx and sample_train
    np.random.seed(42)
    assert gr.sample_idx() == 0 # sample from random choosed group (group 1 for test)
    sample = gr.sample_train(idx = 0, index_list = [0, 1, 2])  # Sample from group 1
    assert sample['lb_e'].shape == (data_size[0], 1)
    assert (sample['lb_e'] == l_e_delta[0]).all()
    assert sample['eig'].shape == (data_size[0], natom[0], nproj)
    assert (sample['eig'] == dm_eig[0]).all()
    # test sample_all
    sample = gr.sample_all(idx = 1)
    assert sample['lb_e'].shape == (data_size[1], 1)
    assert (sample['lb_e'] == l_e_delta[1]).all()
    assert sample['eig'].shape == (data_size[1], natom[1], nproj)
    assert (sample['eig'] == dm_eig[1]).all()
    # test sample_train_group
    np.random.seed(45)
    batch = gr.sample_train_group()
    assert batch['lb_e'].shape == (data_size[1] + data_size[2], 1)
    assert (batch['lb_e'][:1] == l_e_delta[2]).all()
    assert (batch['lb_e'][1:] == l_e_delta[1]).all()
    assert batch['eig'].shape == (data_size[1] + data_size[2], natom[2], nproj)
    assert (batch['eig'][:1] == dm_eig[2]).all()
    assert (batch['eig'][1:] == dm_eig[1]).all()
    # test sample_all_batch
    generator = gr.sample_all_batch()
    for i in range(3):
        batch = next(generator)
        assert batch['lb_e'].shape == (data_size[i], 1)
        assert (batch['lb_e'] == l_e_delta[i]).all()
        assert batch['eig'].shape == (data_size[i], natom[i], nproj)
        assert (batch['eig'] == dm_eig[i]).all()
    with pytest.raises(StopIteration):
        next(generator)
    # Test compute_data_stat
    all_mean, all_std = gr.compute_data_stat()
    assert np.allclose(all_mean, np.array([-0.18484446, -0.07506522,  0.27285756, -0.19529272]), atol=1e-8)
    assert np.allclose(all_std, np.array([0.56544117, 1.20177998, 1.03995579, 0.89537965]), atol=1e-8)
    # Test compute_prefitting
    weight, bias = gr.compute_prefitting()
    assert np.allclose(weight, np.array([-0.0053749, 0.11951934, 0.815563, 0.13809183]), atol=1e-8)
    assert np.isclose(bias, -0.028533653429280666, atol=1e-8)
    # Test element table
    elem_list, elem_const = gr.compute_elem_const()
    assert (elem_list == [3, 6, 8, 10, 25, 26, 27, 29, 32, 44, 49, 51, 60, 79, 89]).all()
    assert np.allclose(elem_const, np.array([0.19558913, 0.1471279, -0.37585689, 0.19558913,
                                             -0.37585689, 0.1471279, -0.09742901, -0.37585689,
                                             -0.09742901, -0.28587458, 0.55784453, 0.19558913,
                                             -0.28587458, -0.09742901, 0.55784453]), atol=1e-8)
    gr.subtract_elem_const(elem_const)
    e_diff = np.dot(gr.group_dict[(2, None)][0].collect_elems(elem_list), elem_const.reshape(-1, 1))
    assert (gr.readers[0].data_ec == l_e_delta[0] - e_diff).all()
    assert (gr.readers[0].t_data["lb_e"] == l_e_delta[0] - e_diff).all()
    gr.revert_elem_const()
    assert (gr.readers[0].data_ec == l_e_delta[0]).all()
    assert (gr.readers[0].data_dm == dm_eig[0]).all()
    assert "elem_const" not in gr.readers[0].atom_info
    # Read data from
    shutil.rmtree("./group1")
    shutil.rmtree("./group2")
    shutil.rmtree("./group3")