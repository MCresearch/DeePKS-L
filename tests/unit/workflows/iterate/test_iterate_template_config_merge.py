"""Regression tests for iterate template config materialization boundaries."""

import io

from ruamel.yaml import YAML

from deepks.workflows.iterate.template import make_run_scf, make_train_task


def _load_yaml_text(content):
    return YAML(typ='safe', pure=True).load(io.StringIO(content))


def test_make_train_task_merges_runtime_paths_into_finalized_task_config():
    task = make_train_task(
        task_config={
            'seed': 12345678,
            'device': 'cuda:0',
            'train_args': {'n_epoch': 2, 'start_lr': 1e-4},
            'data_args': {'batch_size': 2},
        },
        data_train='data_train',
        data_test='data_test',
        save_model='model.pth',
    )

    generated = _load_yaml_text(task.write_files['train_input.yaml'])

    assert generated['seed'] == 12345678
    assert generated['train_args']['n_epoch'] == 2
    assert generated['train_args']['start_lr'] == 1e-4
    assert generated['data_args']['batch_size'] == 2
    assert generated['device'] == 'cuda:0'
    assert generated['systems_train'] == 'data_train/*'
    assert generated['systems_test'] == 'data_test/*'


def test_make_run_scf_merges_runtime_systems_into_finalized_task_config(tmp_path):
    train_sys = tmp_path / "sys.train"
    test_sys = tmp_path / "sys.test"
    train_sys.mkdir()
    test_sys.mkdir()

    task = make_run_scf(
        systems_train=[str(train_sys)],
        systems_test=[str(test_sys)],
        task_config={
            'basis': 'sto-3g',
            'scf_args': {'max_cycle': 3},
            'mol_args': {'charge': 1},
            'device': 'cuda:1',
        },
        sub_size=1,
        group_size=1,
        ingroup_parallel=1,
    )

    generated = _load_yaml_text(task.batch_tasks[0].write_files['_scf_task.yaml'])

    assert generated['basis'] == 'sto-3g'
    assert generated['scf_args']['max_cycle'] == 3
    assert generated['mol_args']['charge'] == 1
    assert generated['device'] == 'cuda:1'
    assert generated['type'] == 'scf'
    assert generated['scf_soft'] == 'pyscf'
