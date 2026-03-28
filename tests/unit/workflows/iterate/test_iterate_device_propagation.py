"""Tests for global device propagation through iterate-generated task inputs."""

import io

from ruamel.yaml import YAML

from deepks.workflows.iterate.prepare import prepare_iterate


def _load_yaml_text(content):
    return YAML(typ='safe', pure=True).load(io.StringIO(content))


def test_prepare_iterate_injects_global_device_into_generated_task_yaml(tmp_path):
    """Top-level device should flow into iterate SCF/train child task configs."""
    train_sys = tmp_path / "sys.train"
    test_sys = tmp_path / "sys.test"
    workdir = tmp_path / "work"

    train_sys.mkdir()
    test_sys.mkdir()

    config = {
        'type': 'iterate',
        'scf_soft': 'pyscf',
        'systems_train': [str(train_sys)],
        'systems_test': [str(test_sys)],
        'n_iter': 1,
        'workdir': str(workdir),
        'share_folder': 'share',
        'device': 'cuda:1',
        'scf_input': {'basis': 'sto-3g'},
        'train_input': {'train_args': {'n_epoch': 1}},
    }

    workflow, _, _ = prepare_iterate(config)

    main_iter = workflow[0]
    scf_yaml = main_iter[0][0].batch_tasks[0].write_files['_scf_task.yaml']
    train_yaml = main_iter[1][0].write_files['train_input.yaml']
    scf_snapshot = _load_yaml_text((workdir / 'share' / 'scf_input.yaml').read_text(encoding='utf-8'))
    train_snapshot = _load_yaml_text((workdir / 'share' / 'train_input.yaml').read_text(encoding='utf-8'))

    assert _load_yaml_text(scf_yaml)['device'] == 'cuda:1'
    assert _load_yaml_text(train_yaml)['device'] == 'cuda:1'
    assert scf_snapshot['device'] == 'cuda:1'
    assert train_snapshot['device'] == 'cuda:1'
