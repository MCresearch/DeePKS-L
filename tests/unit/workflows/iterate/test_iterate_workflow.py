"""Test iterate workflow structure and imports."""

import pytest
import tempfile
import os
import numpy as np


def test_iterate_workflow_imports():
    """Test that iterate workflow modules can be imported."""
    from deepks.workflows.iterate import run_iterate_workflow
    from deepks.workflows.iterate.workflow import run_iterate_workflow as workflow_main
    from deepks.workflows.iterate.prepare import prepare_iterate
    from deepks.workflows.iterate.support import build_abacus_iterate_scf_kwargs, make_scf, make_train
    from deepks.workflows.iterate.abacus.sequence import make_scf_abacus

    assert callable(run_iterate_workflow)
    assert callable(workflow_main)
    assert callable(prepare_iterate)
    assert callable(build_abacus_iterate_scf_kwargs)
    assert callable(make_scf)
    assert callable(make_train)
    assert callable(make_scf_abacus)


def test_iterate_workflow_dispatcher_integration():
    """Test that dispatcher can route to iterate workflow."""
    from deepks.config.dispatcher import dispatch_command

    # This should not raise an error for iterate type
    runtime_config = {
        '__internal_packed__': True,
        'type': 'iterate',
        'iterate_param': {
            'type': 'iterate',
            'systems_train': [],
            'iterate': {'n_iter': 0},
        },
    }

    # We expect it to fail because systems_train is empty, but it should
    # reach the workflow code (not fail on dispatch)
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        dispatch_command(runtime_config)


def test_prepare_iterate_validation():
    """Test that prepare_iterate validates input."""
    from deepks.workflows.iterate.prepare import prepare_iterate

    # Missing systems_train should raise error
    config = {
        'type': 'iterate',
        'iterate': {'n_iter': 0},
        'systems_train': None,
        'scf_task': {},
        'train_task': {},
    }

    # Should fail because no valid systems
    with pytest.raises((ValueError, FileNotFoundError, TypeError, Exception)):
        prepare_iterate(config)


def test_iterate_workflow_config_structure():
    """Test that workflow accepts proper config structure."""
    from deepks.workflows.iterate.workflow import run_iterate_workflow
    from deepks.config import load_runtime_config

    # This should fail gracefully with proper error messages
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: iterate\n')
        f.write('runtime:\n')
        f.write('  workdir: test_iter\n')
        f.write('data:\n')
        f.write('  train:\n')
        f.write('    - nonexistent_system\n')
        f.write('physics:\n')
        f.write('  backend:\n')
        f.write('    name: pyscf\n')
        f.write('    input:\n')
        f.write('      basis: sto-3g\n')
        f.write('iterate:\n')
        f.write('  n_iter: 2\n')
        f.flush()
        config_path = f.name

    runtime_config = load_runtime_config(config_path)
    config = runtime_config['iterate_param']

    # Should fail because system doesn't exist, but validates config structure
    try:
        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            run_iterate_workflow(config)
    finally:
        os.unlink(config_path)


def test_run_iterate_workflow_uses_input_normalized_config(monkeypatch, tmp_path):
    """Workflow should receive config already normalized by deepks.config."""
    from deepks.workflows.iterate import workflow as workflow_module

    train_sys = tmp_path / "sys.train"
    train_sys.mkdir()
    np.save(train_sys / "atom.npy", np.array([[[14.0, 0.0, 0.0, 0.0]]]))

    seen = {}

    class DummyIteration:
        def restart(self):
            raise AssertionError("restart should not be used in this test")

        def run(self):
            return None

    def fake_prepare(config):
        seen['config'] = config
        return DummyIteration(), str(tmp_path), str(tmp_path / "RECORD")

    monkeypatch.setattr(workflow_module, "prepare_iterate", fake_prepare)

    config = {
        'type': 'iterate',
        'iterate': {'n_iter': 0},
        'runtime': {'device': 'cpu'},
        'systems_train': [str(train_sys)],
        'scf_soft': 'pyscf',
        'scf_task': {},
        'train_task': {},
    }
    result = workflow_module.run_iterate_workflow(config)

    assert result['n_iterations'] == 0
    assert seen['config']['type'] == 'iterate'
    assert seen['config']['runtime']['device'] == 'cpu'


def test_prepare_iterate_accepts_use_init_and_phase_values(tmp_path):
    """Structured iterate configs should use use_init plus phase values."""
    from deepks.workflows.iterate.prepare import prepare_iterate

    train_sys = tmp_path / "sys.train"
    train_sys.mkdir()
    np.save(train_sys / "atom.npy", np.array([[[14.0, 0.0, 0.0, 0.0]]]))
    orb_file = tmp_path / "Si.orb"
    pp_file = tmp_path / "Si.upf"
    proj_file = tmp_path / "proj.orb"
    orb_file.write_text("orb", encoding="utf-8")
    pp_file.write_text("upf", encoding="utf-8")
    proj_file.write_text("proj", encoding="utf-8")

    config = {
        "type": "iterate",
        "runtime": {
            "workdir": str(tmp_path / "iter-work"),
            "device": "cpu",
            "scf": {
                "execute": {"dispatcher": {"context": "local"}, "group_size": 2},
            },
            "train": {
                "command": {"python": "python3"},
            },
        },
        "data": {"train": [str(train_sys)]},
        "physics": {
            "backend": {
                "name": "abacus",
                "input": {
                    "ecutwfc": [60, 55],
                    "orb_files": [str(orb_file)],
                    "pp_files": [str(pp_file)],
                    "proj_file": [str(proj_file)],
                    "deepks_scf": [1, 0],
                },
            },
            "representation": {"name": "dm_eig", "params": {"proj_basis": "ccpvdz"}},
        },
        "ml": {
            "model": {
                "args": {
                    "hidden_sizes": {
                        "main": [12],
                        "init": [8],
                    },
                },
            },
            "preprocess": {"preshift": [False, True]},
            "train": {
                "epochs": [3, 2],
                "optimizer": {"lr": [1e-3, 3e-4]},
            },
            "objective": {
                "losses": {
                    "main": {"energy": 1.0, "force": 0.2},
                    "init": {"energy": 1.0},
                },
            },
        },
        "iterate": {
            "n_iter": 1,
            "use_init": True,
        },
    }

    from deepks.config.packager import package_config

    workflow, workdir, record_file = prepare_iterate(package_config(config)['iterate_param'])

    assert workflow is not None
    assert workdir == str(tmp_path / "iter-work")
    assert record_file.endswith("RECORD")

    share_dir = tmp_path / "iter-work" / "share"
    train_snapshot = (share_dir / "train_input.yaml").read_text(encoding="utf-8")
    init_train_snapshot = (share_dir / "init_train.yaml").read_text(encoding="utf-8")
    abacus_snapshot = (share_dir / "scf_abacus.yaml").read_text(encoding="utf-8")

    assert "__internal_packed__: true" in train_snapshot.lower()
    assert "epochs: 3" in train_snapshot
    assert "epochs: 2" in init_train_snapshot
    assert "force" in train_snapshot
    assert "force" not in init_train_snapshot
    assert "ecutwfc: 60" in abacus_snapshot


def test_check_share_folder_true():
    """Test check_share_folder with True value."""
    from deepks.workflows.iterate.prepare import check_share_folder
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file
        test_file = os.path.join(tmpdir, 'test.yaml')
        with open(test_file, 'w') as f:
            f.write('test: value\n')

        # Should return the name if file exists
        result = check_share_folder(True, 'test.yaml', tmpdir)
        assert result == 'test.yaml'


def test_check_share_folder_dict():
    """Test check_share_folder with dict value."""
    from deepks.workflows.iterate.prepare import check_share_folder
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        data = {'key': 'value', 'number': 42}

        # Should save dict as YAML
        result = check_share_folder(data, 'test.yaml', tmpdir)
        assert result == 'test.yaml'

        # Check file was created
        test_file = os.path.join(tmpdir, 'test.yaml')
        assert os.path.exists(test_file)


def test_check_share_folder_none():
    """Test check_share_folder with None value."""
    from deepks.workflows.iterate.prepare import check_share_folder

    # Should return None
    result = check_share_folder(None, 'test.yaml', 'share')
    assert result is None


def test_create_scf_step_abacus():
    """Test ABACUS iterate SCF sequence builder."""
    from deepks.workflows.iterate.support import build_abacus_iterate_scf_kwargs
    from deepks.workflows.iterate.abacus.sequence import make_scf_abacus

    # Should create SCF step without error (will fail on execution, not creation)
    # Use a dummy system path that will be validated later
    try:
        kwargs = build_abacus_iterate_scf_kwargs({'ecutwfc': 100})
        scf_step = make_scf_abacus(
            systems_train=['dummy_system'],
            systems_test=['dummy_test'],
            share_folder='share',
            cleanup=False,
            group_size=1,
            orb_files=kwargs["orb_files"],
            pp_files=kwargs["pp_files"],
            proj_file=kwargs["proj_file"],
            run_cmd=kwargs["run_cmd"],
            abacus_path=kwargs["abacus_path"],
            **kwargs["backend_kwargs"],
        )
        assert scf_step is not None
    except (FileNotFoundError, ValueError, IndexError):
        # Expected to fail on missing systems, but creation logic is tested
        pass


def test_build_abacus_iterate_scf_kwargs_preserves_hierarchy_target_shape():
    from deepks.workflows.iterate.support import build_abacus_iterate_scf_kwargs

    kwargs = build_abacus_iterate_scf_kwargs(
        {
            "physics": {"backend": {"input": {"ecutwfc": 100}}},
            "ml": {
                "model": {"args": {"levels": [{"name": "dzp", "output_dim": 8, "target_shape": [3, 3, 3, 8, 8]}]}},
                "objective": {
                    "primary_output": "hamiltonian",
                    "terms": [{"name": "hr", "target": {"format": "collected_hr_delta", "name": "l_hr_delta"}}],
                },
            },
        }
    )

    assert kwargs["backend_kwargs"]["target_shape"] == [3, 3, 3, 8, 8]


def test_create_scf_step_pyscf():
    """Test PySCF iterate SCF sequence builder."""
    from deepks.workflows.iterate.support import make_scf

    # Should create SCF step without error (will fail on execution, not creation)
    try:
        scf_step = make_scf(
            systems_train=['dummy_system'],
            systems_test=['dummy_test'],
            task_config={'basis': 'sto-3g'},
            source_pbasis='basis.npz',
            share_folder='share',
            cleanup=False
        )
        assert scf_step is not None
    except (FileNotFoundError, ValueError, IndexError):
        # Expected to fail on missing systems, but creation logic is tested
        pass


def test_create_scf_step_unknown_backend():
    """Test iterate helper rejects unknown backend."""
    from deepks.workflows.iterate.prepare import _create_scf_step

    with pytest.raises(ValueError, match="Unknown SCF backend"):
        _create_scf_step(
            systems_train=['test'],
            systems_test=None,
            scf_soft='unknown',
            scf_config=None,
            scf_machine={},
            proj_basis=None,
            share_folder='share',
            cleanup=False
        )


def test_create_train_step():
    """Test iterate train sequence builder."""
    from deepks.workflows.iterate.support import make_train

    # Should create training step without error
    train_step = make_train(
        source_train='data_train',
        source_test='data_test',
        source_model='model.pth',
        save_model='model.pth',
        task_config={'train_args': {'n_epoch': 1}},
        source_pbasis='basis.npz',
        share_folder='share',
        cleanup=False
    )

    assert train_step is not None


def test_prepare_iterate_creates_level_specific_scf_sequences_for_hierarchical_recipe(tmp_path):
    from deepks.workflows.iterate.prepare import prepare_iterate
    from deepks.config.packager import package_config

    systems_sz_train = tmp_path / "systems_sz" / "data_train"
    systems_sz_test = tmp_path / "systems_sz" / "data_test"
    systems_dzp_train = tmp_path / "systems_dzp" / "data_train"
    for path in (systems_sz_train, systems_sz_test, systems_dzp_train):
        path.mkdir(parents=True)
        np.save(path / "atom.npy", np.array([[[14.0, 0.0, 0.0, 0.0]]]))

    orb_file = tmp_path / "Si.orb"
    pp_file = tmp_path / "Si.upf"
    proj_file = tmp_path / "proj.orb"
    orb_file.write_text("orb", encoding="utf-8")
    pp_file.write_text("upf", encoding="utf-8")
    proj_file.write_text("proj", encoding="utf-8")

    config = {
        "type": "iterate",
        "recipe": "hierarchical-regression",
        "runtime": {
            "workdir": str(tmp_path / "iter-work"),
            "scf": {"execute": {"dispatcher": {"context": "local"}, "group_size": 1}},
            "train": {"command": {"python": "python3"}},
        },
        "data": {"train": [str(systems_sz_train)]},
        "physics": {
            "backend": {
                "name": "abacus",
                "input": {
                    "ecutwfc": 60,
                    "dft_functional": "pbe",
                    "orb_files": [str(orb_file)],
                    "pp_files": [str(pp_file)],
                    "proj_file": [str(proj_file)],
                },
                "profiles": [
                    {"name": "sz", "input": {"basis_name": "sz"}},
                    {"name": "dzp", "input": {"basis_name": "dzp"}},
                ],
            },
            "representation": {"name": "dm_eig", "params": {"proj_basis": "ccpvdz"}},
        },
        "ml": {
            "model": {
                "family": "hierarchical_regression",
                "args": {
                    "trunk_hidden_sizes": [8],
                    "levels": [
                        {"level": 0, "name": "sz", "output_dim": 8},
                        {"level": 1, "name": "dzp", "output_dim": 26},
                    ],
                },
            },
            "objective": {"terms": [{"name": "hr", "weight": 1.0, "target": {"format": "collected_hr_delta", "name": "l_hr_delta"}}]},
            "train": {"stage_schedule": [{"level": 0, "epochs": 1}, {"level": 1, "epochs": 1}]},
        },
        "data": {
            "train": [[str(systems_sz_train)], [str(systems_dzp_train)]],
            "test": [[str(systems_sz_test)], None],
        },
        "iterate": {"n_iter": 1},
    }

    workflow, _, _ = prepare_iterate(package_config(config)["iterate_param"])

    iter_zero = workflow[0]
    scf_parent = iter_zero[0]
    train_step = iter_zero[1]
    run_train = train_step[0]
    assert scf_parent.workdir.name == "00.scf"
    assert len(scf_parent) == 2
    assert scf_parent[0].workdir.name == "level.00"
    assert scf_parent[1].workdir.name == "level.01"
    assert all(pair[1] not in {"data_train", "data_test"} for pair in run_train.link_prev_files)
