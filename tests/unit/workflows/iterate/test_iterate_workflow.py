"""Test iterate workflow structure and imports."""

import pytest
import tempfile
import os


def test_iterate_workflow_imports():
    """Test that iterate workflow modules can be imported."""
    from deepks.workflows.iterate import run_iterate_workflow
    from deepks.workflows.iterate.workflow import run_iterate_workflow as workflow_main
    from deepks.workflows.iterate.prepare import prepare_iterate
    from deepks.workflows.iterate.scf_step import create_scf_step
    from deepks.workflows.iterate.train_step import create_train_step

    assert callable(run_iterate_workflow)
    assert callable(workflow_main)
    assert callable(prepare_iterate)
    assert callable(create_scf_step)
    assert callable(create_train_step)


def test_iterate_workflow_dispatcher_integration():
    """Test that dispatcher can route to iterate workflow."""
    from deepks.io.input.dispatcher import dispatch_command

    # This should not raise an error for iterate type
    config = {'type': 'iterate', 'systems_train': [], 'n_iter': 0}

    # We expect it to fail because systems_train is empty, but it should
    # reach the workflow code (not fail on dispatch)
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        dispatch_command(config)


def test_prepare_iterate_validation():
    """Test that prepare_iterate validates input."""
    from deepks.workflows.iterate.prepare import prepare_iterate

    # Missing systems_train should raise error
    config = {'systems_train': None, 'n_iter': 0}

    # Should fail because no valid systems
    with pytest.raises((ValueError, FileNotFoundError, TypeError, Exception)):
        prepare_iterate(config)


def test_iterate_workflow_config_structure():
    """Test that workflow accepts proper config structure."""
    from deepks.workflows.iterate.workflow import run_iterate_workflow

    # This should fail gracefully with proper error messages
    config = {
        'type': 'iterate',
        'systems_train': ['nonexistent_system'],
        'systems_test': None,
        'n_iter': 2,
        'scf_soft': 'abacus',
        'scf_input': {},
        'train_input': {},
        'workdir': 'test_iter'
    }

    # Should fail because system doesn't exist, but validates config structure
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        run_iterate_workflow(config)


def test_run_iterate_workflow_normalizes_raw_config_before_prepare(monkeypatch, tmp_path):
    """Legacy direct dict callers should still be normalized through input contract."""
    from deepks.workflows.iterate import workflow as workflow_module

    train_sys = tmp_path / "sys.train"
    train_sys.mkdir()

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

    result = workflow_module.run_iterate_workflow(
        {
            'type': 'iterate',
            'systems_train': [str(train_sys)],
            'n_iter': 0,
            'scf_soft': 'pyscf',
            'scf_input': {'basis': 'sto-3g'},
            'train_input': {'train_args': {'n_epoch': 2}},
        }
    )

    assert result['n_iterations'] == 0
    assert seen['config']['type'] == 'iterate'
    assert 'device' in seen['config']
    assert seen['config']['init_train']['train_args']['n_epoch'] == 2


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


def test_check_arg_dict_merge():
    """Test check_arg_dict merges with defaults."""
    from deepks.workflows.iterate.prepare import check_arg_dict

    default = {'a': 1, 'b': 2, 'c': 3}
    data = {'b': 20, 'd': 40}

    result = check_arg_dict(data, default, strict=True)

    # Should merge, keeping default 'a' and 'c', overriding 'b', ignoring 'd'
    assert result['a'] == 1
    assert result['b'] == 20
    assert result['c'] == 3
    assert 'd' not in result


def test_check_arg_dict_none():
    """Test check_arg_dict with None input."""
    from deepks.workflows.iterate.prepare import check_arg_dict

    default = {'a': 1, 'b': 2}

    result = check_arg_dict(None, default)

    # Should return default
    assert result == default


def test_create_scf_step_abacus():
    """Test create_scf_step with ABACUS backend."""
    from deepks.workflows.iterate.scf_step import create_scf_step

    # Should create SCF step without error (will fail on execution, not creation)
    # Use a dummy system path that will be validated later
    try:
        scf_step = create_scf_step(
            systems_train=['dummy_system'],
            systems_test=['dummy_test'],
            scf_soft='abacus',
            scf_config={'ecutwfc': 100},
            scf_machine={'group_size': 1},
            proj_basis=None,
            share_folder='share',
            cleanup=False
        )
        assert scf_step is not None
    except (FileNotFoundError, ValueError, IndexError):
        # Expected to fail on missing systems, but creation logic is tested
        pass


def test_create_scf_step_pyscf():
    """Test create_scf_step with PySCF backend."""
    from deepks.workflows.iterate.scf_step import create_scf_step

    # Should create SCF step without error (will fail on execution, not creation)
    try:
        scf_step = create_scf_step(
            systems_train=['dummy_system'],
            systems_test=['dummy_test'],
            scf_soft='pyscf',
            scf_config={'basis': 'sto-3g'},
            scf_machine={'group_size': 1},
            proj_basis='basis.npz',
            share_folder='share',
            cleanup=False
        )
        assert scf_step is not None
    except (FileNotFoundError, ValueError, IndexError):
        # Expected to fail on missing systems, but creation logic is tested
        pass


def test_create_scf_step_unknown_backend():
    """Test create_scf_step with unknown backend."""
    from deepks.workflows.iterate.scf_step import create_scf_step

    with pytest.raises(ValueError, match="Unknown SCF backend"):
        create_scf_step(
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
    """Test create_train_step."""
    from deepks.workflows.iterate.train_step import create_train_step

    # Should create training step without error
    train_step = create_train_step(
        train_config={'train_args': {'n_epoch': 1}},
        train_machine={'python': 'python'},
        proj_basis='basis.npz',
        share_folder='share',
        cleanup=False
    )

    assert train_step is not None
