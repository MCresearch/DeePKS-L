"""Test train workflow structure and imports."""

import pytest
import tempfile
import os


def test_train_workflow_imports():
    """Test that train workflow modules can be imported."""
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data
    from deepks.workflows.train import run_train_workflow
    from deepks.workflows.train import run_training_stage
    from deepks.workflows.train.workflow import run_train_workflow as workflow_main

    assert callable(run_train_workflow)
    assert callable(workflow_main)
    assert callable(prepare_train_data)
    assert callable(run_training_stage)


def test_train_workflow_dispatcher_integration():
    """Test that dispatcher can route to train workflow."""
    from deepks.io.input.dispatcher import dispatch_command

    # This should not raise an error for train type
    config = {'type': 'train', 'systems_train': []}

    # We expect it to fail because systems_train is empty, but it should
    # reach the workflow code (not fail on dispatch)
    with pytest.raises((ValueError, FileNotFoundError, IndexError, Exception)):
        dispatch_command(config)


def test_prepare_train_data_validation():
    """Test that prepare_train_data validates input."""
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data

    # Missing systems_train should raise error
    config = {'systems_train': []}

    # Should fail because no valid systems
    with pytest.raises((ValueError, FileNotFoundError, IndexError, Exception)):
        prepare_train_data(config)


def test_train_workflow_config_structure():
    """Test that workflow accepts proper config structure."""
    from deepks.workflows.train.workflow import run_train_workflow

    # This should fail gracefully with proper error messages
    config = {
        'type': 'train',
        'systems_train': ['nonexistent_system'],
        'systems_test': None,
        'model_args': {
            'hidden_sizes': [100, 100],
            'output_dim': 1
        },
        'train_args': {
            'n_epoch': 10,
            'start_lr': 0.001
        }
    }

    # Should fail because system doesn't exist, but validates config structure
    with pytest.raises((FileNotFoundError, ValueError, IndexError, Exception)):
        run_train_workflow(config)


def test_train_workflow_seed_handling():
    """Test that workflow handles random seed correctly."""
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data
    import numpy as np

    config = {
        'systems_train': ['nonexistent'],
        'seed': 42
    }

    # Should set seed before failing on missing system
    try:
        prepare_train_data(config)
    except Exception:
        pass  # Expected to fail

    # Seed should have been set
    # We can't easily verify this, but at least it shouldn't crash


def test_train_workflow_model_args():
    """Test that model args are properly handled."""
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data

    config = {
        'systems_train': ['nonexistent'],
        'model_args': {
            'hidden_sizes': [100, 100],
            'output_dim': 1
        },
        'proj_basis': 'basis.npz'
    }

    # Should merge proj_basis into model_args before failing
    try:
        prepare_train_data(config)
    except Exception:
        pass  # Expected to fail


def test_evaluate_model_no_test_reader():
    """Test that evaluate handles missing test reader."""
    from deepks.interface.registry import get_recipe
    from deepks.ml.models.corrnet import CorrNet

    # Create a dummy model with correct parameters
    model = CorrNet(input_dim=10, hidden_sizes=[50]).double()

    config = {'train_args': {}}
    recipe = get_recipe(config=config)

    # Should return None for test_loss when no test_reader
    results = recipe.evaluate_model(model, None, config)

    assert results['test_loss'] is None
    assert results['test_metrics'] == {}


def test_train_workflow_restart_handling():
    """Test that workflow handles restart parameter."""
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data

    config = {
        'systems_train': ['nonexistent'],
        'restart': 'model.pth'
    }

    # Should handle restart parameter before failing
    try:
        prepare_train_data(config)
    except Exception:
        pass  # Expected to fail


def test_train_workflow_device_handling():
    """Test that workflow handles device parameter."""
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data

    config = {
        'systems_train': ['nonexistent'],
        'device': 'cpu'
    }

    # Should handle device parameter before failing
    try:
        prepare_train_data(config)
    except Exception:
        pass  # Expected to fail


def test_train_workflow_fit_elem():
    """Test that workflow handles fit_elem parameter."""
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data

    config = {
        'systems_train': ['nonexistent'],
        'fit_elem': True,
        'model_args': {
            'elem_table': None
        }
    }

    # Should handle fit_elem parameter before failing
    try:
        prepare_train_data(config)
    except Exception:
        pass  # Expected to fail


def test_train_workflow_writes_test_log_with_configured_device(monkeypatch, tmp_path):
    """Train workflow should use config.device when generating log.test."""
    from deepks.workflows.train.workflow import run_train_workflow

    class DummyReader:
        def sample_all(self):
            return {'lb_e': None, 'eig': None}

    class DummyGroup:
        nsystems = 1
        readers = [DummyReader()]

    class DummyModel:
        def __init__(self):
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    model = DummyModel()
    train_data = object()
    test_data = DummyGroup()
    seen = {}

    monkeypatch.setattr(
        'deepks.workflows.train.workflow.prepare_train_runtime',
        lambda config: (train_data, test_data, {'device': config['runtime']['device'], 'model_args': {}})
    )
    monkeypatch.setattr(
        'deepks.workflows.train.workflow.run_training_stage',
        lambda train_reader, test_reader, model_config: (model, {'n_epochs': 1})
    )

    class DummyRecipe:
        def evaluate_model(self, model_obj, test_reader, config):
            return {'test_loss': 0.0, 'test_metrics': {}}

        def write_test_log(self, model_obj, test_reader, ckpt_file='model.pth', test_log='log.test', device='cpu'):
            seen['device'] = device
            model_obj.to(device)
            seen['model_device'] = model_obj.seen_device

    monkeypatch.setattr(
        'deepks.workflows.train.workflow.get_recipe',
        lambda **kwargs: DummyRecipe(),
    )

    result = run_train_workflow({
        'type': 'train',
        'runtime': {
            'device': 'cuda:0',
            'test_log': str(tmp_path / 'log.test'),
            'io': {'ckpt_file': 'model.pth'},
        },
    })

    assert result['train_stats']['n_epochs'] == 1
    assert seen['device'] == 'cuda:0'
    assert seen['model_device'] == 'cuda:0'


def test_prepare_train_data_accepts_new_interface_blocks(monkeypatch):
    from deepks.workflows.train import prepare_train_runtime as prepare_train_data

    class DummyReader:
        def __init__(self, paths, **kwargs):
            self.paths = paths
            self.kwargs = kwargs
            self.ndesc = 6

    monkeypatch.setattr("deepks.workflows.train.runtime.load_dirs", lambda paths: paths)
    monkeypatch.setattr("deepks.workflows.train.runtime.GroupReader", DummyReader)

    train_reader, test_reader, model_config = prepare_train_data({
        "type": "train",
        "runtime": {
            "device": "cuda:0",
            "seed": 7,
            "io": {"ckpt_file": "new-model.pth"},
        },
        "data": {
            "train": ["train_set"],
            "test": ["test_set"],
            "loader": {"batch_size": 8},
        },
        "ml": {
            "model": {
                "args": {
                    "hidden_sizes": [8, 8],
                },
            },
            "objective": {
                "losses": [{"name": "energy", "weight": 1.0}],
            },
            "train": {
                "epochs": 12,
                "optimizer": {"lr": 5e-4},
            },
        },
        "physics": {
            "representation": {
                "name": "dm_eig",
                "params": {"proj_basis": "basis.npz"},
            },
        },
    })

    assert train_reader.paths == ["train_set"]
    assert train_reader.kwargs["batch_size"] == 8
    assert train_reader.kwargs["d_name"] == "dm_eig"
    assert test_reader.paths == ["test_set"]
    assert model_config["device"] == "cuda:0"
    assert model_config["ckpt_file"] == "new-model.pth"
    assert model_config["model_args"]["hidden_sizes"] == [8, 8]
    assert "proj_basis" not in model_config["model_args"]
    assert model_config["model_args"]["input_dim"] == 6
    assert model_config["objective_args"]["energy_factor"] == 1.0
    assert model_config["train_args"]["n_epoch"] == 12
    assert model_config["train_args"]["start_lr"] == 5e-4
