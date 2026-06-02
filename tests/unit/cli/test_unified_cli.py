"""Test unified CLI interface."""

import importlib
import pytest
import tempfile
import os
import sys
import types


MAIN_MODULE_NAME = 'deepks.main'


def load_main_module():
    """Load package CLI module."""
    if MAIN_MODULE_NAME in sys.modules:
        return importlib.reload(sys.modules[MAIN_MODULE_NAME])
    return importlib.import_module(MAIN_MODULE_NAME)


def test_cli_help(capsys):
    """Test CLI help message."""
    original_argv = sys.argv

    try:
        sys.argv = ['deepks', '--help']
        main_module = load_main_module()

        with pytest.raises(SystemExit) as exc_info:
            main_module.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert 'DeePKS' in captured.out
        assert 'Configuration file' in captured.out
    finally:
        sys.argv = original_argv


def test_cli_missing_config():
    """Test CLI with missing config file."""
    original_argv = sys.argv

    try:
        sys.argv = ['deepks', 'nonexistent.yaml']
        main_module = load_main_module()

        with pytest.raises(SystemExit) as exc_info:
            main_module.main()

        assert exc_info.value.code == 1
    finally:
        sys.argv = original_argv


def test_cli_config_without_type():
    """Test CLI with config missing type field."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('systems:\n  - sys1\n')
        f.flush()
        config_path = f.name

    original_argv = sys.argv

    try:
        sys.argv = ['deepks', config_path]
        main_module = load_main_module()

        with pytest.raises(SystemExit) as exc_info:
            main_module.main()

        assert exc_info.value.code == 1
    finally:
        sys.argv = original_argv
        os.unlink(config_path)


def test_cli_config_loading():
    """Test schema-derived config loading and merging."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: scf\n')
        f.write('data:\n  systems:\n    - sys1\n')
        f.write('physics:\n')
        f.write('  backend:\n')
        f.write('    name: pyscf\n')
        f.write('    input:\n')
        f.write('      basis: ccpvtz\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import load_runtime_config

        runtime = load_runtime_config(config_path)
        raw = runtime['scf_param']

        assert raw['type'] == 'scf'
        assert raw['physics']['backend']['name'] == 'pyscf'
        assert raw['physics']['backend']['input']['basis'] == 'ccpvtz'
        assert raw['runtime']['verbose'] == 1
        assert 'mol_args' in raw['physics']['backend']['input']

    finally:
        os.unlink(config_path)


def test_cli_iterate_phase_configuration():
    """Test iterate uses the new phase-value configuration style."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: iterate\n')
        f.write('data:\n')
        f.write('  train:\n')
        f.write('    - sys1\n')
        f.write('physics:\n')
        f.write('  backend:\n')
        f.write('    name: pyscf\n')
        f.write('    input:\n')
        f.write('      basis: ccpvdz\n')
        f.write('      scf_args:\n')
        f.write('        main:\n')
        f.write('          conv_tol: 1e-7\n')
        f.write('        init:\n')
        f.write('          conv_tol: 1e-5\n')
        f.write('iterate:\n')
        f.write('  n_iter: 5\n')
        f.write('  use_init: true\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import load_runtime_config

        runtime = load_runtime_config(config_path)
        final = runtime['iterate_param']
        scf_main = final['iterate']['tasks']['main']['scf']['scf_param']
        scf_init = final['iterate']['tasks']['init']['scf']['scf_param']

        assert scf_main['physics']['backend']['input']['basis'] == 'ccpvdz'
        assert scf_main['physics']['backend']['input']['scf_args']['conv_tol'] == 1e-7
        assert scf_init['physics']['backend']['input']['scf_args']['conv_tol'] == 1e-5
        assert final['iterate']['use_init'] is True

    finally:
        os.unlink(config_path)


def test_cli_expands_dotted_backend_keys_into_nested_blocks():
    """Test dotted backend keys expand into backend blocks."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: scf\n')
        f.write('data.systems:\n')
        f.write('  - sys1\n')
        f.write('physics.backend.name: pyscf\n')
        f.write('physics.backend.input.basis: sto-3g\n')
        f.write('physics.backend.input.mol_args.charge: 1\n')
        f.write('physics.backend.input.scf_args.max_cycle: 3\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import load_runtime_config

        runtime = load_runtime_config(config_path)
        raw = runtime['scf_param']

        assert raw['physics']['backend']['name'] == 'pyscf'
        assert raw['physics']['backend']['input']['basis'] == 'sto-3g'
        assert raw['physics']['backend']['input']['mol_args']['charge'] == 1
        assert raw['physics']['backend']['input']['scf_args']['max_cycle'] == 3
    finally:
        os.unlink(config_path)


def test_cli_normalizes_iterate_phase_values():
    """Test iterate phase values stay on the new structured schema."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: iterate\n')
        f.write('data:\n')
        f.write('  train:\n')
        f.write('    - sys1\n')
        f.write('physics:\n')
        f.write('  backend:\n')
        f.write('    name: abacus\n')
        f.write('    input:\n')
        f.write('      orb_files:\n')
        f.write('        - orb\n')
        f.write('      pp_files:\n')
        f.write('        - upf\n')
        f.write('      scf_nmax:\n')
        f.write('        - 50\n')
        f.write('        - 88\n')
        f.write('runtime:\n')
        f.write('  scf:\n')
        f.write('    command:\n')
        f.write('      abacus_path: abacus\n')
        f.write('iterate:\n')
        f.write('  use_init: true\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import load_runtime_config

        runtime = load_runtime_config(config_path)
        raw = runtime['iterate_param']
        assert raw['iterate']['tasks']['main']['scf']['scf_param']['physics']['backend']['input']['scf_nmax'] == 50
        assert raw['iterate']['tasks']['init']['scf']['scf_param']['physics']['backend']['input']['scf_nmax'] == 88
        assert raw['iterate']['tasks']['main']['scf']['scf_param']['physics']['backend']['input']['orb_files'] == ['orb']
        assert raw['iterate']['tasks']['main']['scf']['scf_param']['runtime']['scf']['command']['abacus_path'] == 'abacus'
        assert raw['iterate']['use_init'] is True

    finally:
        os.unlink(config_path)


def test_cli_auto_detects_cuda_device_when_user_omits_device(monkeypatch):
    """Test that input normalization auto-selects CUDA when available."""
    fake_torch = types.ModuleType('torch')
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: test\n')
        f.write('data:\n')
        f.write('  test:\n')
        f.write('    - sys1\n')
        f.write('ml:\n')
        f.write('  checkpoint:\n')
        f.write('    file: model.pth\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import load_runtime_config

        runtime = load_runtime_config(config_path)

        assert runtime['test_param']['runtime']['device'] == 'cuda:0'
    finally:
        os.unlink(config_path)


def test_cli_normalizes_interface_style_train_config():
    """Test new scheme/model/physics/objective blocks map into runtime config."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: train\n')
        f.write('recipe: corrnet-energy\n')
        f.write('data:\n')
        f.write('  train:\n')
        f.write('    - sys1\n')
        f.write('ml:\n')
        f.write('  model:\n')
        f.write('    family: corrnet\n')
        f.write('    args:\n')
        f.write('      hidden_sizes:\n')
        f.write('        - 8\n')
        f.write('  objective:\n')
        f.write('    losses:\n')
        f.write('      energy: 1.0\n')
        f.write('      force:\n')
        f.write('        weight: 0.2\n')
        f.write('        loss:\n')
        f.write('          cap: 0.5\n')
        f.write('physics:\n')
        f.write('  representation:\n')
        f.write('    name: dm_eig\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import load_runtime_config

        runtime = load_runtime_config(config_path)
        raw = runtime['train_param']

        assert raw['recipe'] == 'corrnet-energy'
        assert raw['ml']['model']['args']['hidden_sizes'] == [8]
        assert raw['physics']['representation']['name'] == 'dm_eig'
        assert raw['ml']['objective']['losses']['energy'] == 1.0
        assert raw['ml']['objective']['losses']['force']['weight'] == 0.2
        assert raw['ml']['objective']['losses']['force']['loss']['cap'] == 0.5
    finally:
        os.unlink(config_path)


def test_dispatcher_type_detection():
    """Test type dispatcher."""
    from deepks.io.input.dispatcher import dispatch_command

    with pytest.raises(ValueError, match="Unknown type"):
        dispatch_command({'type': 'invalid'})
