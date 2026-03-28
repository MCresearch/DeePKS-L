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
        f.write('scf_soft: pyscf\n')
        f.write('systems:\n  - sys1\n')
        f.write('basis: ccpvtz\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import build_runtime_config

        runtime = build_runtime_config(config_path)
        raw = runtime['raw_config']
        scf_param = runtime['scf_param']

        assert raw['type'] == 'scf'
        assert raw['scf_soft'] == 'pyscf'
        assert raw['scf_pyscf']['basis'] == 'ccpvtz'
        assert raw['verbose'] == 1
        assert 'mol_args' in raw['scf_pyscf']
        assert scf_param['scf_pyscf']['basis'] == 'ccpvtz'

    finally:
        os.unlink(config_path)


def test_cli_iterate_inheritance():
    """Test iterate inheritance with nested backend config."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: iterate\n')
        f.write('scf_soft: pyscf\n')
        f.write('systems_train:\n  - sys1\n')
        f.write('n_iter: 5\n')
        f.write('scf_input:\n')
        f.write('  basis: ccpvdz\n')
        f.write('  scf_args:\n')
        f.write('    conv_tol: 1e-7\n')
        f.write('init_scf:\n')
        f.write('  scf_args:\n')
        f.write('    conv_tol: 1e-5\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import build_runtime_config

        runtime = build_runtime_config(config_path)
        final = runtime['raw_config']

        assert final['init_scf']['basis'] == 'ccpvdz'
        assert final['init_scf']['scf_args']['conv_tol'] == 1e-5
        assert final['scf_pyscf']['basis'] == 'ccpvdz'

    finally:
        os.unlink(config_path)


def test_cli_normalizes_flat_backend_keys_into_nested_blocks():
    """Test flat backend keys normalize into backend blocks."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: scf\n')
        f.write('systems:\n  - sys1\n')
        f.write('basis: sto-3g\n')
        f.write('mol_args:\n')
        f.write('  charge: 1\n')
        f.write('scf_args:\n')
        f.write('  max_cycle: 3\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import build_runtime_config

        runtime = build_runtime_config(config_path)
        raw = runtime['raw_config']
        scf_param = runtime['scf_param']

        assert raw['scf_soft'] == 'pyscf'
        assert raw['scf_pyscf']['basis'] == 'sto-3g'
        assert raw['scf_pyscf']['mol_args']['charge'] == 1
        assert raw['scf_pyscf']['scf_args']['max_cycle'] == 3
        assert scf_param['scf_pyscf']['basis'] == 'sto-3g'
        assert scf_param['scf_pyscf']['mol_args']['charge'] == 1
        assert scf_param['scf_pyscf']['scf_args']['max_cycle'] == 3

    finally:
        os.unlink(config_path)


def test_cli_normalizes_iterate_abacus_init_compatibility():
    """Test iterate ABACUS init compatibility normalization."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: iterate\n')
        f.write('scf_soft: abacus\n')
        f.write('systems_train:\n  - sys1\n')
        f.write('scf_abacus:\n')
        f.write('  orb_files:\n')
        f.write('    - orb\n')
        f.write('  pp_files:\n')
        f.write('    - upf\n')
        f.write('  abacus_path: abacus\n')
        f.write('init_scf:\n')
        f.write('  scf_nmax: 88\n')
        f.write('  orb_files:\n')
        f.write('    - orb\n')
        f.write('  pp_files:\n')
        f.write('    - upf\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import build_runtime_config

        runtime = build_runtime_config(config_path)
        raw = runtime['raw_config']
        iterate_param = runtime['iterate_param']

        assert raw['init_scf'] is True
        assert raw['init_scf_abacus']['scf_nmax'] == 88
        assert raw['scf_abacus']['abacus_path'] == 'abacus'
        assert raw['init_scf_abacus']['orb_files'] == ['orb']
        assert iterate_param['init_scf_abacus']['scf_nmax'] == 88

    finally:
        os.unlink(config_path)


def test_cli_auto_detects_cuda_device_when_user_omits_device(monkeypatch):
    """Test that input normalization auto-selects CUDA when available."""
    fake_torch = types.ModuleType('torch')
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('type: test\n')
        f.write('systems_test:\n')
        f.write('  - sys1\n')
        f.write('model_file: model.pth\n')
        f.flush()
        config_path = f.name

    try:
        from deepks.io.input import build_runtime_config

        runtime = build_runtime_config(config_path)

        assert runtime['raw_config']['device'] == 'cuda:0'
        assert runtime['global_param']['device'] == 'cuda:0'
    finally:
        os.unlink(config_path)


def test_dispatcher_type_detection():
    """Test type dispatcher."""
    from deepks.io.input.dispatcher import dispatch_command

    with pytest.raises(ValueError, match="Unknown type"):
        dispatch_command({'type': 'invalid'})
