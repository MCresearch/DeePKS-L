"""Unit tests for unified I/O input module."""

import pytest
import tempfile
import os
from deepks.io.input import load_config, merge_configs, validate_config, get_default_config
from deepks.io.input.merger import apply_parameter_inheritance
from deepks.io.input.config import normalize_config


class TestDefaults:
    """Test default configuration generation."""

    def test_scf_pyscf_defaults(self):
        config = get_default_config('scf', 'pyscf')
        assert config['scf_pyscf']['basis'] == 'ccpvdz'
        assert config['verbose'] == 1
        assert 'dump_fields' in config
        assert 'mol_args' in config['scf_pyscf']
        assert 'scf_args' in config['scf_pyscf']

    def test_scf_abacus_defaults(self):
        config = get_default_config('scf', 'abacus')
        assert config['scf_abacus']['ecutwfc'] == 50
        assert config['scf_abacus']['dft_functional'] == 'pbe'
        assert config['scf_abacus']['basis_type'] == 'lcao'
        assert config['verbose'] == 1

    def test_train_defaults(self):
        config = get_default_config('train')
        assert 'model_args' in config
        assert 'train_args' in config
        assert config['train_args']['n_epoch'] == 1000

    def test_iterate_defaults(self):
        config = get_default_config('iterate', 'pyscf')
        assert config['n_iter'] == 10
        assert 'basis' in config['scf_pyscf']  # SCF defaults in nested block
        assert 'model_args' in config  # Should include train defaults


class TestMerge:
    """Test configuration merging."""

    def test_simple_merge(self):
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = merge_configs(base, override)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_deep_merge(self):
        base = {
            'verbose': 1,
            'scf_args': {
                'conv_tol': 1e-7,
                'max_cycle': 50,
            }
        }
        override = {
            'verbose': 2,
            'scf_args': {
                'conv_tol': 1e-9,
            }
        }
        result = merge_configs(base, override)
        assert result['verbose'] == 2
        assert result['scf_args']['conv_tol'] == 1e-9
        assert result['scf_args']['max_cycle'] == 50

    def test_non_dict_override(self):
        base = {'a': {'b': 1}}
        override = {'a': 'string'}
        result = merge_configs(base, override)
        assert result['a'] == 'string'


class TestInheritance:
    """Test parameter inheritance."""

    def test_scf_inheritance(self):
        config = {
            'scf_input': {
                'basis': 'ccpvdz',
                'scf_args': {'conv_tol': 1e-7},
            },
            'init_scf': {
                'scf_args': {'conv_tol': 1e-5},
            }
        }
        result = apply_parameter_inheritance(config)
        assert result['init_scf']['basis'] == 'ccpvdz'
        assert result['init_scf']['scf_args']['conv_tol'] == 1e-5

    def test_train_inheritance(self):
        config = {
            'train_input': {
                'n_epoch': 1000,
                'batch_size': 16,
            }
        }
        result = apply_parameter_inheritance(config)
        assert 'init_train' in result
        assert result['init_train']['n_epoch'] == 1000
        assert result['init_train']['batch_size'] == 16

    def test_machine_inheritance(self):
        config = {
            'scf_machine': {
                'sub_size': 1,
                'group_size': 1,
            }
        }
        result = apply_parameter_inheritance(config)
        assert 'init_scf_machine' in result
        assert result['init_scf_machine']['sub_size'] == 1


class TestNormalization:
    """Test schema-driven normalization (replaces old separate_backend_params)."""

    def test_flat_pyscf_keys_fold_into_nested_block(self):
        config = {
            'type': 'scf',
            'systems': ['sys1'],
            'basis': 'ccpvdz',
            'mol_args': {'charge': 0},
        }
        result = normalize_config(config, 'scf')
        assert result['scf_pyscf']['basis'] == 'ccpvdz'
        assert result['scf_pyscf']['mol_args']['charge'] == 0
        assert 'basis' not in result  # flat key folded away

    def test_flat_abacus_keys_fold_into_nested_block(self):
        config = {
            'type': 'scf',
            'scf_soft': 'abacus',
            'systems': ['sys1'],
            'ecutwfc': 100,
            'orb_files': ['Si.orb'],
            'pp_files': ['Si.upf'],
        }
        result = normalize_config(config, 'scf')
        assert result['scf_abacus']['ecutwfc'] == 100
        assert result['scf_abacus']['orb_files'] == ['Si.orb']
        assert 'ecutwfc' not in result

    def test_no_backend_params_still_normalizes(self):
        config = {
            'type': 'scf',
            'systems': ['sys1'],
        }
        result = normalize_config(config, 'scf')
        assert 'scf_soft' in result


class TestValidation:
    """Test configuration validation."""

    def test_valid_scf_config(self):
        config = {
            'systems': ['sys1'],
            'scf_soft': 'pyscf',
            'basis': 'ccpvdz',
        }
        # Should not raise
        validate_config(config, 'scf')

    def test_missing_systems(self):
        config = {
            'scf_soft': 'pyscf',
            'basis': 'ccpvdz',
        }
        with pytest.raises(ValueError, match="requires 'systems'"):
            validate_config(config, 'scf')

    def test_invalid_backend(self):
        config = {
            'systems': ['sys1'],
            'scf_soft': 'invalid',
        }
        with pytest.raises(ValueError, match="Invalid scf_soft"):
            validate_config(config, 'scf')

    def test_abacus_missing_required(self):
        config = {
            'systems': ['sys1'],
            'scf_soft': 'abacus',
        }
        with pytest.raises(ValueError, match="requires 'orb_files'"):
            validate_config(config, 'scf')

    def test_invalid_type(self):
        config = {
            'systems': ['sys1'],
            'scf_soft': 'pyscf',
            'basis': 123,  # Should be string
        }
        with pytest.raises(TypeError, match="must be string"):
            validate_config(config, 'scf')


class TestLoader:
    """Test configuration loader."""

    def test_load_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('systems:\n  - sys1\nbasis: ccpvdz\n')
            f.flush()
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config['systems'] == ['sys1']
            assert config['basis'] == 'ccpvdz'
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/file.yaml')

    def test_load_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content:\n')
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load"):
                load_config(config_path)
        finally:
            os.unlink(config_path)
