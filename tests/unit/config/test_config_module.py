"""Unit tests for unified I/O input module."""

import pytest
import tempfile
import os
from deepks.config import load_config, merge_configs, validate_config, get_default_config
from deepks.config.normalize import normalize_config
from deepks.config.packager import get_packed_payload, package_config
from deepks.workflows.iterate.support import resolve_hierarchical_iterate_levels


class TestDefaults:
    """Test default configuration generation."""

    def test_scf_pyscf_defaults(self):
        config = get_default_config('scf', 'pyscf')
        assert config['type'] == 'scf'
        assert config['runtime']['verbose'] == 1
        assert config['physics']['backend']['name'] == 'pyscf'
        assert config['physics']['backend']['input']['basis'] == 'ccpvdz'
        assert 'mol_args' in config['physics']['backend']['input']
        assert 'scf_args' in config['physics']['backend']['input']

    def test_scf_abacus_defaults(self):
        config = get_default_config('scf', 'abacus')
        assert config['type'] == 'scf'
        assert config['runtime']['verbose'] == 1
        assert config['physics']['backend']['name'] == 'abacus'
        assert config['physics']['backend']['input']['ecutwfc'] == 50
        assert config['physics']['backend']['input']['dft_functional'] == 'pbe'
        assert config['physics']['backend']['input']['basis_type'] == 'lcao'

    def test_train_defaults(self):
        config = get_default_config('train')
        assert config['type'] == 'train'
        assert config['ml']['model']['args']
        assert config['ml']['train']['epochs'] == 1000
        assert config['ml']['objective']['losses'] == []
        assert 'energy_factor' not in config['ml']['objective']
        assert 'force_factor' not in config['ml']['objective']

    def test_iterate_defaults(self):
        config = get_default_config('iterate', 'pyscf')
        assert config['type'] == 'iterate'
        assert config['iterate']['n_iter'] == 10
        assert config['physics']['backend']['input']['basis'] == 'ccpvdz'
        assert config['ml']['model']['args']


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


class TestNormalization:
    """Test schema-driven normalization."""

    def test_dotted_keys_expand_into_nested_block(self):
        config = {
            'type': 'scf',
            'data.systems': ['sys1'],
            'physics.backend.name': 'pyscf',
            'physics.backend.input.basis': 'ccpvdz',
            'physics.backend.input.mol_args.charge': 0,
        }
        result = normalize_config(config)
        assert result['data']['systems'] == ['sys1']
        assert result['physics']['backend']['name'] == 'pyscf'
        assert result['physics']['backend']['input']['basis'] == 'ccpvdz'
        assert result['physics']['backend']['input']['mol_args']['charge'] == 0

    def test_mixed_dotted_and_nested_keys_merge(self):
        config = {
            'type': 'scf',
            'physics.backend.name': 'abacus',
            'physics': {'backend': {'input': {'orb_files': ['Si.orb'], 'pp_files': ['Si.upf']}}},
            'physics.backend.input.ecutwfc': 100,
        }
        result = normalize_config(config)
        assert result['physics']['backend']['name'] == 'abacus'
        assert result['physics']['backend']['input']['ecutwfc'] == 100
        assert result['physics']['backend']['input']['orb_files'] == ['Si.orb']

    def test_conflicting_dotted_values_raise(self):
        config = {
            'physics.backend.name': 'abacus',
            'physics': {'backend': {'name': 'pyscf'}},
        }
        with pytest.raises(ValueError, match='Conflicting configuration values'):
            normalize_config(config)


class TestValidation:
    """Test configuration validation."""

    def test_valid_scf_config(self):
        config = {
            'data': {'systems': ['sys1']},
            'physics': {'backend': {'name': 'pyscf', 'input': {'basis': 'ccpvdz'}}},
        }
        # Should not raise
        validate_config(config, 'scf')

    def test_missing_systems(self):
        config = {
            'physics': {'backend': {'name': 'pyscf', 'input': {'basis': 'ccpvdz'}}},
        }
        with pytest.raises(ValueError, match="data.systems"):
            validate_config(config, 'scf')

    def test_missing_backend_name(self):
        config = {
            'data': {'systems': ['sys1']},
        }
        with pytest.raises(ValueError, match="physics.backend.name"):
            validate_config(config, 'scf')

    def test_invalid_backend(self):
        config = {
            'data': {'systems': ['sys1']},
            'physics': {'backend': {'name': 'invalid'}},
        }
        with pytest.raises(ValueError, match="Invalid physics\\.backend\\.name"):
            validate_config(config, 'scf')

    def test_abacus_missing_required(self):
        config = {
            'data': {'systems': ['sys1']},
            'physics': {'backend': {'name': 'abacus', 'input': {}}},
        }
        with pytest.raises(ValueError, match="requires 'orb_files'"):
            validate_config(config, 'scf')

    def test_invalid_type(self):
        config = {
            'data': {'systems': ['sys1']},
            'physics': {'backend': {'name': 'pyscf', 'input': {'basis': 123}}},
        }
        with pytest.raises(TypeError, match="must be string"):
            validate_config(config, 'scf')

    def test_valid_hierarchical_iterate_config(self):
        config = {
            'recipe': 'hierarchical-regression',
            'physics': {
                'backend': {
                    'name': 'abacus',
                    'input': {'orb_files': ['Si.orb'], 'pp_files': ['Si.upf']},
                    'profiles': [
                        {'name': 'sz', 'input': {'basis_name': 'sz'}},
                        {'name': 'dzp', 'input_template': 'scf_dzp.yaml'},
                    ],
                },
            },
            'ml': {
                'model': {
                    'args': {
                        'levels': [
                            {'name': 'sz', 'output_dim': 8},
                            {'name': 'dzp', 'output_dim': 26},
                        ]
                    }
                }
            },
            'data': {
                'train': [['systems_sz/data_train'], ['systems_dzp/data_train']],
                'test': [['systems_sz/data_test'], None],
            },
        }
        validate_config(config, 'iterate')

    def test_hierarchical_iterate_allows_global_backend_input_without_profiles(self):
        config = {
            'recipe': 'hierarchical-regression',
            'physics': {
                'backend': {'name': 'abacus', 'input': {'orb_files': ['Si.orb'], 'pp_files': ['Si.upf']}},
            },
            'ml': {
                'model': {'args': {'levels': [{'name': 'sz', 'output_dim': 8}]}}
            },
            'data': {
                'train': [['systems_sz/data_train']],
            },
        }
        validate_config(config, 'iterate')

    def test_hierarchical_iterate_rejects_missing_effective_scf_input(self):
        config = {
            'recipe': 'hierarchical-regression',
            'physics': {
                'backend': {'name': 'abacus', 'input': {}},
            },
            'ml': {
                'model': {'args': {'levels': [{'name': 'sz', 'output_dim': 8}]}}
            },
            'data': {
                'train': [['systems_sz/data_train']],
            },
        }
        with pytest.raises(ValueError, match="effective backend input"):
            validate_config(config, 'iterate')

    def test_hierarchical_iterate_allows_level_local_orb_files_with_global_pp_files(self):
        config = {
            'recipe': 'hierarchical-regression',
            'physics': {
                'backend': {
                    'name': 'abacus',
                    'input': {'pp_files': ['Si.upf']},
                    'profiles': [{'name': 'sz', 'input': {'orb_files': ['Si_sz.orb']}}],
                },
            },
            'ml': {
                'train': {'stage_schedule': [{'level': 0, 'epochs': 10}]},
                'model': {'args': {'levels': [{'name': 'sz', 'output_dim': 8}]}}
            },
            'data': {
                'train': [['systems_sz/data_train']],
            },
        }
        validate_config(config, 'iterate')


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


class TestPackager:
    """Test packed-output cleanup for current objective schema."""

    def test_package_train_objective_strips_legacy_fields_when_losses_present(self):
        config = get_default_config('train')
        config['recipe'] = 'corrnet-energy'
        config['data'] = {'train': ['data_train/*']}
        config['ml']['objective'] = {
            'losses': [{'name': 'energy', 'weight': 1.0}],
            'energy_factor': 9.9,
            'force_factor': 8.8,
            'energy_per_atom': 2,
        }

        packed = package_config(config)['train_param']

        assert packed['ml']['objective'] == {
            'losses': [{'name': 'energy', 'weight': 1.0}],
            'energy_per_atom': 2,
        }

    def test_package_train_objective_converts_legacy_fields_to_losses(self):
        config = get_default_config('train')
        config['recipe'] = 'corrnet-energy'
        config['data'] = {'train': ['data_train/*']}
        config['ml']['objective'] = {
            'energy_factor': 1.0,
            'force_factor': 0.5,
            'force_loss': {'cap': 0.25},
            'phi_factor': 1.0,
            'phi_occ': 4,
            'energy_per_atom': 2,
        }

        packed = package_config(config)['train_param']

        assert packed['ml']['objective'] == {
            'losses': [
                {'name': 'energy', 'weight': 1.0},
                {'name': 'force', 'weight': 0.5, 'loss': {'cap': 0.25}},
                {'name': 'phi', 'weight': 1.0, 'occ': 4},
            ],
            'energy_per_atom': 2,
        }

    def test_package_iterate_preserves_hierarchical_level_metadata(self):
        config = get_default_config('iterate', 'abacus')
        config['recipe'] = 'hierarchical-regression'
        config['physics']['backend']['profiles'] = [
            {'name': 'sz', 'input': {'basis_name': 'sz'}},
            {'name': 'dzp', 'input_template': 'scf_dzp.yaml'},
        ]
        config['ml']['model']['args']['levels'] = [
            {'name': 'sz', 'output_dim': 8},
            {'name': 'dzp', 'output_dim': 26},
        ]
        config['data']['train'] = [['systems_sz/data_train'], ['systems_dzp/data_train']]
        config['data']['test'] = [None, ['systems_dzp/data_test']]

        packed = package_config(config)['iterate_param']

        assert packed['data']['test'][1] == ['systems_dzp/data_test']
        assert packed['ml']['checkpoint'] == {}
        stage_specs = packed['iterate']['tasks']['main']['train']['train_param']['data']['stages']
        assert packed['iterate']['tasks']['main']['train']['train_param']['ml']['model']['args']['levels'][0]['name'] == 'sz'
        assert packed['iterate']['tasks']['main']['scf']['scf_param']['ml']['model']['args']['levels'][0]['name'] == 'sz'
        assert 'target' not in stage_specs[0]
        assert packed['iterate']['tasks']['main']['train']['train_param']['ml']['objective']['terms'][0]['target']['format'] == 'collected_hr_delta'
        assert stage_specs[1]['train'] == ['../00.scf/level.01/data_train/*']

    def test_package_iterate_hierarchical_uses_global_term_targets(self):
        config = get_default_config('iterate', 'abacus')
        config['recipe'] = 'hierarchical-regression'
        config['physics']['backend']['profiles'] = [
            {'name': 'sz', 'input': {'basis_name': 'sz'}},
        ]
        config['ml']['model']['args']['levels'] = [
            {'name': 'sz', 'output_dim': 1, 'target_shape': [1]},
        ]
        config['data']['train'] = [['systems_sz/data_train']]
        config['data']['test'] = [['systems_sz/data_test']]
        config['ml']['objective'] = {
            'primary_output': 'energy',
            'terms': [
                {'name': 'energy', 'weight': 1.0, 'target': {'format': 'collected_energy_delta', 'name': 'l_e_delta'}},
            ]
        }

        packed = package_config(config)['iterate_param']

        objective_terms = packed['iterate']['tasks']['main']['train']['train_param']['ml']['objective']['terms']
        assert objective_terms[0]['target']['format'] == 'collected_energy_delta'
        assert objective_terms[0]['target']['name'] == 'l_e_delta'


class TestHierarchicalIterateHelpers:
    def test_resolve_hierarchical_iterate_levels_merges_metadata(self):
        config = {
            'recipe': 'hierarchical-regression',
            'physics': {
                'backend': {
                    'input': {'dft_functional': 'pbe', 'ecutwfc': 60},
                    'profiles': [
                        {'name': 'sz', 'input': {'basis_name': 'sz'}},
                        {'name': 'dzp', 'input': {'basis_name': 'dzp'}},
                    ],
                },
            },
            'ml': {
                'model': {
                    'args': {
                        'levels': [
                            {'name': 'sz', 'output_dim': 8},
                            {'name': 'dzp', 'output_dim': 26},
                        ]
                    }
                },
                'train': {
                    'stage_schedule': [{'level': 0, 'epochs': 2}, {'level': 1, 'epochs': 3}],
                }
            },
            'data': {
                'train': [['systems_sz/data_train'], ['systems_dzp/data_train']],
            },
        }

        levels = resolve_hierarchical_iterate_levels(config, require_complete=True)

        assert [item['level'] for item in levels] == [0, 1]
        assert levels[0]['model_level']['name'] == 'sz'
        assert levels[1]['systems']['train_paths'] == ['systems_dzp/data_train']
        assert levels[0]['merged_backend_input']['dft_functional'] == 'pbe'
        assert levels[0]['merged_backend_input']['basis_name'] == 'sz'

    def test_materialize_hierarchical_level_scf_config_overrides_packed_child_backend_input(self):
        from deepks.workflows.iterate.support.task_params import (
            build_abacus_iterate_scf_kwargs,
            materialize_hierarchical_level_scf_config,
        )

        config = {
            'type': 'iterate',
            'recipe': 'hierarchical-regression',
            'runtime': {'scf': {'command': {'abacus_path': '/bin/echo', 'run_cmd': 'mpirun'}}},
            'physics': {
                'backend': {
                    'name': 'abacus',
                    'input': {'pp_files': ['Si.upf'], 'proj_file': ['jle.orb']},
                    'profiles': [{'name': 'sz', 'input': {'orb_files': ['Si_sz.orb']}}],
                },
            },
            'ml': {
                'model': {
                    'args': {
                        'levels': [
                            {
                                'name': 'sz',
                                'output_dim': 8,
                                'target_shape': [3, 3, 3, 8, 8],
                            }
                        ]
                    }
                },
                'objective': {
                    'primary_output': 'hamiltonian',
                    'terms': [{'name': 'hr', 'target': {'format': 'collected_hr_delta', 'name': 'l_hr_delta'}}],
                },
            },
            'data': {
                'train': [['systems_sz/data_train']],
            },
            'iterate': {'n_iter': 1},
        }

        packed = package_config(config)
        payload = get_packed_payload(packed)
        scf_child = payload['iterate']['tasks']['main']['scf']
        level_meta = resolve_hierarchical_iterate_levels(payload, require_complete=True)[0]
        materialized = materialize_hierarchical_level_scf_config(scf_child, level_meta)
        kwargs = build_abacus_iterate_scf_kwargs(materialized)

        assert kwargs['orb_files'] == ['Si_sz.orb']
        assert kwargs['pp_files'] == ['Si.upf']
        assert kwargs['proj_file'] == ['jle.orb']
        assert kwargs['backend_kwargs']['target_shape'] == [3, 3, 3, 8, 8]

    def test_build_abacus_iterate_scf_kwargs_ignores_scalar_target_shape(self):
        from deepks.workflows.iterate.support.task_params import build_abacus_iterate_scf_kwargs

        scf_config = {
            'type': 'scf',
            'scf_param': {
                'physics': {
                    'backend': {
                        'input': {
                            'orb_files': ['Si.orb'],
                            'pp_files': ['Si.upf'],
                            'proj_file': ['proj.orb'],
                        }
                    }
                },
                'ml': {
                    'objective': {'primary_output': 'energy', 'terms': []},
                    'model': {
                        'args': {
                            'levels': [
                                {'name': 'sz', 'target_shape': [1]},
                            ]
                        }
                    }
                },
            },
            '__internal_packed__': True,
        }

        kwargs = build_abacus_iterate_scf_kwargs(scf_config)

        assert 'target_shape' not in kwargs['backend_kwargs']

    def test_build_abacus_iterate_scf_kwargs_uses_hr_term_target_shape(self):
        from deepks.workflows.iterate.support.task_params import build_abacus_iterate_scf_kwargs

        scf_config = {
            'type': 'scf',
            'scf_param': {
                'physics': {
                    'backend': {
                        'input': {
                            'orb_files': ['Si.orb'],
                            'pp_files': ['Si.upf'],
                            'proj_file': ['proj.orb'],
                        }
                    }
                },
                'ml': {
                    'objective': {
                        'primary_output': 'energy',
                        'terms': [{'name': 'hr', 'target': {'format': 'collected_hr_delta', 'name': 'l_hr_delta'}}],
                    },
                    'model': {
                        'args': {
                            'levels': [
                                {
                                    'name': 'sz',
                                    'target_shape': [9, 9, 9, 8, 8],
                                },
                            ]
                        }
                    }
                },
            },
            '__internal_packed__': True,
        }

        kwargs = build_abacus_iterate_scf_kwargs(scf_config)

        assert kwargs['backend_kwargs']['target_shape'] == [9, 9, 9, 8, 8]
