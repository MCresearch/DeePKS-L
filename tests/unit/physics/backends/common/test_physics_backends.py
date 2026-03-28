"""Test physics backends structure and interfaces."""

import pytest
import tempfile
import os
import numpy as np


def test_backend_imports():
    """Test that backend modules can be imported."""
    from deepks.physics.backends import (
        PhysicsBackend,
        SCFBackend,
        get_backend,
        get_scf_backend,
        get_physics_backend
    )

    assert PhysicsBackend is not None
    assert SCFBackend is not None
    assert callable(get_backend)
    assert callable(get_scf_backend)
    assert callable(get_physics_backend)


def test_abacus_backend_creation():
    """Test ABACUS backend can be created."""
    from deepks.physics.backends import get_backend

    backend = get_backend('abacus')
    assert backend is not None
    assert backend.backend_name == 'abacus'


def test_abacus_backend_config():
    """Test ABACUS backend with configuration."""
    from deepks.physics.backends import get_backend

    config = {
        'ecutwfc': 100,
        'scf_thr': 1e-7,
        'scf_nmax': 100
    }

    backend = get_backend('abacus', config)
    assert backend.config['ecutwfc'] == 100
    assert backend.config['scf_thr'] == 1e-7


def test_abacus_input_generation():
    """Test ABACUS input file generation."""
    from deepks.physics.backends.abacus.input_generator import (
        make_abacus_scf_input,
        make_abacus_scf_kpt
    )

    params = {
        'ecutwfc': 100,
        'scf_thr': 1e-7,
        'scf_nmax': 100,
        'gamma_only': 1
    }

    # Test INPUT generation
    input_content = make_abacus_scf_input(params)
    assert 'INPUT_PARAMETERS' in input_content
    assert 'ecutwfc 100' in input_content
    assert 'scf_thr 1.000000e-07' in input_content

    # Test KPT generation
    kpt_content = make_abacus_scf_kpt(params)
    assert 'K_POINTS' in kpt_content
    assert 'Gamma' in kpt_content


def test_abacus_stru_generation():
    """Test ABACUS STRU file generation."""
    from deepks.physics.backends.abacus.input_generator import make_abacus_scf_stru

    system_data = {
        'atom_names': ['H', 'O'],
        'atom_numbs': [2, 1],
        'cells': np.array([[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]]),
        'coords': [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])]
    }

    # Use proper file names that match element extraction logic
    pp_files = ['H_ONCV.upf', 'O_ONCV.upf']

    params = {
        'lattice_constant': 1.0,
        'lattice_vector': [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        'coord_type': 'Cartesian'
    }

    stru_content = make_abacus_scf_stru(system_data, pp_files, params)

    assert 'ATOMIC_SPECIES' in stru_content
    assert 'LATTICE_CONSTANT' in stru_content
    assert 'LATTICE_VECTORS' in stru_content
    assert 'ATOMIC_POSITIONS' in stru_content
    assert 'H' in stru_content
    assert 'O' in stru_content


def test_backend_factory_unknown():
    """Test that unknown backend raises error."""
    from deepks.physics.backends import get_backend

    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend('unknown_backend')


def test_get_scf_backend():
    """Test get_scf_backend function."""
    from deepks.physics.backends import get_scf_backend

    backend = get_scf_backend('abacus')
    assert backend is not None
    assert hasattr(backend, 'run_scf')
    assert hasattr(backend, 'collect_stats')


def test_get_physics_backend_backward_compat():
    """Test backward compatibility function."""
    from deepks.physics.backends import get_physics_backend

    backend = get_physics_backend('abacus')
    assert backend is not None
    assert backend.backend_name == 'abacus'


@pytest.mark.skip(reason="PySCF not in test_env")
def test_pyscf_backend_creation():
    """Test PySCF backend can be created (skipped in test_env)."""
    from deepks.physics.backends import get_backend

    backend = get_backend('pyscf')
    assert backend is not None
    assert backend.backend_name == 'pyscf'


@pytest.mark.skip(reason="PySCF not in test_env")
def test_pyscf_backend_not_implemented():
    """Test PySCF backend raises NotImplementedError (skipped in test_env)."""
    from deepks.physics.backends import get_backend

    backend = get_backend('pyscf')

    with pytest.raises(NotImplementedError):
        backend.run_scf([])


def test_abacus_backend_required_files():
    """Test ABACUS backend required files."""
    from deepks.physics.backends import get_backend

    config = {'gamma_only': 1, 'k_points': None}
    backend = get_backend('abacus', config)

    required = backend.get_required_files()
    assert 'INPUT' in required
    assert 'STRU' in required
    assert 'KPT' in required


def test_abacus_backend_output_files():
    """Test ABACUS backend output files."""
    from deepks.physics.backends import get_backend

    config = {
        'deepks_out_labels': 1,
        'cal_force': 1
    }
    backend = get_backend('abacus', config)

    output = backend.get_output_files()
    assert any('running_scf.log' in f for f in output)
    assert any('deepks.dm_eig' in f for f in output)


def test_abacus_parser_functions():
    """Test ABACUS parser function imports."""
    from deepks.physics.backends.abacus.parser import (
        parse_abacus_energy,
        parse_abacus_forces,
        parse_abacus_stress,
        parse_abacus_descriptor,
        parse_abacus_bandgap,
        check_convergence
    )

    assert callable(parse_abacus_energy)
    assert callable(parse_abacus_forces)
    assert callable(parse_abacus_stress)
    assert callable(parse_abacus_descriptor)
    assert callable(parse_abacus_bandgap)
    assert callable(check_convergence)
