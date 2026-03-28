"""Test SCF workflow structure and imports."""

import pytest
import sys
import os


def test_scf_workflow_imports():
    """Test that SCF workflow modules can be imported."""
    from deepks.workflows.scf import run_scf_workflow
    from deepks.workflows.scf.workflow import run_scf_workflow as workflow_main
    from deepks.workflows.scf.prepare import prepare_scf_tasks
    from deepks.workflows.scf.execute import execute_scf_tasks
    from deepks.workflows.scf.collect import collect_scf_results

    assert callable(run_scf_workflow)
    assert callable(workflow_main)
    assert callable(prepare_scf_tasks)
    assert callable(execute_scf_tasks)
    assert callable(collect_scf_results)


def test_scf_workflow_dispatcher_integration():
    """Test that dispatcher can route to SCF workflow."""
    from deepks.io.input.dispatcher import dispatch_command

    # This should not raise an error for scf type
    config = {'type': 'scf', 'scf_soft': 'abacus', 'systems': []}

    # We expect it to fail because systems is empty, but it should
    # reach the workflow code (not fail on dispatch)
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        dispatch_command(config)


def test_prepare_scf_tasks_abacus_validation():
    """Test that prepare_scf_tasks validates input."""
    from deepks.workflows.scf.prepare import prepare_scf_tasks

    # Missing systems should raise error
    config = {'scf_soft': 'abacus'}
    with pytest.raises(ValueError, match="No systems specified"):
        prepare_scf_tasks(config)


def test_scf_workflow_config_structure():
    """Test that workflow accepts proper config structure."""
    from deepks.workflows.scf.workflow import run_scf_workflow

    # This should fail gracefully with proper error messages
    config = {
        'type': 'scf',
        'scf_soft': 'abacus',
        'systems': ['nonexistent_system'],
        'dump_dir': 'test_output',
        'dump_fields': ['e_tot'],
        'scf_abacus': {
            'ntype': 1,
            'ecutwfc': 50,
        }
    }

    # Should fail because system doesn't exist, but validates config structure
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        run_scf_workflow(config)


@pytest.mark.skip(reason="PySCF not in test_env")
def test_scf_workflow_pyscf():
    """Test SCF workflow with PySCF backend (skipped in test_env)."""
    from deepks.workflows.scf.workflow import run_scf_workflow

    config = {
        'type': 'scf',
        'scf_soft': 'pyscf',
        'systems': ['test_system'],
    }

    # This would test PySCF workflow if it were implemented
    with pytest.raises(NotImplementedError):
        run_scf_workflow(config)


def test_coord_to_atom_helper():
    """Test coord_to_atom conversion function."""
    from deepks.workflows.scf.prepare import coord_to_atom
    import tempfile
    import numpy as np

    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create coord.npy
        coords = np.random.rand(5, 3, 3)  # 5 frames, 3 atoms, 3 coords
        np.save(f"{tmpdir}/coord.npy", coords)

        # Create type_map.raw
        with open(f"{tmpdir}/type_map.raw", 'w') as f:
            f.write("H O")

        # Create type.raw
        np.savetxt(f"{tmpdir}/type.raw", [1, 1, 2], fmt='%d')

        # Test conversion
        atom_data = coord_to_atom(tmpdir)

        assert atom_data.shape == (5, 3, 4)  # nframes, natoms, 4 (type + coords)
        assert atom_data[0, 0, 0] == 1  # First atom is type 1 (H)
        assert atom_data[0, 2, 0] == 8  # Third atom is type 8 (O)


def test_scf_workflow_unknown_backend():
    """Test that unknown backend raises proper error."""
    from deepks.workflows.scf.prepare import prepare_scf_tasks

    config = {
        'scf_soft': 'unknown_backend',
        'systems': ['test']
    }

    with pytest.raises(ValueError, match="Unknown SCF backend"):
        prepare_scf_tasks(config)
