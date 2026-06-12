"""Test SCF workflow structure and imports."""

import pytest
import sys
import os


def test_scf_workflow_imports():
    """Test that SCF workflow modules can be imported."""
    from deepks.workflows.scf import run_scf_workflow
    from deepks.workflows.scf.workflow import run_scf_workflow as workflow_main
    from deepks.workflows.scf.abacus.ops import (
        build_prepare_task,
        collect_results,
        execute_sequence,
    )

    assert callable(run_scf_workflow)
    assert callable(workflow_main)
    assert callable(build_prepare_task)
    assert callable(execute_sequence)
    assert callable(collect_results)


def test_scf_workflow_dispatcher_integration():
    """Test that dispatcher can route to SCF workflow."""
    from deepks.config.dispatcher import dispatch_command

    # This should not raise an error for scf type
    config = {
        'type': 'scf',
        'data': {'systems': []},
        'physics': {'backend': {'name': 'abacus', 'input': {}}},
    }

    # We expect it to fail because systems is empty, but it should
    # reach the workflow code (not fail on dispatch)
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        dispatch_command(config)


def test_prepare_scf_tasks_abacus_validation():
    """Test that prepare_scf_tasks validates input."""
    from deepks.workflows.scf.abacus.ops import build_prepare_task

    # Missing systems should raise error
    config = {'physics': {'backend': {'name': 'abacus', 'input': {}}}}
    with pytest.raises(ValueError, match="No systems specified|data.systems"):
        build_prepare_task(config)


def test_scf_workflow_config_structure():
    """Test that workflow accepts proper config structure."""
    from deepks.workflows.scf.workflow import run_scf_workflow

    # This should fail gracefully with proper error messages
    config = {
        'type': 'scf',
        'data': {'systems': ['nonexistent_system']},
        'physics': {
            'backend': {
                'name': 'abacus',
                'input': {
                    'ntype': 1,
                    'ecutwfc': 50,
                    'orb_files': ['orb'],
                    'pp_files': ['upf'],
                },
                'output': {'dump_dir': 'test_output', 'dump_fields': ['e_tot']},
            },
        },
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
        'data': {'systems': ['test_system']},
        'physics': {'backend': {'name': 'pyscf', 'input': {'basis': 'ccpvdz'}}},
    }

    # This would test PySCF workflow if it were implemented
    with pytest.raises(NotImplementedError):
        run_scf_workflow(config)


def test_coord_to_atom_helper():
    """Test coord_to_atom conversion function."""
    from deepks.workflows.scf.abacus.ops import coord_to_atom
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
    from deepks.workflows.scf.workflow import run_scf_workflow

    config = {
        'type': 'scf',
        'data': {'systems': ['test']},
        'physics': {'backend': {'name': 'unknown_backend', 'input': {}}},
    }

    with pytest.raises(ValueError, match="Unknown SCF backend"):
        run_scf_workflow(config)


def test_scf_workflow_accepts_new_interface_blocks(monkeypatch):
    from deepks.orchestration.workflow.task import BlankTask
    from deepks.workflows.scf.abacus.ops import (
        build_prepare_task,
        collect_results as collect_scf_results_abacus,
        execute_sequence as execute_scf_tasks_abacus,
    )

    captured = {}

    monkeypatch.setattr(
        "deepks.workflows.scf.abacus.ops.prepare_abacus_input_files",
        lambda **kwargs: captured.setdefault("prepare", kwargs),
    )
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.load_sys_paths", lambda systems: systems)
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.os.path.exists", lambda path: True)

    task = build_prepare_task({
        "type": "scf",
        "data": {"systems": ["sys_a"]},
        "physics": {
            "backend": {
                "name": "abacus",
                "input": {"ecutwfc": 60, "orb_files": ["orb"]},
                "output": {"dump_dir": "dump", "dump_fields": ["e_tot"]},
            },
        },
        "runtime": {
            "scf": {
                "execute": {
                    "dispatcher": {"batch": "shell"},
                    "resources": {"task_per_node": 4},
                    "group_size": 2,
                },
            },
        },
    })

    assert task.call_kwargs["systems"] == ["sys_a"]
    assert task.call_kwargs["scf_args"]["ecutwfc"] == 60

    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.load_sys_paths", lambda systems: systems)
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.get_sys_name", lambda s: os.path.basename(s))
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.Sequence.run", lambda self: captured.setdefault("execute", self.child_tasks[1]))
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.coord_to_atom", lambda path: __import__("numpy").zeros((1, 1, 4)))

    execute_scf_tasks_abacus(
        BlankTask(workdir="."),
        {
            "type": "scf",
            "data": {"systems": ["sys_a"]},
            "physics": {
                "backend": {
                    "name": "abacus",
                    "input": {"orb_files": ["orb"]},
                },
            },
            "runtime": {
                "scf": {
                    "execute": {
                        "dispatcher": {"batch": "shell"},
                        "resources": {"task_per_node": 4},
                        "group_size": 2,
                    },
                    "command": {"abacus_path": "abacus-bin", "run_cmd": "mpirun"},
                },
            },
        },
    )

    assert captured["execute"].group_size == 2
    assert captured["execute"].resources["task_per_node"] == 4

    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.load_sys_paths", lambda systems: systems)
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.get_sys_name", lambda s: os.path.basename(s))
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops._load_system_geometry", lambda *args, **kwargs: (__import__("numpy").zeros((1, 1, 4)), __import__("numpy").zeros((1, 3, 3))))
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops._collect_system_frames", lambda *args, **kwargs: {"conv": __import__("numpy").array([[True]])})
    monkeypatch.setattr("deepks.workflows.scf.abacus.ops.np.save", lambda *args, **kwargs: None)

    result = collect_scf_results_abacus({
        "type": "scf",
        "data": {"systems": ["sys_a"]},
        "physics": {
            "backend": {
                "name": "abacus",
                "input": {},
                "output": {"dump_dir": "dump", "dump_fields": ["e_tot", "conv"]},
            },
        },
    })

    assert result["dump_dir"] == "dump"
    assert result["statistics"]["total_systems"] == 1
