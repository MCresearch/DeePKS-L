"""SCF workflow - Prepare stage.

This module handles the preparation stage of SCF workflow:
- Create working directories for each system/frame
- Generate backend-specific input files (INPUT, STRU, KPT for ABACUS)
- Set up file links and shared resources
"""

import os
import numpy as np
from pathlib import Path
from collections import Counter

from deepks.utils import load_sys_paths, get_sys_name
from deepks.default import NAME_TYPE, TYPE_NAME
from deepks.orchestration.workflow.task import PythonTask


def coord_to_atom(path):
    """Convert coord.npy and type.raw to atom.npy format.

    Args:
        path: System directory path

    Returns:
        np.ndarray: Atom data with shape (nframes, natoms, 4)
                   First column is atom type, rest are coordinates
    """
    try:
        coords = np.load(f"{path}/coord.npy")
    except FileNotFoundError:
        raise FileNotFoundError(f"coord.npy not found in {path}")

    nframes = coords.shape[0]
    if coords.shape[2] != 3:
        raise ValueError("coord.npy should have shape (nframes, natoms, 3)")

    # Get type mapping
    with open(f"{path}/type_map.raw") as fp:
        my_type_map = [NAME_TYPE[i] for i in fp.read().split()]

    atom_types = np.loadtxt(f"{path}/type.raw", ndmin=1).astype(int)
    atom_types = np.array([int(my_type_map[i-1]) for i in atom_types])
    atom_types = atom_types.reshape(1, -1).repeat(nframes, axis=0)

    # Insert atom types as first column
    atom_data = np.insert(coords, 0, values=atom_types, axis=2)
    return atom_data


def prepare_abacus_input_files(systems, scf_args, orb_files, pp_files, proj_file):
    """Prepare ABACUS input files for all systems.

    This function creates INPUT, STRU, and KPT files for each frame
    in each system.

    Args:
        systems: List of system paths
        scf_args: ABACUS-specific arguments
        orb_files: Orbital files
        pp_files: Pseudopotential files
        proj_file: Projection files

    Returns:
        None (files are written to disk)
    """
    from deepks.pipelines.iterate.generator_abacus import (
        make_abacus_scf_input,
        make_abacus_scf_stru,
        make_abacus_scf_kpt
    )

    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]

    for i, sys_path in enumerate(sys_paths):
        # Load atom data
        try:
            atom_data = np.load(f"{sys_path}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_path)

        # Load box data if exists
        if os.path.isfile(f"{sys_path}/box.npy"):
            cell_data = np.load(f"{sys_path}/box.npy")
            if cell_data.shape != (atom_data.shape[0], 9):
                raise ValueError(
                    f"box.npy should have shape (nframes, 9), "
                    f"but got {cell_data.shape}!"
                )

        nframes = atom_data.shape[0]

        # Create ABACUS directory
        abacus_dir = f"{sys_path}/ABACUS"
        if not os.path.exists(abacus_dir):
            os.mkdir(abacus_dir)

        # Load system-specific config if exists
        scf_args_local = dict(scf_args)
        if os.path.exists(f"{sys_path}/group_scf_abacus.yaml"):
            from deepks.utils import load_yaml
            local_config = load_yaml(f"{sys_path}/group_scf_abacus.yaml")
            scf_args_local.update(local_config)

        # Process each frame
        for f in range(nframes):
            frame_dir = f"{abacus_dir}/{f}"
            if not os.path.exists(frame_dir):
                os.mkdir(frame_dir)

            # Get atom types and counts for this frame
            frame_data = atom_data[f]
            atoms = frame_data[:, 0]
            nta = Counter(atoms)  # {atom_type: count}

            # Prepare system data
            sys_data = {
                'atom_names': [TYPE_NAME[int(it)] for it in nta.keys()],
                'atom_numbs': list(nta.values()),
                'cells': np.array([scf_args_local["lattice_vector"]]),
                'coords': [frame_data[:, 1:]]
            }

            # Use box data if available
            if os.path.isfile(f"{sys_path}/box.npy"):
                sys_data['cells'] = [cell_data[f]]

            # Write STRU file
            with open(f"{frame_dir}/STRU", "w") as stru_file:
                stru_file.write(
                    make_abacus_scf_stru(sys_data, pp_files, scf_args_local)
                )

            # Write INPUT file
            with open(f"{frame_dir}/INPUT", "w") as input_file:
                input_file.write(make_abacus_scf_input(scf_args_local))

            # Write KPT file if needed
            if (scf_args_local.get("k_points") is not None or
                scf_args_local.get("gamma_only") is True):
                with open(f"{frame_dir}/KPT", "w") as kpt_file:
                    kpt_file.write(make_abacus_scf_kpt(scf_args_local))


def prepare_scf_tasks(config):
    """Prepare SCF tasks (Stage 1).

    This function creates the preparation task that generates all
    input files for SCF calculations.

    Args:
        config: Configuration dictionary

    Returns:
        PythonTask: Task that prepares input files
    """
    scf_soft = config.get('scf_soft', 'pyscf')

    if scf_soft.lower() == 'abacus':
        return prepare_scf_tasks_abacus(config)
    elif scf_soft.lower() == 'pyscf':
        # PySCF preparation (to be implemented or kept as is)
        raise NotImplementedError(
            "PySCF workflow not yet implemented in new architecture"
        )
    else:
        raise ValueError(f"Unknown SCF backend: {scf_soft}")


def prepare_scf_tasks_abacus(config):
    """Prepare ABACUS SCF tasks.

    Args:
        config: Configuration dictionary with ABACUS-specific parameters

    Returns:
        PythonTask: Preparation task
    """
    from deepks.utils import flat_file_list

    # Extract systems
    systems = config.get('systems', [])
    if not systems:
        raise ValueError("No systems specified in config")

    # Extract ABACUS-specific parameters
    scf_abacus = config.get('scf_abacus', {})

    # Get file paths
    orb_files = scf_abacus.get('orb_files', [])
    pp_files = scf_abacus.get('pp_files', [])
    proj_file = scf_abacus.get('proj_file', [])

    # Convert to absolute paths
    orb_files = [os.path.abspath(s) for s in flat_file_list(orb_files, sort=False)]
    pp_files = [os.path.abspath(s) for s in flat_file_list(pp_files, sort=False)]
    proj_file = [os.path.abspath(s) for s in flat_file_list(proj_file, sort=False)]

    # Prepare arguments for input file generation
    scf_args = dict(scf_abacus)
    scf_args['orb_files'] = orb_files
    scf_args['pp_files'] = pp_files
    scf_args['proj_file'] = proj_file

    # Create preparation task
    task = PythonTask(
        prepare_abacus_input_files,
        call_kwargs={
            'systems': systems,
            'scf_args': scf_args,
            'orb_files': orb_files,
            'pp_files': pp_files,
            'proj_file': proj_file
        },
        outlog="prepare.log",
        errlog="prepare.err",
        workdir='.'
    )

    return task
