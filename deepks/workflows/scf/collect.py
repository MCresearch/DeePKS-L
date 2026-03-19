"""SCF workflow - Collect stage.

This module handles the collection stage of SCF workflow:
- Parse ABACUS output files
- Extract energies, forces, descriptors, etc.
- Aggregate results into numpy arrays
- Save to dump directory
"""

import os
import numpy as np
from collections import Counter

from deepks.utils import load_sys_paths, get_sys_name
from deepks.default import TYPE_NAME, NAME_TYPE


def coord_to_atom(path):
    """Convert coord.npy and type.raw to atom.npy format.

    Args:
        path: System directory path

    Returns:
        np.ndarray: Atom data with shape (nframes, natoms, 4)
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

    atom_data = np.insert(coords, 0, values=atom_types, axis=2)
    return atom_data


def collect_scf_results(prepare_task, config):
    """Collect SCF results (Stage 3).

    This function parses the output files and aggregates results.

    Args:
        prepare_task: Preparation task (not used, for consistency)
        config: Configuration dictionary

    Returns:
        dict: Results dictionary with paths and statistics
    """
    scf_soft = config.get('scf_soft', 'pyscf')

    if scf_soft.lower() == 'abacus':
        return collect_scf_results_abacus(config)
    elif scf_soft.lower() == 'pyscf':
        raise NotImplementedError(
            "PySCF workflow not yet implemented in new architecture"
        )
    else:
        raise ValueError(f"Unknown SCF backend: {scf_soft}")


def collect_scf_results_abacus(config):
    """Collect ABACUS SCF results.

    This function gathers statistics from ABACUS calculations and
    saves them to the dump directory.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Results with paths and statistics
    """
    systems = config.get('systems', [])
    dump_dir = config.get('dump_dir', 'scf_results')
    dump_fields = config.get('dump_fields', ['e_tot', 'dm_eig', 'conv'])

    scf_abacus = config.get('scf_abacus', {})
    cal_force = scf_abacus.get('cal_force', 0)
    cal_stress = scf_abacus.get('cal_stress', 0)
    deepks_bandgap = scf_abacus.get('deepks_bandgap', 0)
    deepks_v_delta = scf_abacus.get('deepks_v_delta', 0)
    lattice_constant = scf_abacus.get('lattice_constant', 1.0)
    coord_type = scf_abacus.get('coord_type', 'Cartesian')
    lattice_vector = scf_abacus.get('lattice_vector', np.eye(3))

    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
    sys_names = [os.path.basename(get_sys_name(s)) for s in sys_paths]

    # Create dump directory
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    results = {
        'dump_dir': dump_dir,
        'systems': [],
        'statistics': {}
    }

    # Process each system
    for i, sys_path in enumerate(sys_paths):
        sys_name = sys_names[i]
        sys_dump_dir = os.path.join(dump_dir, sys_name)

        if not os.path.exists(sys_dump_dir):
            os.makedirs(sys_dump_dir)

        # Load atom data
        try:
            atom_data = np.load(f"{sys_path}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_path)

        nframes = atom_data.shape[0]
        natoms = atom_data.shape[1]

        # Load box data if exists
        if os.path.isfile(f"{sys_path}/box.npy"):
            box_data = np.load(f"{sys_path}/box.npy")
        else:
            box_data = np.array([lattice_vector])
            box_data = box_data.reshape(1, 9).repeat(nframes, axis=0)

        box_data = box_data.reshape(nframes, 3, 3)

        # Convert coordinates to Bohr if needed
        if coord_type == "Direct":
            atom_data[:, :, 1:4] = np.matmul(atom_data[:, :, 1:4], box_data)

        atom_data[:, :, 1:4] *= lattice_constant
        box_data *= lattice_constant

        # Initialize result arrays
        conv = np.full((nframes, 1), False)
        dm_eig = None
        e_base = None
        f_base = None
        s_base = None
        e_tot = None
        f_tot = None
        s_tot = None
        bandgap = None
        v_delta_precondition = None

        # Parse results for each frame
        for f in range(nframes):
            frame_dir = f"{sys_path}/ABACUS/{f}"
            out_dir = f"{frame_dir}/OUT.ABACUS"

            # Check convergence
            conv_file = f"{frame_dir}/conv"
            if os.path.exists(conv_file):
                with open(conv_file, 'r') as cf:
                    conv_line = cf.read().strip()
                    if 'converge' in conv_line.lower():
                        conv[f] = True

            # Parse energy
            if 'e_tot' in dump_fields or 'e_base' in dump_fields:
                energy = parse_abacus_energy(out_dir)
                if energy is not None:
                    if e_tot is None:
                        e_tot = np.zeros((nframes, 1))
                        e_base = np.zeros((nframes, 1))
                    e_tot[f] = energy
                    e_base[f] = energy

            # Parse forces
            if cal_force and ('f_tot' in dump_fields or 'f_base' in dump_fields):
                forces = parse_abacus_forces(out_dir, natoms)
                if forces is not None:
                    if f_tot is None:
                        f_tot = np.zeros((nframes, natoms, 3))
                        f_base = np.zeros((nframes, natoms, 3))
                    f_tot[f] = forces
                    f_base[f] = forces

            # Parse stress
            if cal_stress and ('s_tot' in dump_fields or 's_base' in dump_fields):
                stress = parse_abacus_stress(out_dir)
                if stress is not None:
                    if s_tot is None:
                        s_tot = np.zeros((nframes, 3, 3))
                        s_base = np.zeros((nframes, 3, 3))
                    s_tot[f] = stress
                    s_base[f] = stress

            # Parse descriptor (dm_eig)
            if 'dm_eig' in dump_fields:
                descriptor = parse_abacus_descriptor(out_dir)
                if descriptor is not None:
                    if dm_eig is None:
                        # Initialize with first frame's shape
                        dm_eig = np.zeros((nframes,) + descriptor.shape)
                    dm_eig[f] = descriptor

            # Parse bandgap
            if deepks_bandgap and 'bandgap' in dump_fields:
                bg = parse_abacus_bandgap(out_dir)
                if bg is not None:
                    if bandgap is None:
                        bandgap = np.zeros((nframes, 1))
                    bandgap[f] = bg

            # Parse v_delta_precondition
            if deepks_v_delta and 'v_delta_precondition' in dump_fields:
                v_delta = parse_abacus_v_delta(out_dir, natoms)
                if v_delta is not None:
                    if v_delta_precondition is None:
                        v_delta_precondition = np.zeros((nframes, natoms))
                    v_delta_precondition[f] = v_delta

        # Save results
        save_results = {
            'atom': atom_data,
            'box': box_data.reshape(nframes, 9),
            'conv': conv
        }

        if e_tot is not None and 'e_tot' in dump_fields:
            save_results['e_tot'] = e_tot
        if e_base is not None and 'e_base' in dump_fields:
            save_results['e_base'] = e_base
        if f_tot is not None and 'f_tot' in dump_fields:
            save_results['f_tot'] = f_tot
        if f_base is not None and 'f_base' in dump_fields:
            save_results['f_base'] = f_base
        if s_tot is not None and 's_tot' in dump_fields:
            save_results['s_tot'] = s_tot
        if s_base is not None and 's_base' in dump_fields:
            save_results['s_base'] = s_base
        if dm_eig is not None and 'dm_eig' in dump_fields:
            save_results['dm_eig'] = dm_eig
        if bandgap is not None and 'bandgap' in dump_fields:
            save_results['bandgap'] = bandgap
        if v_delta_precondition is not None and 'v_delta_precondition' in dump_fields:
            save_results['v_delta_precondition'] = v_delta_precondition

        # Save to files
        for key, value in save_results.items():
            np.save(f"{sys_dump_dir}/{key}.npy", value)

        results['systems'].append({
            'name': sys_name,
            'path': sys_dump_dir,
            'nframes': nframes,
            'natoms': natoms,
            'converged': int(conv.sum())
        })

    # Compute overall statistics
    total_frames = sum(s['nframes'] for s in results['systems'])
    total_converged = sum(s['converged'] for s in results['systems'])

    results['statistics'] = {
        'total_systems': len(systems),
        'total_frames': total_frames,
        'total_converged': total_converged,
        'convergence_rate': total_converged / total_frames if total_frames > 0 else 0.0
    }

    return results


def parse_abacus_energy(out_dir):
    """Parse total energy from ABACUS output.

    Args:
        out_dir: ABACUS output directory

    Returns:
        float: Total energy in eV, or None if not found
    """
    running_log = f"{out_dir}/running_scf.log"
    if not os.path.exists(running_log):
        return None

    try:
        with open(running_log, 'r') as f:
            for line in f:
                if 'final etot is' in line.lower():
                    # Example: "final etot is  -123.456 eV"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'is' and i + 1 < len(parts):
                            return float(parts[i + 1])
    except Exception:
        pass

    return None


def parse_abacus_forces(out_dir, natoms):
    """Parse forces from ABACUS output.

    Args:
        out_dir: ABACUS output directory
        natoms: Number of atoms

    Returns:
        np.ndarray: Forces with shape (natoms, 3), or None if not found
    """
    # ABACUS outputs forces in running_scf.log or separate file
    # Implementation depends on ABACUS version
    # Placeholder for now
    return None


def parse_abacus_stress(out_dir):
    """Parse stress tensor from ABACUS output.

    Args:
        out_dir: ABACUS output directory

    Returns:
        np.ndarray: Stress tensor with shape (3, 3), or None if not found
    """
    # Placeholder
    return None


def parse_abacus_descriptor(out_dir):
    """Parse descriptor (dm_eig) from ABACUS output.

    Args:
        out_dir: ABACUS output directory

    Returns:
        np.ndarray: Descriptor array, or None if not found
    """
    # Look for deepks descriptor file
    desc_file = f"{out_dir}/deepks_descriptor.dat"
    if os.path.exists(desc_file):
        try:
            return np.loadtxt(desc_file)
        except Exception:
            pass

    return None


def parse_abacus_bandgap(out_dir):
    """Parse bandgap from ABACUS output.

    Args:
        out_dir: ABACUS output directory

    Returns:
        float: Bandgap in eV, or None if not found
    """
    # Placeholder
    return None


def parse_abacus_v_delta(out_dir, natoms):
    """Parse v_delta_precondition from ABACUS output.

    Args:
        out_dir: ABACUS output directory
        natoms: Number of atoms

    Returns:
        np.ndarray: v_delta with shape (natoms,), or None if not found
    """
    # Placeholder
    return None
