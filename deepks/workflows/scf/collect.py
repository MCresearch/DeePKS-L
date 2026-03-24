"""SCF workflow - Collect stage.

This module handles the collection stage of SCF workflow:
- Parse ABACUS output files
- Extract energies, forces, descriptors, etc.
- Aggregate results into numpy arrays
- Save to dump directory
"""

import os
from dataclasses import asdict

import numpy as np

from deepks.physics.defaults import NAME_TYPE
from deepks.physics.backends.abacus.parser import parse_abacus_output
from deepks.io.utils import get_sys_name, load_sys_paths

from .types import SCFResult


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

    with open(f"{path}/type_map.raw") as fp:
        my_type_map = [NAME_TYPE[i] for i in fp.read().split()]

    atom_types = np.loadtxt(f"{path}/type.raw", ndmin=1).astype(int)
    atom_types = np.array([int(my_type_map[i - 1]) for i in atom_types])
    atom_types = atom_types.reshape(1, -1).repeat(nframes, axis=0)

    atom_data = np.insert(coords, 0, values=atom_types, axis=2)
    return atom_data


def _load_system_geometry(sys_path, lattice_vector, lattice_constant, coord_type):
    """Load atom/box arrays and normalize coordinates for dumping."""
    try:
        atom_data = np.load(f"{sys_path}/atom.npy")
    except FileNotFoundError:
        atom_data = coord_to_atom(sys_path)

    nframes = atom_data.shape[0]

    if os.path.isfile(f"{sys_path}/box.npy"):
        box_data = np.load(f"{sys_path}/box.npy")
    else:
        box_data = np.array([lattice_vector])
        box_data = box_data.reshape(1, 9).repeat(nframes, axis=0)

    box_data = box_data.reshape(nframes, 3, 3)

    if coord_type == "Direct":
        atom_data[:, :, 1:4] = np.matmul(atom_data[:, :, 1:4], box_data)

    atom_data[:, :, 1:4] *= lattice_constant
    box_data *= lattice_constant

    return atom_data, box_data


def _build_parser_fields(dump_fields, cal_force, cal_stress, deepks_bandgap, deepks_v_delta):
    """Translate workflow dump fields into backend parser fields."""
    parser_fields = {"conv"}

    if any(field in dump_fields for field in ("e_tot", "e_base")):
        parser_fields.update({"e_tot", "e_base"})
    if cal_force and any(field in dump_fields for field in ("f_tot", "f_base")):
        parser_fields.update({"f_tot", "f_base"})
    if cal_stress and any(field in dump_fields for field in ("s_tot", "s_base")):
        parser_fields.update({"s_tot", "s_base"})
    if "dm_eig" in dump_fields:
        parser_fields.add("dm_eig")
    if deepks_bandgap and "bandgap" in dump_fields:
        parser_fields.add("bandgap")
    if deepks_v_delta and "v_delta_precondition" in dump_fields:
        parser_fields.add("v_delta")

    return parser_fields


def _initialize_result_buffers(nframes, natoms):
    """Prepare per-system arrays lazily filled during frame parsing."""
    return {
        "conv": np.full((nframes, 1), False),
        "e_tot": None,
        "e_base": None,
        "f_tot": None,
        "f_base": None,
        "s_tot": None,
        "s_base": None,
        "dm_eig": None,
        "bandgap": None,
        "v_delta_precondition": None,
        "natoms": natoms,
        "nframes": nframes,
    }


def _store_scalar_field(buffers, key, frame_index, value):
    if value is None:
        return
    if buffers[key] is None:
        buffers[key] = np.zeros((buffers["nframes"], 1))
    buffers[key][frame_index] = value


def _store_tensor_field(buffers, key, frame_index, value, shape):
    if value is None:
        return
    if buffers[key] is None:
        buffers[key] = np.zeros((buffers["nframes"],) + shape)
    buffers[key][frame_index] = value


def _collect_system_frames(sys_path, dump_fields, parser_fields, natoms):
    """Collect all parsed frame-level outputs for one system."""
    frame_dirs = sorted(
        entry.path
        for entry in os.scandir(os.path.join(sys_path, "ABACUS"))
        if entry.is_dir() and entry.name.isdigit()
    )

    buffers = _initialize_result_buffers(len(frame_dirs), natoms)

    for frame_index, frame_dir in enumerate(frame_dirs):
        parsed = parse_abacus_output(frame_dir, fields=list(parser_fields), natoms=natoms)
        buffers["conv"][frame_index] = parsed.get("convergence", False)

        _store_scalar_field(buffers, "e_tot", frame_index, parsed.get("e_tot"))
        _store_scalar_field(buffers, "e_base", frame_index, parsed.get("e_base"))
        _store_scalar_field(buffers, "bandgap", frame_index, parsed.get("bandgap"))

        _store_tensor_field(buffers, "f_tot", frame_index, parsed.get("f_tot"), (natoms, 3))
        _store_tensor_field(buffers, "f_base", frame_index, parsed.get("f_base"), (natoms, 3))
        _store_tensor_field(buffers, "s_tot", frame_index, parsed.get("s_tot"), (3, 3))
        _store_tensor_field(buffers, "s_base", frame_index, parsed.get("s_base"), (3, 3))

        descriptor = parsed.get("dm_eig")
        if descriptor is not None:
            _store_tensor_field(buffers, "dm_eig", frame_index, descriptor, descriptor.shape)

        v_delta = parsed.get("v_delta")
        if v_delta is not None:
            _store_tensor_field(
                buffers,
                "v_delta_precondition",
                frame_index,
                v_delta,
                (natoms,),
            )

    save_results = {"conv": buffers["conv"]}
    for key in (
        "e_tot",
        "e_base",
        "f_tot",
        "f_base",
        "s_tot",
        "s_base",
        "dm_eig",
        "bandgap",
        "v_delta_precondition",
    ):
        if buffers[key] is not None and key in dump_fields:
            save_results[key] = buffers[key]

    return save_results



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
    if scf_soft.lower() == 'pyscf':
        raise NotImplementedError(
            "PySCF workflow not yet implemented in new architecture"
        )
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
    parser_fields = _build_parser_fields(
        dump_fields,
        cal_force,
        cal_stress,
        deepks_bandgap,
        deepks_v_delta,
    )

    os.makedirs(dump_dir, exist_ok=True)
    result = SCFResult(dump_dir=dump_dir)

    for sys_path, sys_name in zip(sys_paths, sys_names):
        sys_dump_dir = os.path.join(dump_dir, sys_name)
        os.makedirs(sys_dump_dir, exist_ok=True)

        atom_data, box_data = _load_system_geometry(
            sys_path,
            lattice_vector=lattice_vector,
            lattice_constant=lattice_constant,
            coord_type=coord_type,
        )
        nframes = atom_data.shape[0]
        natoms = atom_data.shape[1]

        save_results = _collect_system_frames(sys_path, dump_fields, parser_fields, natoms)
        save_results["atom"] = atom_data
        save_results["box"] = box_data.reshape(nframes, 9)

        for key, value in save_results.items():
            np.save(f"{sys_dump_dir}/{key}.npy", value)

        result.systems.append({
            'name': sys_name,
            'path': sys_dump_dir,
            'nframes': nframes,
            'natoms': natoms,
            'converged': int(save_results['conv'].sum())
        })

    total_frames = sum(system['nframes'] for system in result.systems)
    total_converged = sum(system['converged'] for system in result.systems)
    result.statistics = {
        'total_systems': len(systems),
        'total_frames': total_frames,
        'total_converged': total_converged,
        'convergence_rate': total_converged / total_frames if total_frames > 0 else 0.0,
    }

    return asdict(result)
