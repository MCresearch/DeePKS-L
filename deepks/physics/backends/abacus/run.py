"""ABACUS SCF calculation runner."""

import os
import sys
import subprocess
import numpy as np
from deepks.io.utils import load_sys_paths, get_sys_name, get_with_prefix, load_array
from deepks.physics.backends.constants import DEFAULT_DUMP_FIELDS


def load_system_data(path):
    """Load system data from various formats.

    Supports:
    - atom.npy: (nframes, natoms, 4) array with [element_num, x, y, z]
    - coord.npy + type.raw: coordinates and element types separately
    - xyz files: standard xyz format

    Args:
        path: System path (directory or xyz file)

    Returns:
        frames: List of atom lists, each atom is [element, [x, y, z]]
        lattice_info: Dict with 'lattice_vector', 'lattice_constant' if available
    """
    from deepks.physics.constants import TYPE_NAME

    lattice_info = {}

    # Check if path is xyz file
    if path.endswith('.xyz'):
        from deepks.physics.backends.abacus.utils import parse_xyz
        frames_data = parse_xyz(path)
        frames = []
        for frame in frames_data:
            atom_list = []
            for atom_line in frame['atoms']:
                parts = atom_line.split()
                elem = parts[0]
                coord = [float(parts[1]), float(parts[2]), float(parts[3])]
                atom_list.append([elem, coord])
            frames.append(atom_list)
        return frames, lattice_info

    # Try to load from npy files
    try:
        # Try atom.npy format (nframes, natoms, 4)
        atom_array = load_array(get_with_prefix("atom", path, prefer=".npy"))
        assert len(atom_array.shape) == 3 and atom_array.shape[2] == 4, atom_array.shape
        nframes = atom_array.shape[0]
        elements = np.rint(atom_array[:, :, 0]).astype(int)
        coords = atom_array[:, :, 1:]
    except FileNotFoundError:
        # Try coord.npy + type.raw format
        coords = load_array(get_with_prefix("coord", path, prefer=".npy"))
        assert len(coords.shape) == 3 and coords.shape[2] == 3, coords.shape
        nframes = coords.shape[0]
        elements = np.loadtxt(os.path.join(path, "type.raw"), dtype=str)\
                     .reshape(1, -1).repeat(nframes, axis=0)

    # Try to load lattice information
    try:
        cell_file = os.path.join(path, "cell.npy")
        if os.path.exists(cell_file):
            cell = np.load(cell_file)
            if cell.ndim == 2:  # Single frame
                lattice_info['lattice_vector'] = cell
            elif cell.ndim == 3:  # Multiple frames, use first
                lattice_info['lattice_vector'] = cell[0]
    except Exception:
        pass

    # Convert to atom list format
    frames = []
    for i in range(nframes):
        atom_list = []
        for e, c in zip(elements[i], coords[i]):
            # Convert element number to symbol if needed
            if isinstance(e, (int, np.integer)):
                elem = TYPE_NAME.get(int(e), 'X')
            else:
                elem = str(e)
            atom_list.append([elem, c])
        frames.append(atom_list)

    return frames, lattice_info


def write_stru_file(path, atom, lattice_vector, lattice_constant,
                    coord_type, orb_files, pp_files, proj_file):
    """Write ABACUS STRU file.

    Args:
        path: Working directory
        atom: List of [element, [x, y, z]] pairs
        lattice_vector: 3x3 lattice vector matrix
        lattice_constant: Lattice constant in Bohr
        coord_type: 'Cartesian' or 'Direct'
        orb_files: List of orbital files for each element
        pp_files: List of pseudopotential files for each element
        proj_file: List of projection files for DeePKS
    """
    with open(os.path.join(path, "STRU"), 'w') as f:
        f.write("ATOMIC_SPECIES\n")
        # Extract unique elements
        elements = []
        seen = set()
        for a in atom:
            elem = a[0]
            if elem not in seen:
                elements.append(elem)
                seen.add(elem)

        for i, elem in enumerate(elements):
            pp_file = pp_files[i] if i < len(pp_files) else pp_files[0]
            f.write(f"{elem} 1.0 {pp_file}\n")

        f.write(f"\nLATTICE_CONSTANT\n{lattice_constant}\n")
        f.write("\nLATTICE_VECTORS\n")
        for vec in lattice_vector:
            f.write(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n")

        f.write(f"\nATOMIC_POSITIONS\n{coord_type}\n")
        for elem in elements:
            elem_atoms = [a for a in atom if a[0] == elem]
            f.write(f"\n{elem}\n0.0\n{len(elem_atoms)}\n")
            for a in elem_atoms:
                coord = a[1]
                f.write(f"{coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f} 1 1 1\n")

        # Write numerical orbital files
        f.write("\nNUMERICAL_ORBITAL\n")
        for i, elem in enumerate(elements):
            orb_file = orb_files[i] if i < len(orb_files) else orb_files[0]
            f.write(f"{orb_file}\n")

        # Write DeePKS descriptor files
        if proj_file:
            f.write("\nNUMERICAL_DESCRIPTOR\n")
            for pf in proj_file:
                f.write(f"{pf}\n")


def write_input_file(path, **abacus_args):
    """Write ABACUS INPUT file.

    Args:
        path: Working directory
        **abacus_args: ABACUS-specific arguments
    """
    with open(os.path.join(path, "INPUT"), 'w') as f:
        f.write("INPUT_PARAMETERS\n")

        # Basic settings
        f.write(f"calculation scf\n")
        f.write(f"basis_type {abacus_args.get('basis_type', 'lcao')}\n")
        f.write(f"nspin {abacus_args.get('nspin', 1)}\n")
        f.write(f"symmetry {abacus_args.get('symmetry', 0)}\n")

        # DFT settings
        f.write(f"dft_functional {abacus_args.get('dft_functional', 'pbe')}\n")
        f.write(f"ecutwfc {abacus_args.get('ecutwfc', 50)}\n")

        # SCF settings
        f.write(f"scf_thr {abacus_args.get('scf_thr', 1e-7)}\n")
        f.write(f"scf_nmax {abacus_args.get('scf_nmax', 50)}\n")
        f.write(f"mixing_type {abacus_args.get('mixing_type', 'pulay')}\n")
        f.write(f"mixing_beta {abacus_args.get('mixing_beta', 0.4)}\n")

        # K-points
        if abacus_args.get('gamma_only', 1):
            f.write(f"gamma_only 1\n")
        elif abacus_args.get('kspacing'):
            f.write(f"kspacing {abacus_args['kspacing']}\n")

        # Smearing
        f.write(f"smearing_method {abacus_args.get('smearing_method', 'gaussian')}\n")
        f.write(f"smearing_sigma {abacus_args.get('smearing_sigma', 0.02)}\n")

        # Force and stress
        f.write(f"cal_force {abacus_args.get('cal_force', 0)}\n")
        f.write(f"cal_stress {abacus_args.get('cal_stress', 0)}\n")

        # DeePKS settings
        f.write(f"deepks_out_labels {abacus_args.get('deepks_out_labels', 1)}\n")
        f.write(f"deepks_scf {abacus_args.get('deepks_scf', 0)}\n")
        f.write(f"deepks_bandgap {abacus_args.get('deepks_bandgap', 0)}\n")
        f.write(f"deepks_v_delta {abacus_args.get('deepks_v_delta', 0)}\n")

        # Output settings
        f.write(f"out_wfc_lcao {abacus_args.get('out_wfc_lcao', 0)}\n")

        # Bands
        if abacus_args.get('nbands'):
            f.write(f"nbands {abacus_args['nbands']}\n")


def write_kpt_file(path, k_points):
    """Write ABACUS KPT file.

    Args:
        path: Working directory
        k_points: List of [kx, ky, kz] or None
    """
    if k_points is None:
        return

    with open(os.path.join(path, "KPT"), 'w') as f:
        f.write("K_POINTS\n0\n")
        f.write("Gamma\n")
        f.write(f"{k_points[0]} {k_points[1]} {k_points[2]}\n")
        f.write("0 0 0\n")


def run_abacus(work_dir, abacus_path, run_cmd, verbose=0):
    """Run ABACUS calculation.

    Args:
        work_dir: Working directory
        abacus_path: Path to ABACUS executable
        run_cmd: MPI run command (e.g., 'mpirun -np 8')
        verbose: Verbosity level

    Returns:
        bool: True if successful, False otherwise
    """
    cmd = f"{run_cmd} {abacus_path}"

    if verbose:
        print(f"Running ABACUS in {work_dir}")
        print(f"Command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"ABACUS failed with return code {result.returncode}", file=sys.stderr)
            if verbose > 1:
                print(f"stderr: {result.stderr}", file=sys.stderr)
            return False

        if verbose > 1:
            print(f"stdout: {result.stdout}")

        return True

    except subprocess.TimeoutExpired:
        print(f"ABACUS calculation timed out in {work_dir}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ABACUS calculation failed: {e}", file=sys.stderr)
        return False


def parse_abacus_output(work_dir, dump_fields):
    """Parse ABACUS output files.

    Args:
        work_dir: Working directory
        dump_fields: List of fields to extract

    Returns:
        dict: Parsed results
    """
    results = {}

    # Parse energy
    if "e_tot" in dump_fields or "e_base" in dump_fields:
        try:
            with open(os.path.join(work_dir, "OUT.ABACUS/running_scf.log")) as f:
                for line in f:
                    if "final etot is" in line:
                        energy = float(line.split()[-2])
                        results["e_tot"] = np.array([energy])
                        results["e_base"] = np.array([energy])
                        break
        except FileNotFoundError:
            print(f"Warning: running_scf.log not found in {work_dir}", file=sys.stderr)

    # Parse convergence
    if "conv" in dump_fields:
        try:
            with open(os.path.join(work_dir, "OUT.ABACUS/running_scf.log")) as f:
                converged = False
                for line in f:
                    if "convergence has been achieved" in line or "charge density convergence is achieved" in line:
                        converged = True
                        break
                results["conv"] = np.array([converged])
        except FileNotFoundError:
            results["conv"] = np.array([False])

    # Parse descriptor (dm_eig)
    if "dm_eig" in dump_fields:
        desc_file = os.path.join(work_dir, "OUT.ABACUS/deepks.descriptor")
        if os.path.exists(desc_file):
            results["dm_eig"] = np.loadtxt(desc_file)
        else:
            print(f"Warning: descriptor file not found in {work_dir}", file=sys.stderr)

    # Parse forces
    if "f_tot" in dump_fields or "force" in dump_fields:
        try:
            forces = []
            with open(os.path.join(work_dir, "OUT.ABACUS/running_scf.log")) as f:
                in_force_section = False
                for line in f:
                    if "TOTAL-FORCE (eV/Angstrom)" in line:
                        in_force_section = True
                        next(f)  # Skip separator line
                        continue
                    if in_force_section:
                        if line.strip() == "" or "---" in line:
                            break
                        parts = line.split()
                        if len(parts) >= 6:  # atom_type x y z fx fy fz
                            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                            forces.append([fx, fy, fz])
            if forces:
                results["f_tot"] = np.array(forces)
                results["force"] = np.array(forces)
        except FileNotFoundError:
            pass

    # Parse stress
    if "stress" in dump_fields:
        try:
            with open(os.path.join(work_dir, "OUT.ABACUS/running_scf.log")) as f:
                for line in f:
                    if "TOTAL-STRESS (KBAR)" in line:
                        next(f)  # Skip separator
                        stress_lines = [next(f) for _ in range(3)]
                        stress = []
                        for sline in stress_lines:
                            parts = sline.split()
                            stress.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        results["stress"] = np.array(stress)
                        break
        except (FileNotFoundError, StopIteration):
            pass

    # Parse atom information
    if "atom" in dump_fields:
        # TODO: Parse atom information from STRU file
        pass

    return results


def main(systems, model_file=None, proj_basis=None, device="cpu",
         dump_dir=".", dump_fields=None, group=False, verbose=0,
         **abacus_args):
    """Run ABACUS SCF calculations.

    Args:
        systems: List of system paths
        model_file: Path to DeePKS model (for future integration)
        proj_basis: Projection basis files
        device: Computation device (not used for ABACUS)
        dump_dir: Output directory
        dump_fields: Fields to output
        group: Whether to group results
        verbose: Verbosity level
        **abacus_args: ABACUS-specific arguments
    """
    if dump_fields is None:
        dump_fields = list(DEFAULT_DUMP_FIELDS)

    # Get ABACUS parameters
    abacus_path = abacus_args.get('abacus_path', '/usr/local/bin/ABACUS.mpi')
    run_cmd = abacus_args.get('run_cmd', 'mpirun')
    orb_files = abacus_args.get('orb_files', ['orb'])
    pp_files = abacus_args.get('pp_files', ['upf'])
    proj_file = abacus_args.get('proj_file', ['orb']) if proj_basis else None

    lattice_vector = abacus_args.get('lattice_vector', np.eye(3))
    lattice_constant = abacus_args.get('lattice_constant', 1)
    coord_type = abacus_args.get('coord_type', 'Cartesian')
    k_points = abacus_args.get('k_points')

    systems = load_sys_paths(systems)

    for sys_path in systems:
        sys_name = get_sys_name(os.path.basename(sys_path.rstrip(os.path.sep)))
        work_dir = os.path.join(dump_dir, sys_name)
        os.makedirs(work_dir, exist_ok=True)

        if verbose:
            print(f"Processing system: {sys_path}")

        try:
            # Load system data using unified loader
            frames, lattice_info = load_system_data(sys_path)

            # Use lattice from data if available, otherwise use parameters
            if 'lattice_vector' in lattice_info:
                lattice_vector = lattice_info['lattice_vector']
            else:
                lattice_vector = abacus_args.get('lattice_vector', np.eye(3))

            if 'lattice_constant' in lattice_info:
                lattice_constant = lattice_info['lattice_constant']
            else:
                lattice_constant = abacus_args.get('lattice_constant', 1)

            # Process each frame
            for frame_idx, atom in enumerate(frames):
                if len(frames) > 1:
                    frame_dir = os.path.join(work_dir, f"frame_{frame_idx}")
                    os.makedirs(frame_dir, exist_ok=True)
                else:
                    frame_dir = work_dir

                if verbose and len(frames) > 1:
                    print(f"  Frame {frame_idx}/{len(frames)}")

                # Write ABACUS input files
                write_stru_file(frame_dir, atom, lattice_vector, lattice_constant,
                               coord_type, orb_files, pp_files, proj_file)
                write_input_file(frame_dir, **abacus_args)
                write_kpt_file(frame_dir, k_points)

                # Run ABACUS
                success = run_abacus(frame_dir, abacus_path, run_cmd, verbose)

                if not success:
                    print(f"ABACUS calculation failed for {sys_path} frame {frame_idx}", file=sys.stderr)
                    continue

                # Parse and save results
                results = parse_abacus_output(frame_dir, dump_fields)

                # Save results
                for field, data in results.items():
                    if len(frames) > 1:
                        # For multiple frames, save in frame directory
                        np.save(os.path.join(frame_dir, f"{field}.npy"), data)
                    else:
                        # For single frame, save in system directory
                        np.save(os.path.join(work_dir, f"{field}.npy"), data)

            if verbose:
                print(f"Finished {sys_path}")

        except Exception as e:
            print(f"Error processing {sys_path}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    if verbose:
        print("All ABACUS calculations completed")


if __name__ == "__main__":
    # For testing
    pass
