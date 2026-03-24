"""ABACUS output parser.

This module parses ABACUS output files and extracts results.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List


def parse_abacus_energy(out_dir: str) -> Optional[float]:
    """Parse total energy from ABACUS output.

    Args:
        out_dir: Output directory (OUT.ABACUS)

    Returns:
        float: Total energy in eV, or None if not found
    """
    running_log = os.path.join(out_dir, "running_scf.log")

    if not os.path.exists(running_log):
        return None

    try:
        with open(running_log, 'r') as f:
            for line in f:
                if 'final etot is' in line.lower():
                    # Format: "final etot is    -123.456 eV"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'is' and i + 1 < len(parts):
                            return float(parts[i + 1])
    except Exception as e:
        print(f"Warning: Failed to parse energy from {running_log}: {e}")
        return None

    return None


def parse_abacus_forces(out_dir: str, natoms: int) -> Optional[np.ndarray]:
    """Parse atomic forces from ABACUS output.

    Args:
        out_dir: Output directory (OUT.ABACUS)
        natoms: Number of atoms

    Returns:
        np.ndarray: Forces array with shape (natoms, 3) in eV/Angstrom,
                   or None if not found
    """
    running_log = os.path.join(out_dir, "running_scf.log")

    if not os.path.exists(running_log):
        return None

    try:
        forces = []
        with open(running_log, 'r') as f:
            lines = f.readlines()

        # Find force section
        in_force_section = False
        for line in lines:
            if 'TOTAL-FORCE (eV/Angstrom)' in line:
                in_force_section = True
                continue

            if in_force_section:
                if line.strip() == '' or '---' in line:
                    break

                parts = line.split()
                if len(parts) >= 6:  # atom_type x y z fx fy fz
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    forces.append([fx, fy, fz])

        if len(forces) == natoms:
            return np.array(forces)

    except Exception as e:
        print(f"Warning: Failed to parse forces from {running_log}: {e}")
        return None

    return None


def parse_abacus_stress(out_dir: str) -> Optional[np.ndarray]:
    """Parse stress tensor from ABACUS output.

    Args:
        out_dir: Output directory (OUT.ABACUS)

    Returns:
        np.ndarray: Stress tensor with shape (3, 3) in kbar,
                   or None if not found
    """
    running_log = os.path.join(out_dir, "running_scf.log")

    if not os.path.exists(running_log):
        return None

    try:
        with open(running_log, 'r') as f:
            lines = f.readlines()

        # Find stress section
        in_stress_section = False
        stress_lines = []

        for line in lines:
            if 'TOTAL-STRESS (KBAR)' in line:
                in_stress_section = True
                continue

            if in_stress_section:
                if line.strip() == '' or '---' in line:
                    break

                parts = line.split()
                if len(parts) >= 3:
                    stress_lines.append([float(parts[0]), float(parts[1]), float(parts[2])])

        if len(stress_lines) == 3:
            return np.array(stress_lines)

    except Exception as e:
        print(f"Warning: Failed to parse stress from {running_log}: {e}")
        return None

    return None


def parse_abacus_descriptor(out_dir: str, natoms: int) -> Optional[np.ndarray]:
    """Parse DeepKS descriptor from ABACUS output.

    Args:
        out_dir: Output directory (OUT.ABACUS)
        natoms: Number of atoms

    Returns:
        np.ndarray: Descriptor matrix, or None if not found
    """
    dm_file = os.path.join(out_dir, "deepks.dm_eig")

    if not os.path.exists(dm_file):
        return None

    try:
        dm_eig = np.loadtxt(dm_file)
        return dm_eig
    except Exception as e:
        print(f"Warning: Failed to parse descriptor from {dm_file}: {e}")
        return None


def parse_abacus_bandgap(out_dir: str) -> Optional[float]:
    """Parse bandgap from ABACUS output.

    Args:
        out_dir: Output directory (OUT.ABACUS)

    Returns:
        float: Bandgap in eV, or None if not found
    """
    bandgap_file = os.path.join(out_dir, "deepks.bandgap")

    if not os.path.exists(bandgap_file):
        return None

    try:
        with open(bandgap_file, 'r') as f:
            line = f.readline().strip()
            return float(line)
    except Exception as e:
        print(f"Warning: Failed to parse bandgap from {bandgap_file}: {e}")
        return None


def parse_abacus_v_delta(out_dir: str, natoms: int) -> Optional[np.ndarray]:
    """Parse v_delta preconditioner from ABACUS output.

    Args:
        out_dir: Output directory (OUT.ABACUS)
        natoms: Number of atoms

    Returns:
        np.ndarray: v_delta array, or None if not found
    """
    v_delta_file = os.path.join(out_dir, "deepks.v_delta_precondition")

    if not os.path.exists(v_delta_file):
        return None

    try:
        v_delta = np.loadtxt(v_delta_file)
        return v_delta
    except Exception as e:
        print(f"Warning: Failed to parse v_delta from {v_delta_file}: {e}")
        return None


def check_convergence(work_dir: str) -> bool:
    """Check if SCF calculation converged.

    Args:
        work_dir: Working directory containing conv file

    Returns:
        bool: True if converged, False otherwise
    """
    conv_file = os.path.join(work_dir, "conv")

    if not os.path.exists(conv_file):
        return False

    try:
        with open(conv_file, 'r') as f:
            content = f.read().lower()
            return 'converge' in content
    except Exception:
        return False


def parse_abacus_output(work_dir: str, fields: Optional[List[str]] = None,
                       natoms: Optional[int] = None) -> Dict[str, Any]:
    """Parse ABACUS output files.

    Args:
        work_dir: Working directory containing ABACUS output
        fields: List of fields to extract (e.g., ['energy', 'forces'])
               If None, extract all available fields
        natoms: Number of atoms (required for forces, descriptor)

    Returns:
        dict: Parsed results with requested fields
    """
    out_dir = os.path.join(work_dir, "OUT.ABACUS")
    results = {}

    if fields is None:
        fields = ['energy', 'forces', 'stress', 'descriptor',
                 'bandgap', 'v_delta', 'convergence']

    # Parse convergence
    if 'convergence' in fields or 'conv' in fields:
        results['convergence'] = check_convergence(work_dir)

    # Parse energy
    if 'energy' in fields or 'e_tot' in fields or 'e_base' in fields:
        energy = parse_abacus_energy(out_dir)
        if energy is not None:
            results['energy'] = energy
            results['e_tot'] = energy
            results['e_base'] = energy

    # Parse forces
    if ('forces' in fields or 'f_tot' in fields or 'f_base' in fields) and natoms:
        forces = parse_abacus_forces(out_dir, natoms)
        if forces is not None:
            results['forces'] = forces
            results['f_tot'] = forces
            results['f_base'] = forces

    # Parse stress
    if ('stress' in fields or 's_tot' in fields or 's_base' in fields):
        stress = parse_abacus_stress(out_dir)
        if stress is not None:
            results['stress'] = stress
            results['s_tot'] = stress
            results['s_base'] = stress

    # Parse descriptor
    if ('descriptor' in fields or 'dm_eig' in fields) and natoms:
        descriptor = parse_abacus_descriptor(out_dir, natoms)
        if descriptor is not None:
            results['descriptor'] = descriptor
            results['dm_eig'] = descriptor

    # Parse bandgap
    if 'bandgap' in fields:
        bandgap = parse_abacus_bandgap(out_dir)
        if bandgap is not None:
            results['bandgap'] = bandgap

    # Parse v_delta
    if 'v_delta' in fields and natoms:
        v_delta = parse_abacus_v_delta(out_dir, natoms)
        if v_delta is not None:
            results['v_delta'] = v_delta

    return results
