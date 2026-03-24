"""ABACUS output parser.

This module parses ABACUS output files and extracts DeePKS results.
All quantities are read from the npy/csr files written by ABACUS
into the OUT.ABACUS directory of each frame calculation.

File layout (all under OUT.ABACUS/):
    deepks_etot.npy     - total energy (Hartree)
    deepks_ebase.npy    - baseline energy (Hartree)
    deepks_dm_eig.npy   - descriptor matrix, shape (natoms, ndesc)
    deepks_ftot.npy     - total forces (Hartree/Bohr)
    deepks_fbase.npy    - baseline forces (Hartree/Bohr)
    deepks_stot.npy     - total stress (Hartree/Bohr^3)
    deepks_sbase.npy    - baseline stress (Hartree/Bohr^3)
    deepks_otot.npy     - total bandgap-related (Hartree)
    deepks_obase.npy    - baseline bandgap-related (Hartree)
    deepks_orbpre.npy   - orbital precalculation
    deepks_htot.npy     - total Hamiltonian (gamma-only)
    deepks_hbase.npy    - baseline Hamiltonian (gamma-only)
    deepks_vdpre.npy    - v_delta precalc (deepks_v_delta==1)
    deepks_phialpha.npy - psi_alpha (deepks_v_delta==2)
    deepks_gevdm.npy    - grad_evdm (deepks_v_delta==2)
    deepks_gradvx.npy   - gradient of vx
    deepks_gvepsl.npy   - gradient of stress
    deepks_phialpha_r.npy - psi_alpha in R-space (k-point)
    deepks_hrtot.csr    - total Hamiltonian CSR (k-point)
    deepks_hrdelta.csr  - delta Hamiltonian CSR (k-point)

Convergence is stored in a ``conv`` text file (frame working directory).
Format: ``{frame_index} ... CONVERGED ...`` or ``{frame_index} ... NOT CONVERGED ...``.
The ``#`` character may appear as part of the ABACUS log prefix and is stripped.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

def check_convergence(work_dir: str) -> bool:
    """Check if SCF calculation converged by reading the ``conv`` file.

    The ``conv`` file is written by the run command::

        echo {frame_index}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv

    Its content looks like::

        0 ... #CONVERGED (NOT CONVERGED) ...

    Args:
        work_dir: Frame working directory (parent of OUT.ABACUS and ``conv``).

    Returns:
        True if the frame converged.
    """
    conv_file = os.path.join(work_dir, "conv")
    if not os.path.exists(conv_file):
        return False
    try:
        with open(conv_file) as f:
            tokens = f.read().split()
        tokens = [t.strip('#') for t in tokens]
        return "CONVERGED" in tokens and "NOT" not in tokens
    except Exception:
        return False


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _load_npy(out_dir: str, name: str) -> Optional[np.ndarray]:
    """Load a npy file from OUT.ABACUS if it exists."""
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# individual field parsers — all read npy files, no unit conversion
# ---------------------------------------------------------------------------

def parse_abacus_energy(out_dir: str) -> Optional[float]:
    """Read total energy from ``deepks_etot.npy`` (Hartree)."""
    arr = _load_npy(out_dir, "deepks_etot.npy")
    return float(arr.flat[0]) if arr is not None else None


def parse_abacus_base_energy(out_dir: str) -> Optional[float]:
    """Read baseline energy from ``deepks_ebase.npy`` (Hartree)."""
    arr = _load_npy(out_dir, "deepks_ebase.npy")
    return float(arr.flat[0]) if arr is not None else None


def parse_abacus_descriptor(out_dir: str, natoms: int) -> Optional[np.ndarray]:
    """Read descriptor matrix from ``deepks_dm_eig.npy``.

    Returns shape ``(natoms, ndesc)``.
    """
    arr = _load_npy(out_dir, "deepks_dm_eig.npy")
    if arr is None:
        return None
    if arr.ndim == 1 and arr.size % natoms == 0:
        arr = arr.reshape(natoms, -1)
    return arr


def parse_abacus_forces(out_dir: str, natoms: int) -> Optional[np.ndarray]:
    """Read total forces from ``deepks_ftot.npy`` (Hartree/Bohr)."""
    return _load_npy(out_dir, "deepks_ftot.npy")


def parse_abacus_base_forces(out_dir: str, natoms: int) -> Optional[np.ndarray]:
    """Read baseline forces from ``deepks_fbase.npy`` (Hartree/Bohr)."""
    return _load_npy(out_dir, "deepks_fbase.npy")


def parse_abacus_stress(out_dir: str) -> Optional[np.ndarray]:
    """Read total stress from ``deepks_stot.npy`` (Hartree/Bohr^3)."""
    return _load_npy(out_dir, "deepks_stot.npy")


def parse_abacus_base_stress(out_dir: str) -> Optional[np.ndarray]:
    """Read baseline stress from ``deepks_sbase.npy`` (Hartree/Bohr^3)."""
    return _load_npy(out_dir, "deepks_sbase.npy")


def parse_abacus_bandgap(out_dir: str) -> Optional[np.ndarray]:
    """Read total bandgap-related quantity from ``deepks_otot.npy``."""
    return _load_npy(out_dir, "deepks_otot.npy")


def parse_abacus_base_bandgap(out_dir: str) -> Optional[np.ndarray]:
    """Read baseline bandgap-related quantity from ``deepks_obase.npy``."""
    return _load_npy(out_dir, "deepks_obase.npy")


def parse_abacus_v_delta(out_dir: str, natoms: int) -> Optional[np.ndarray]:
    """Read v_delta precalc from ``deepks_vdpre.npy`` (deepks_v_delta==1)."""
    return _load_npy(out_dir, "deepks_vdpre.npy")


def parse_abacus_phialpha(out_dir: str) -> Optional[np.ndarray]:
    """Read psi_alpha from ``deepks_phialpha.npy`` (deepks_v_delta==2)."""
    return _load_npy(out_dir, "deepks_phialpha.npy")


def parse_abacus_gevdm(out_dir: str) -> Optional[np.ndarray]:
    """Read grad_evdm from ``deepks_gevdm.npy`` (deepks_v_delta==2)."""
    return _load_npy(out_dir, "deepks_gevdm.npy")


# ---------------------------------------------------------------------------
# unified entry point
# ---------------------------------------------------------------------------

def parse_abacus_output(work_dir: str,
                        fields: Optional[List[str]] = None,
                        natoms: Optional[int] = None) -> Dict[str, Any]:
    """Parse one ABACUS frame directory and return a results dict.

    Reads npy files from ``<work_dir>/OUT.ABACUS/``.
    Convergence is read from ``<work_dir>/conv``.

    Supported field names (``fields=None`` returns all available):
        ``e_tot``, ``e_base``, ``dm_eig``, ``conv``,
        ``f_tot``, ``f_base``, ``s_tot``, ``s_base``,
        ``o_tot``, ``o_base``, ``v_delta``,
        ``phialpha``, ``gevdm``.

    Args:
        work_dir: Frame working directory (parent of OUT.ABACUS).
        fields:   Fields to extract; ``None`` means all.
        natoms:   Number of atoms (needed for dm_eig reshape).

    Returns:
        dict mapping field name to value (ndarray or scalar).
    """
    out_dir = os.path.join(work_dir, "OUT.ABACUS")
    if fields is None:
        fields = {'e_tot', 'e_base', 'dm_eig', 'conv',
                  'f_tot', 'f_base', 's_tot', 's_base',
                  'o_tot', 'o_base', 'v_delta', 'phialpha', 'gevdm'}
    else:
        fields = set(fields)

    results: Dict[str, Any] = {}

    if 'conv' in fields:
        results['conv'] = check_convergence(work_dir)

    if 'e_tot' in fields:
        v = parse_abacus_energy(out_dir)
        if v is not None:
            results['e_tot'] = v

    if 'e_base' in fields:
        v = parse_abacus_base_energy(out_dir)
        if v is not None:
            results['e_base'] = v

    if 'dm_eig' in fields:
        v = parse_abacus_descriptor(out_dir, natoms or 1)
        if v is not None:
            results['dm_eig'] = v

    if 'f_tot' in fields:
        v = parse_abacus_forces(out_dir, natoms or 1)
        if v is not None:
            results['f_tot'] = v

    if 'f_base' in fields:
        v = parse_abacus_base_forces(out_dir, natoms or 1)
        if v is not None:
            results['f_base'] = v

    if 's_tot' in fields:
        v = parse_abacus_stress(out_dir)
        if v is not None:
            results['s_tot'] = v

    if 's_base' in fields:
        v = parse_abacus_base_stress(out_dir)
        if v is not None:
            results['s_base'] = v

    if 'o_tot' in fields:
        v = parse_abacus_bandgap(out_dir)
        if v is not None:
            results['o_tot'] = v

    if 'o_base' in fields:
        v = parse_abacus_base_bandgap(out_dir)
        if v is not None:
            results['o_base'] = v

    if 'v_delta' in fields:
        v = parse_abacus_v_delta(out_dir, natoms or 1)
        if v is not None:
            results['v_delta'] = v

    if 'phialpha' in fields:
        v = parse_abacus_phialpha(out_dir)
        if v is not None:
            results['phialpha'] = v

    if 'gevdm' in fields:
        v = parse_abacus_gevdm(out_dir)
        if v is not None:
            results['gevdm'] = v

    return results
