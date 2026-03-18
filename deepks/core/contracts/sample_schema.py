"""Shared sample schema metadata for cross-layer data contracts."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SampleSchema:
    """Canonical key names and shape hints for model samples."""

    energy_key: str = "lb_e"
    descriptor_key: str = "eig"
    force_key: str = "lb_f"
    stress_key: str = "lb_s"
    orbital_key: str = "lb_o"
    hamiltonian_k_key: str = "lb_vd"
    hamiltonian_r_key: str = "lb_vdr"

    # Shape conventions are informational and used for schema checks.
    energy_shape: Tuple[str, ...] = ("nframe", "1")
    descriptor_shape: Tuple[str, ...] = ("nframe", "natom", "ndesc")
    force_shape: Tuple[str, ...] = ("nframe", "natom", "3")
    stress_shape: Tuple[str, ...] = ("nframe", "6")
