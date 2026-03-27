"""Physics backend base interface.

This module defines the abstract base class for all physics backends.
Each backend (PySCF, ABACUS) must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class PhysicsBackend(ABC):
    """Abstract base class for physics calculation backends.

    A physics backend handles:
    1. Input file generation
    2. Calculation execution
    3. Output parsing

    Each backend must implement these core methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize backend with configuration.

        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def generate_input(self, system_data: Dict[str, Any],
                      output_dir: str, **kwargs) -> None:
        """Generate input files for calculation.

        Args:
            system_data: System information (atoms, coords, cell, etc.)
            output_dir: Directory to write input files
            **kwargs: Backend-specific parameters

        Returns:
            None (files are written to disk)
        """
        pass

    @abstractmethod
    def run_calculation(self, work_dir: str, **kwargs) -> Dict[str, Any]:
        """Run the calculation.

        Args:
            work_dir: Working directory containing input files
            **kwargs: Runtime parameters (nproc, timeout, etc.)

        Returns:
            dict: Execution status and metadata
        """
        pass

    @abstractmethod
    def parse_output(self, work_dir: str,
                    fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse calculation output files.

        Args:
            work_dir: Working directory containing output files
            fields: List of fields to extract (e.g., ['energy', 'forces'])

        Returns:
            dict: Parsed results with requested fields
        """
        pass

    def validate_config(self) -> bool:
        """Validate backend configuration.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        return True

    def get_required_files(self) -> List[str]:
        """Get list of required input files.

        Returns:
            list: List of required file names
        """
        return []

    def get_output_files(self) -> List[str]:
        """Get list of expected output files.

        Returns:
            list: List of output file names
        """
        return []


class SCFBackend(PhysicsBackend):
    """Base class for SCF (Self-Consistent Field) backends.

    This extends PhysicsBackend with SCF-specific methods.
    """

    def run_scf(self, systems: List[str], **kwargs) -> Dict[str, Any]:
        """Run SCF calculation on multiple systems.

        Args:
            systems: List of system paths
            **kwargs: SCF parameters

        Returns:
            dict: SCF results
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_scf."
        )

    def collect_stats(self, systems: List[str], **kwargs) -> Dict[str, Any]:
        """Collect statistics from SCF results.

        Backends are not required to implement this method.  The stats
        workflow calls backend-agnostic utilities in
        ``deepks.physics.backends.stats`` directly.

        Args:
            systems: List of system paths
            **kwargs: Collection parameters

        Returns:
            dict: Statistics and aggregated data
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement collect_stats. "
            "Use deepks.physics.backends.stats utilities directly."
        )
