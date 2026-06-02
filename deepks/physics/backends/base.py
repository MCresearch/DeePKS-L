"""Physics backend base interface.

This module defines the abstract base class for all physics backends.
Each backend (PySCF, ABACUS) must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from deepks.physics.base import BackendRunner


class PhysicsBackend(BackendRunner, ABC):
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

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Stable backend identifier."""

    @property
    def name(self):
        return self.backend_name

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

    def prepare(self, systems, config, workdir):
        merged = {**self.config, **(config or {})}
        self.generate_input(systems, workdir, **merged)
        return {
            "systems": systems,
            "workdir": workdir,
            "config": merged,
        }

    def run(self, prepared, runtime_config):
        workdir = prepared.get("workdir") if isinstance(prepared, dict) else prepared
        return self.run_calculation(workdir, **(runtime_config or {}))

    def collect(self, prepared, runtime_config):
        workdir = prepared.get("workdir") if isinstance(prepared, dict) else prepared
        fields = None
        if isinstance(runtime_config, dict):
            fields = runtime_config.get("fields")
        return self.parse_output(workdir, fields=fields)


class SCFBackend(PhysicsBackend):
    """Marker base class for SCF-capable backends."""
