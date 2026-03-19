"""ABACUS backend implementation.

This module implements the ABACUS backend for physics calculations.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional

from ..base import SCFBackend
from .input_generator import make_abacus_scf_input, make_abacus_scf_stru, make_abacus_scf_kpt
from .parser import (
    parse_abacus_output,
    check_convergence,
    parse_abacus_energy,
    parse_abacus_forces,
    parse_abacus_stress,
    parse_abacus_descriptor,
    parse_abacus_bandgap,
    parse_abacus_v_delta
)


class AbacusBackend(SCFBackend):
    """ABACUS backend for physics calculations.

    This backend handles ABACUS-specific operations:
    - Generate INPUT, STRU, KPT files
    - Run ABACUS calculations
    - Parse ABACUS output files
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ABACUS backend.

        Args:
            config: ABACUS-specific configuration
        """
        super().__init__(config)
        self.backend_name = 'abacus'

    def generate_input(self, system_data: Dict[str, Any],
                      output_dir: str, **kwargs) -> None:
        """Generate ABACUS input files.

        Args:
            system_data: System information with keys:
                - atom_names: List of element symbols
                - atom_numbs: List of atom counts per type
                - cells: Cell vectors
                - coords: Atomic coordinates
            output_dir: Directory to write input files
            **kwargs: ABACUS parameters (ecutwfc, scf_thr, etc.)

        Returns:
            None (files are written to disk)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Merge config with kwargs
        params = {**self.config, **kwargs}

        # Generate INPUT file
        input_content = make_abacus_scf_input(params)
        with open(os.path.join(output_dir, "INPUT"), 'w') as f:
            f.write(input_content)

        # Generate STRU file
        pp_files = params.get('pp_files', [])
        stru_content = make_abacus_scf_stru(system_data, pp_files, params)
        with open(os.path.join(output_dir, "STRU"), 'w') as f:
            f.write(stru_content)

        # Generate KPT file if needed
        if (params.get("k_points") is not None or
            params.get("gamma_only") is True):
            kpt_content = make_abacus_scf_kpt(params)
            with open(os.path.join(output_dir, "KPT"), 'w') as f:
                f.write(kpt_content)

    def run_calculation(self, work_dir: str, **kwargs) -> Dict[str, Any]:
        """Run ABACUS calculation.

        Note: This method prepares the command but doesn't execute it directly.
        Execution is handled by the orchestration layer.

        Args:
            work_dir: Working directory containing input files
            **kwargs: Runtime parameters

        Returns:
            dict: Execution metadata
        """
        params = {**self.config, **kwargs}

        abacus_path = params.get('abacus_path', 'abacus')
        run_cmd = params.get('run_cmd', 'mpirun')
        nproc = params.get('task_per_node', 1)

        command = f"{run_cmd} -n {nproc} {abacus_path}"

        return {
            'command': command,
            'work_dir': work_dir,
            'backend': 'abacus'
        }

    def parse_output(self, work_dir: str,
                    fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse ABACUS output files.

        Args:
            work_dir: Working directory containing output files
            fields: List of fields to extract

        Returns:
            dict: Parsed results
        """
        if fields is None:
            fields = ['e_tot', 'conv']

        out_dir = os.path.join(work_dir, "OUT.ABACUS")

        results = parse_abacus_output(out_dir, fields)
        results['converged'] = check_convergence(work_dir)

        return results

    def validate_config(self) -> bool:
        """Validate ABACUS configuration.

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['ecutwfc', 'scf_thr', 'scf_nmax']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required ABACUS parameter: {key}")

        return True

    def get_required_files(self) -> List[str]:
        """Get list of required input files.

        Returns:
            list: ['INPUT', 'STRU', 'KPT']
        """
        files = ['INPUT', 'STRU']

        if (self.config.get("k_points") is not None or
            self.config.get("gamma_only") == 1 or
            self.config.get("gamma_only") is True):
            files.append('KPT')

        return files

    def get_output_files(self) -> List[str]:
        """Get list of expected output files.

        Returns:
            list: Output file names
        """
        files = ['OUT.ABACUS/running_scf.log']

        if self.config.get('deepks_out_labels') == 1:
            files.append('OUT.ABACUS/deepks.dm_eig')

        if self.config.get('cal_force') == 1:
            files.append('OUT.ABACUS/running_scf.log')  # Forces in log

        if self.config.get('deepks_bandgap', 0) > 0:
            files.append('OUT.ABACUS/deepks.bandgap')

        return files

    def run_scf(self, systems: List[str], **kwargs) -> Dict[str, Any]:
        """Run SCF calculation on multiple systems.

        This method delegates to the SCF workflow.

        Args:
            systems: List of system paths
            **kwargs: SCF parameters

        Returns:
            dict: SCF results
        """
        from deepks.workflows.scf import run_scf_workflow

        config = {
            'type': 'scf',
            'scf_soft': 'abacus',
            'systems': systems,
            **self.config,
            **kwargs
        }

        return run_scf_workflow(config)

    def collect_stats(self, systems: List[str], **kwargs) -> Dict[str, Any]:
        """Collect statistics from SCF results.

        Args:
            systems: List of system paths
            **kwargs: Collection parameters

        Returns:
            dict: Statistics
        """
        # This will be implemented when we refactor the stats workflow
        raise NotImplementedError(
            "collect_stats will be implemented in stats workflow refactoring"
        )
