"""Physics backend adapter implementations for CLI/pipeline integration."""

import numpy as np
from deepks.core.contracts import PhysicsBackend


class PySCFPhysicsBackend(PhysicsBackend):
    """PySCF physics backend implementation."""

    def run_scf(self, **kwargs):
        """Run PySCF SCF calculation.

        Expected kwargs:
            systems, model_file, basis, proj_basis, penalty_terms,
            device, dump_dir, dump_fields, group, mol_args, scf_args, verbose
        """
        # Lazy import to avoid requiring pyscf at module load time
        from deepks.core.physics.pyscf.run import main as scf_run_main

        # Extract PySCF-specific parameters
        basis = kwargs.pop('basis', 'ccpvdz')
        mol_args = kwargs.pop('mol_args', {})
        scf_args = kwargs.pop('scf_args', {})

        # Call PySCF implementation
        return scf_run_main(
            basis=basis,
            mol_args=mol_args,
            scf_args=scf_args,
            **kwargs
        )

    def collect_stats(self, **kwargs):
        """Collect PySCF statistics."""
        # Lazy import to avoid requiring pyscf at module load time
        from deepks.core.physics.pyscf.stats import print_stats
        return print_stats(**kwargs)

    def validate_args(self, **kwargs):
        """Validate PySCF arguments."""
        # Check required parameters
        if 'systems' not in kwargs:
            raise ValueError("PySCF backend requires 'systems' parameter")

        # Validate basis if provided
        basis = kwargs.get('basis', 'ccpvdz')
        if not isinstance(basis, str):
            raise TypeError(f"basis must be string, got {type(basis)}")

        # Validate mol_args if provided
        mol_args = kwargs.get('mol_args', {})
        if not isinstance(mol_args, dict):
            raise TypeError(f"mol_args must be dict, got {type(mol_args)}")

        # Validate scf_args if provided
        scf_args = kwargs.get('scf_args', {})
        if not isinstance(scf_args, dict):
            raise TypeError(f"scf_args must be dict, got {type(scf_args)}")


class ABACUSPhysicsBackend(PhysicsBackend):
    """ABACUS physics backend implementation."""

    def run_scf(self, **kwargs):
        """Run ABACUS SCF calculation.

        Expected kwargs:
            systems, model_file, proj_basis, device, dump_dir, dump_fields,
            group, verbose, and ABACUS-specific parameters
        """
        from deepks.core.physics.abacus.run import main as abacus_run_main

        # Extract ABACUS-specific parameters
        abacus_args = {
            'abacus_path': kwargs.pop('abacus_path', '/usr/local/bin/ABACUS.mpi'),
            'run_cmd': kwargs.pop('run_cmd', 'mpirun'),
            'orb_files': kwargs.pop('orb_files', ['orb']),
            'pp_files': kwargs.pop('pp_files', ['upf']),
            'proj_file': kwargs.pop('proj_file', ['orb']),
            'lattice_constant': kwargs.pop('lattice_constant', 1),
            'lattice_vector': kwargs.pop('lattice_vector', np.eye(3)),
            'coord_type': kwargs.pop('coord_type', 'Cartesian'),
            'nspin': kwargs.pop('nspin', 1),
            'symmetry': kwargs.pop('symmetry', 0),
            'nbands': kwargs.pop('nbands', None),
            'ecutwfc': kwargs.pop('ecutwfc', 50),
            'scf_thr': kwargs.pop('scf_thr', 1e-7),
            'scf_nmax': kwargs.pop('scf_nmax', 50),
            'dft_functional': kwargs.pop('dft_functional', 'pbe'),
            'basis_type': kwargs.pop('basis_type', 'lcao'),
            'gamma_only': kwargs.pop('gamma_only', 1),
            'k_points': kwargs.pop('k_points', None),
            'kspacing': kwargs.pop('kspacing', None),
            'smearing_method': kwargs.pop('smearing_method', 'gaussian'),
            'smearing_sigma': kwargs.pop('smearing_sigma', 0.02),
            'mixing_type': kwargs.pop('mixing_type', 'pulay'),
            'mixing_beta': kwargs.pop('mixing_beta', 0.4),
            'cal_force': kwargs.pop('cal_force', 0),
            'cal_stress': kwargs.pop('cal_stress', 0),
            'deepks_bandgap': kwargs.pop('deepks_bandgap', 0),
            'deepks_v_delta': kwargs.pop('deepks_v_delta', 0),
            'deepks_out_labels': kwargs.pop('deepks_out_labels', 1),
            'deepks_scf': kwargs.pop('deepks_scf', 0),
            'out_wfc_lcao': kwargs.pop('out_wfc_lcao', 0),
        }

        # Call ABACUS implementation
        return abacus_run_main(**kwargs, **abacus_args)

    def collect_stats(self, **kwargs):
        """Collect ABACUS statistics.

        Note: Currently uses same stats collection as PySCF
        since output format is compatible.
        """
        return print_stats(**kwargs)

    def validate_args(self, **kwargs):
        """Validate ABACUS arguments."""
        # Check required parameters
        if 'systems' not in kwargs:
            raise ValueError("ABACUS backend requires 'systems' parameter")

        required_files = ['orb_files', 'pp_files']
        for param in required_files:
            if param not in kwargs:
                raise ValueError(f"ABACUS backend requires '{param}' parameter")
            if not isinstance(kwargs[param], list):
                raise TypeError(f"{param} must be list, got {type(kwargs[param])}")

        # Validate abacus_path
        abacus_path = kwargs.get('abacus_path', '/usr/local/bin/ABACUS.mpi')
        if not isinstance(abacus_path, str):
            raise TypeError(f"abacus_path must be string, got {type(abacus_path)}")

        # Validate numerical parameters
        ecutwfc = kwargs.get('ecutwfc', 50)
        if not isinstance(ecutwfc, (int, float)) or ecutwfc <= 0:
            raise ValueError(f"ecutwfc must be positive number, got {ecutwfc}")

