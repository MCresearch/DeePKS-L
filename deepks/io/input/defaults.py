"""Default configuration values for DeePKS."""

# Common defaults
DEFAULT_COMMON = {
    'verbose': 1,
    'device': 'cpu',
}

# SCF defaults (backend-agnostic)
DEFAULT_SCF_COMMON = {
    'dump_fields': ['e_tot', 'e_base', 'dm_eig', 'conv'],
    'group': False,
}

# PySCF-specific defaults
DEFAULT_SCF_PYSCF = {
    'basis': 'ccpvdz',
    'proj_basis': 'ccpvdz',
    'mol_args': {
        'charge': 0,
        'spin': 0,
        'unit': 'Angstrom',
    },
    'scf_args': {
        'conv_tol': 1e-7,
        'max_cycle': 50,
    },
}

# ABACUS-specific defaults
DEFAULT_SCF_ABACUS = {
    'abacus_path': '/usr/local/bin/ABACUS.mpi',
    'run_cmd': 'mpirun',
    'lattice_constant': 1,
    'coord_type': 'Cartesian',
    'nspin': 1,
    'symmetry': 0,
    'ecutwfc': 50,
    'scf_thr': 1e-7,
    'scf_nmax': 50,
    'dft_functional': 'pbe',
    'basis_type': 'lcao',
    'gamma_only': 1,
    'smearing_method': 'gaussian',
    'smearing_sigma': 0.02,
    'mixing_type': 'pulay',
    'mixing_beta': 0.4,
    'cal_force': 0,
    'cal_stress': 0,
    'deepks_out_labels': 1,
    'deepks_scf': 0,
}

# Training defaults
DEFAULT_TRAIN = {
    'model_args': {
        'hidden_sizes': [100, 100, 100],
        'output_scale': 100.0,
        'use_resnet': True,
    },
    'preprocess_args': {
        'preshift': True,
        'prescale': [1, 1],
    },
    'train_args': {
        'n_epoch': 1000,
        'batch_size': 16,
        'eval_batch_size': None,
        'start_lr': 0.01,
        'decay_rate': 0.96,
        'decay_steps': 100,
        'print_freq': 100,
        'save_freq': 1000,
        'restart': None,
        'ckpt_file': None,
    },
}

# Test defaults
DEFAULT_TEST = {
    'batch_size': 16,
}

# Iterate defaults
DEFAULT_ITERATE = {
    'n_iter': 10,
    'workdir': '.',
    'share_folder': 'share',
    'cleanup': False,
    'strict': True,
}


def get_default_config(command=None, scf_soft='pyscf'):
    """Get default configuration for a command.

    Args:
        command: Command name ('scf', 'train', 'test', 'iterate')
        scf_soft: SCF backend ('pyscf' or 'abacus')

    Returns:
        dict: Default configuration
    """
    config = DEFAULT_COMMON.copy()

    if command == 'scf':
        config.update(DEFAULT_SCF_COMMON)
        if scf_soft.lower() == 'pyscf':
            config.update(DEFAULT_SCF_PYSCF)
        elif scf_soft.lower() == 'abacus':
            config.update(DEFAULT_SCF_ABACUS)
    elif command == 'train':
        config.update(DEFAULT_TRAIN)
    elif command == 'test':
        config.update(DEFAULT_TEST)
    elif command == 'iterate':
        config.update(DEFAULT_ITERATE)
        config.update(DEFAULT_SCF_COMMON)
        # Add SCF defaults based on backend
        if scf_soft.lower() == 'pyscf':
            config.update(DEFAULT_SCF_PYSCF)
        elif scf_soft.lower() == 'abacus':
            config.update(DEFAULT_SCF_ABACUS)
        config.update(DEFAULT_TRAIN)

    return config
