import os
import numpy as np
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BOHR2ANG = 0.52917721067
NAME_TYPE = {'X': 0, # Ghost
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
            'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
            'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
            'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
            'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
            'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
            'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
            'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
            'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
            'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
            'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
            'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
            'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
            'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
            'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
            'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113,
            'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118
        } #dict
TYPE_NAME = {v:k for k, v in NAME_TYPE.items()}
ELEMENTS = list(NAME_TYPE.keys())

# Default settings for abacus
DEFAULT_SCF_ARGS_ABACUS = {
    # for STRU
    "orb_files": ["orb"], 
    "pp_files": ["upf"], 
    "proj_file": ["orb"], 
    "lattice_constant": 1,
    "lattice_vector": np.eye(3,dtype=int),
    "coord_type": "Cartesian",
    # for INPUT
    "nspin": 1,
    "symmetry": 0,
    "nbands": None,
    "ecutwfc": 50,
    "scf_thr": 1e-7,
    "scf_nmax": 50,
    "dft_functional": "pbe", 
    "basis_type": "lcao",
    "gamma_only": 1,
    "k_points": None, # also used for KPT file if not None
    "kspacing": None,
    "smearing_method":"gaussian",
    "smearing_sigma": 0.02,
    "mixing_type": "pulay",
    "mixing_beta": 0.4,
    "cal_force": 0,
    "cal_stress": 0,
    "deepks_bandgap": 0,
    "deepks_v_delta": 0,
    "deepks_out_labels": 1,
    "deepks_scf": 0,
    "out_wfc_lcao": 0,
    # for running
    "run_cmd": "mpirun",
    "sub_size": 1,
    "abacus_path": "/usr/local/bin/ABACUS.mpi",
}

## Default settings for pyscf
_zeta = 1.5**np.array([17,13,10,7,5,3,2,1,0,-1,-2,-3])
_coef = np.diag(np.ones(_zeta.size)) - np.diag(np.ones(_zeta.size-1), k=1)
_table = np.concatenate([_zeta.reshape(-1,1), _coef], axis=1)
DEFAULT_BASIS = [[0, *_table.tolist()], [1, *_table.tolist()], [2, *_table.tolist()]]
DEFAULT_SYMB = "Ne"
DEFAULT_UNIT = "Bohr"
DEFAULT_FNAMES = {"e_tot", "e_base", "dm_eig", "conv"}
DEFAULT_HF_ARGS = {
    "conv_tol": 1e-9
}
DEFAULT_SCF_ARGS = {
    "conv_tol": 1e-7,
    # "level_shift": 0.1,
    # "diis_space": 20
}
MOL_ATTRIBUTE = {"charge", "basis", "unit"} # other molecule properties

## default settings for make scf tasks
SCF_CMD = "{python} -u -m deepks.scf.run"
TRN_CMD = "{python} -u -m deepks.model.train"
DEFAULT_SCF_RES = {
    "time_limit": "24:00:00",
    "cpus_per_task": 8,
    "mem_limit": 8,
    "envs": {
        "PYSCF_MAX_MEMORY": 8000
    }
}
DEFAULT_SCF_SUB_RES = {
    "numb_node": 1,
    "task_per_node": 1,
    "cpus_per_task": 8,
    "exclusive": True
}
DEFAULT_TRN_RES = {
    "time_limit": "24:00:00",
    "cpus_per_task": 8,
    # "numb_gpu": 1, # do not use gpu by default
    "mem_limit": 8
}
DEFAULT_DPDISPATCHER_RES = {
    "number_node": 1,
    "cpu_per_node": 8,
    "group_size": 1
}

## from iterate/iterate.py
DEFAULT_SCF_MACHINE = {
    "sub_size": 1, # how many systems is put in one task (folder)
    "sub_res": None, # the resources for sub step when ingroup_parallel > 1
    "group_size": 1, # how many tasks are submitted in one job
    "ingroup_parallel": 1, #how many tasks can run at same time in one job
    "dispatcher": None, # use default lazy-local slurm defined in task.py
    "resources": None, # use default 10 core defined in templete.py
    "python": "python", # use current python in path
    "dpdispatcher_machine": None, # (only need for dispatcher=="dpdispatcher")
    "dpdispatcher_resources": None # (only need for dispatcher=="dpdispatcher")
}

# args not specified here may cause error
DEFAULT_TRN_MACHINE = {
    "dispatcher": None, # use default lazy-local slurm defined in task.py
    "resources": None, # use default 10 core defined in templete.py
    "python": "python", # use current python in path
    "dpdispatcher_machine": None, # (only need for dispatcher=="dpdispatcher")
    "dpdispatcher_resources": None # (only need for dispatcher=="dpdispatcher")
}

SCF_ARGS_NAME = "scf_input.yaml"
SCF_ARGS_NAME_ABACUS="scf_abacus.yaml"   #for abacus, caoyu add 2021-07-26
INIT_SCF_NAME_ABACUS="init_scf_abacus.yaml"   #for abacus init, caoyu add 2021-12-17
TRN_ARGS_NAME = "train_input.yaml"
INIT_SCF_NAME = "init_scf.yaml"
INIT_TRN_NAME = "init_train.yaml"

DATA_TRAIN = "data_train"
DATA_TEST  = "data_test"
PROJ_BASIS = "proj_basis.npz"

SCF_STEP_DIR = "00.scf"
TRN_STEP_DIR = "01.train"

RECORD = "RECORD"

SYS_TRAIN = "systems_train"
SYS_TEST = "systems_test"
DEFAULT_TRAIN = "systems_train.raw"
DEFAULT_TEST = "systems_test.raw"

MODEL_FILE = "model.pth"
CMODEL_FILE = "model.ptg"