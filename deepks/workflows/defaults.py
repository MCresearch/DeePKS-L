"""Workflow-level command, runtime, resource, and file-name defaults."""

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SCF_CMD = "{python} -u -m deepks.physics.backends.pyscf.run"
TRN_CMD = "{python} -u -m deepks.ml.train.train"

DEFAULT_SCF_RES = {
    "time_limit": "24:00:00",
    "cpus_per_task": 8,
    "mem_limit": 8,
    "envs": {
        "PYSCF_MAX_MEMORY": 8000,
    },
}
DEFAULT_SCF_SUB_RES = {
    "numb_node": 1,
    "task_per_node": 1,
    "cpus_per_task": 8,
    "exclusive": True,
}
DEFAULT_TRN_RES = {
    "time_limit": "24:00:00",
    "cpus_per_task": 8,
    "mem_limit": 8,
}
DEFAULT_DPDISPATCHER_RES = {
    "number_node": 1,
    "cpu_per_node": 8,
    "group_size": 1,
}

DEFAULT_SCF_MACHINE = {
    "sub_size": 1,
    "sub_res": None,
    "group_size": 1,
    "ingroup_parallel": 1,
    "dispatcher": None,
    "resources": None,
    "python": "python",
    "dpdispatcher_machine": None,
    "dpdispatcher_resources": None,
}
DEFAULT_TRN_MACHINE = {
    "dispatcher": None,
    "resources": None,
    "python": "python",
    "dpdispatcher_machine": None,
    "dpdispatcher_resources": None,
}

SCF_ARGS_NAME = "scf_input.yaml"
SCF_ARGS_NAME_ABACUS = "scf_abacus.yaml"
INIT_SCF_NAME_ABACUS = "init_scf_abacus.yaml"
TRN_ARGS_NAME = "train_input.yaml"
INIT_SCF_NAME = "init_scf.yaml"
INIT_TRN_NAME = "init_train.yaml"

DATA_TRAIN = "data_train"
DATA_TEST = "data_test"
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
