"""Iterate task-template fallback settings."""

SCF_CMD = "deepks"

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
