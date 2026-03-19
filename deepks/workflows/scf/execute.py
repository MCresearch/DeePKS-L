"""SCF workflow - Execute stage.

This module handles the execution stage of SCF workflow:
- Submit SCF calculations to scheduler
- Support multiple dispatchers (local shell, Slurm, PBS, dpdispatcher)
- Handle batch execution and resource management
"""

import os
import numpy as np

from deepks.utils import load_sys_paths, get_sys_name
from deepks.orchestration.workflow.task import BatchTask, GroupBatchTask, DPDispatcherTask
from deepks.orchestration.workflow.workflow import Sequence


def coord_to_atom(path):
    """Convert coord.npy and type.raw to atom.npy format.

    Args:
        path: System directory path

    Returns:
        np.ndarray: Atom data with shape (nframes, natoms, 4)
    """
    from deepks.default import NAME_TYPE

    try:
        coords = np.load(f"{path}/coord.npy")
    except FileNotFoundError:
        raise FileNotFoundError(f"coord.npy not found in {path}")

    nframes = coords.shape[0]
    if coords.shape[2] != 3:
        raise ValueError("coord.npy should have shape (nframes, natoms, 3)")

    # Get type mapping
    with open(f"{path}/type_map.raw") as fp:
        my_type_map = [NAME_TYPE[i] for i in fp.read().split()]

    atom_types = np.loadtxt(f"{path}/type.raw", ndmin=1).astype(int)
    atom_types = np.array([int(my_type_map[i-1]) for i in atom_types])
    atom_types = atom_types.reshape(1, -1).repeat(nframes, axis=0)

    atom_data = np.insert(coords, 0, values=atom_types, axis=2)
    return atom_data


def execute_scf_tasks(prepare_task, config):
    """Execute SCF tasks (Stage 2).

    This function runs the SCF calculations using the orchestration layer.

    Args:
        prepare_task: Preparation task from stage 1
        config: Configuration dictionary

    Returns:
        None (results are written to disk)
    """
    scf_soft = config.get('scf_soft', 'pyscf')

    if scf_soft.lower() == 'abacus':
        execute_scf_tasks_abacus(prepare_task, config)
    elif scf_soft.lower() == 'pyscf':
        raise NotImplementedError(
            "PySCF workflow not yet implemented in new architecture"
        )
    else:
        raise ValueError(f"Unknown SCF backend: {scf_soft}")


def execute_scf_tasks_abacus(prepare_task, config):
    """Execute ABACUS SCF tasks.

    Args:
        prepare_task: Preparation task
        config: Configuration dictionary

    Returns:
        None
    """
    # Extract configuration
    systems = config.get('systems', [])
    dispatcher = config.get('scf_machine', {}).get('dispatcher', {}).get('batch', 'shell')
    resources = config.get('scf_machine', {}).get('resources', {})
    group_size = config.get('scf_machine', {}).get('group_size', 1)

    # ABACUS-specific config
    scf_abacus = config.get('scf_abacus', {})
    abacus_path = scf_abacus.get('abacus_path', 'abacus')
    run_cmd = scf_abacus.get('run_cmd', 'mpirun')
    task_per_node = resources.get('task_per_node', 1)

    outlog = "out.log"
    errlog = "err.log"

    # Get system paths and names
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
    sys_names = [os.path.basename(get_sys_name(s)) for s in sys_paths]

    # Build execution task
    if dispatcher == 'dpdispatcher':
        run_task = build_dpdispatcher_task(
            sys_paths, sys_names, abacus_path, run_cmd,
            task_per_node, outlog, errlog, config
        )
    else:
        run_task = build_batch_task(
            sys_paths, sys_names, abacus_path, run_cmd,
            task_per_node, group_size, dispatcher, resources,
            outlog, errlog, config
        )

    # Create workflow and execute
    workflow = Sequence([prepare_task, run_task], workdir='.')
    workflow.run()


def build_dpdispatcher_task(sys_paths, sys_names, abacus_path, run_cmd,
                            task_per_node, outlog, errlog, config):
    """Build DPDispatcher task for ABACUS execution.

    Args:
        sys_paths: List of system paths
        sys_names: List of system names
        abacus_path: Path to ABACUS executable
        run_cmd: MPI run command
        task_per_node: Tasks per node
        outlog: Output log filename
        errlog: Error log filename
        config: Configuration dictionary

    Returns:
        DPDispatcherTask: Task for dpdispatcher execution
    """
    from dpdispatcher import Task

    scf_machine = config.get('scf_machine', {})
    dpdispatcher_config = scf_machine.get('dispatcher', {})
    dpdispatcher_machine = dpdispatcher_config.get('machine', {})
    dpdispatcher_resources = scf_machine.get('resources', {})

    # Get forward/backward files
    scf_abacus = config.get('scf_abacus', {})
    orb_files = scf_abacus.get('orb_files', [])
    pp_files = scf_abacus.get('pp_files', [])
    proj_file = scf_abacus.get('proj_file', [])

    from deepks.utils import flat_file_list
    orb_files = [os.path.abspath(s) for s in flat_file_list(orb_files, sort=False)]
    pp_files = [os.path.abspath(s) for s in flat_file_list(pp_files, sort=False)]
    proj_file = [os.path.abspath(s) for s in flat_file_list(proj_file, sort=False)]
    forward_files = orb_files + pp_files + proj_file

    backward_files = ["OUT.ABACUS/", "conv"]

    # Build task list
    task_list = []
    for i, pth in enumerate(sys_paths):
        try:
            atom_data = np.load(f"{pth}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(pth)

        nframes = atom_data.shape[0]

        for f in range(nframes):
            task_dict = {
                "command": (
                    f"cd {sys_names[i]}/ABACUS/{f}/ && "
                    f"{run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog} && "
                    f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv && "
                    f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log`"
                ),
                "task_work_path": ".",
                "forward_files": [f"./{sys_names[i]}/ABACUS/{f}/"],
                "backward_files": [f"./{sys_names[i]}/ABACUS/{f}/"],
                "outlog": outlog,
                "errlog": errlog
            }
            task_list.append(Task.load_from_dict(task_dict))

    return DPDispatcherTask(
        task_list,
        work_base="systems",
        outlog=outlog,
        machine=dpdispatcher_machine,
        resources=dpdispatcher_resources,
        forward_files=forward_files,
        backward_files=backward_files
    )


def build_batch_task(sys_paths, sys_names, abacus_path, run_cmd,
                    task_per_node, group_size, dispatcher, resources,
                    outlog, errlog, config):
    """Build batch task for ABACUS execution.

    Args:
        sys_paths: List of system paths
        sys_names: List of system names
        abacus_path: Path to ABACUS executable
        run_cmd: MPI run command
        task_per_node: Tasks per node
        group_size: Group size for batch execution
        dispatcher: Dispatcher type
        resources: Resource configuration
        outlog: Output log filename
        errlog: Error log filename
        config: Configuration dictionary

    Returns:
        GroupBatchTask: Task for batch execution
    """
    # Get forward/backward files
    scf_abacus = config.get('scf_abacus', {})
    orb_files = scf_abacus.get('orb_files', [])
    pp_files = scf_abacus.get('pp_files', [])
    proj_file = scf_abacus.get('proj_file', [])

    from deepks.utils import flat_file_list
    orb_files = [os.path.abspath(s) for s in flat_file_list(orb_files, sort=False)]
    pp_files = [os.path.abspath(s) for s in flat_file_list(pp_files, sort=False)]
    proj_file = [os.path.abspath(s) for s in flat_file_list(proj_file, sort=False)]
    forward_files = orb_files + pp_files + proj_file

    backward_files = ["OUT.ABACUS/", "conv"]

    # Build batch tasks
    batch_tasks = []
    for i, pth in enumerate(sys_paths):
        try:
            atom_data = np.load(f"{pth}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(pth)

        nframes = atom_data.shape[0]

        for f in range(nframes):
            cmd = (
                f"cd {sys_names[i]}/ABACUS/{f}/ && "
                f"{run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog} && "
                f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv && "
                f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log`"
            )

            batch_tasks.append(BatchTask(
                cmds=cmd,
                workdir="systems",
                forward_files=[f"./{sys_names[i]}/ABACUS/{f}/"],
                backward_files=[f"./{sys_names[i]}/ABACUS/{f}/"]
            ))

    return GroupBatchTask(
        batch_tasks,
        group_size=group_size,
        workdir="./",
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        forward_files=forward_files,
        backward_files=backward_files
    )
