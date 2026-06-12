"""ABACUS workflow adapters.

These helpers bridge packed SCF workflow parameters to backend-specific
prepare / execute / collect operations. They live in ``physics`` so the
``workflows`` layer can remain a thin orchestration shell.
"""

import os
from collections import Counter
from dataclasses import asdict

import numpy as np

from deepks.io.utils import flat_file_list, get_sys_name, load_sys_paths
from deepks.orchestration.workflow.task import BatchTask, DPDispatcherTask, GroupBatchTask, PythonTask
from deepks.orchestration.workflow.workflow import Sequence
from deepks.physics.backends.types import SCFResult
from deepks.physics.constants import NAME_TYPE, TYPE_NAME

from deepks.physics.backends.abacus.input_generator import (
    make_abacus_scf_input,
    make_abacus_scf_kpt,
    make_abacus_scf_stru,
)
from deepks.physics.backends.abacus.parser import parse_abacus_output


def coord_to_atom(path):
    """Convert ``coord.npy`` and ``type.raw`` to atom.npy-like arrays."""
    try:
        coords = np.load(f"{path}/coord.npy")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"coord.npy not found in {path}") from exc

    nframes = coords.shape[0]
    if coords.shape[2] != 3:
        raise ValueError("coord.npy should have shape (nframes, natoms, 3)")

    with open(f"{path}/type_map.raw") as fp:
        my_type_map = [NAME_TYPE[i] for i in fp.read().split()]

    atom_types = np.loadtxt(f"{path}/type.raw", ndmin=1).astype(int)
    atom_types = np.array([int(my_type_map[i - 1]) for i in atom_types])
    atom_types = atom_types.reshape(1, -1).repeat(nframes, axis=0)
    return np.insert(coords, 0, values=atom_types, axis=2)


def prepare_abacus_input_files(systems, scf_args, orb_files, pp_files, proj_file):
    """Generate ABACUS input files for all systems/frames."""
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]

    for sys_path in sys_paths:
        try:
            atom_data = np.load(f"{sys_path}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_path)

        if os.path.isfile(f"{sys_path}/box.npy"):
            cell_data = np.load(f"{sys_path}/box.npy")
            if cell_data.shape != (atom_data.shape[0], 9):
                raise ValueError(
                    f"box.npy should have shape (nframes, 9), but got {cell_data.shape}!"
                )

        nframes = atom_data.shape[0]
        abacus_dir = f"{sys_path}/ABACUS"
        os.makedirs(abacus_dir, exist_ok=True)

        scf_args_local = dict(scf_args)
        if os.path.exists(f"{sys_path}/group_scf_abacus.yaml"):
            from deepks.io.utils import load_yaml

            local_config = load_yaml(f"{sys_path}/group_scf_abacus.yaml")
            scf_args_local.update(local_config)

        for frame_index in range(nframes):
            frame_dir = f"{abacus_dir}/{frame_index}"
            os.makedirs(frame_dir, exist_ok=True)

            frame_data = atom_data[frame_index]
            atoms = frame_data[:, 0]
            nta = Counter(atoms)
            sys_data = {
                "atom_names": [TYPE_NAME[int(it)] for it in nta.keys()],
                "atom_numbs": list(nta.values()),
                "cells": np.array([scf_args_local["lattice_vector"]]),
                "coords": [frame_data[:, 1:]],
            }
            if os.path.isfile(f"{sys_path}/box.npy"):
                sys_data["cells"] = [cell_data[frame_index]]

            with open(f"{frame_dir}/STRU", "w") as stru_file:
                stru_file.write(make_abacus_scf_stru(sys_data, pp_files, scf_args_local))
            with open(f"{frame_dir}/INPUT", "w") as input_file:
                input_file.write(make_abacus_scf_input(scf_args_local))
            if (
                scf_args_local.get("k_points") is not None
                or scf_args_local.get("gamma_only") is True
            ):
                with open(f"{frame_dir}/KPT", "w") as kpt_file:
                    kpt_file.write(make_abacus_scf_kpt(scf_args_local))


def build_prepare_task(config):
    """Create the prepare-stage task for ABACUS SCF."""
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    systems = [] if data.get("systems") is None else data.get("systems", [])
    if not systems:
        raise ValueError("No systems specified in config")
    sys_paths = load_sys_paths(systems)
    if not sys_paths:
        raise FileNotFoundError(f"no valid system paths found from: {systems}")
    for system in sys_paths:
        if not os.path.exists(system):
            raise FileNotFoundError(f"system path not found: {system}")

    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_input = dict(backend.get("input", {})) if isinstance(backend.get("input"), dict) else {}

    orb_files = [os.path.abspath(s) for s in flat_file_list(backend_input.get("orb_files", []), sort=False)]
    pp_files = [os.path.abspath(s) for s in flat_file_list(backend_input.get("pp_files", []), sort=False)]
    proj_file = [os.path.abspath(s) for s in flat_file_list(backend_input.get("proj_file", []), sort=False)]

    scf_args = dict(backend_input)
    scf_args["orb_files"] = orb_files
    scf_args["pp_files"] = pp_files
    scf_args["proj_file"] = proj_file

    return PythonTask(
        prepare_abacus_input_files,
        call_kwargs={
            "systems": systems,
            "scf_args": scf_args,
            "orb_files": orb_files,
            "pp_files": pp_files,
            "proj_file": proj_file,
        },
        outlog="prepare.log",
        errlog="prepare.err",
        workdir=".",
    )


def _forward_files_from_backend_input(backend_input):
    orb_files = [os.path.abspath(s) for s in flat_file_list(backend_input.get("orb_files", []), sort=False)]
    pp_files = [os.path.abspath(s) for s in flat_file_list(backend_input.get("pp_files", []), sort=False)]
    proj_file = [os.path.abspath(s) for s in flat_file_list(backend_input.get("proj_file", []), sort=False)]
    return orb_files + pp_files + proj_file


def _build_dpdispatcher_task(sys_paths, sys_names, abacus_path, run_cmd,
                             task_per_node, outlog, errlog, backend_input, execute_cfg):
    from dpdispatcher import Task

    dispatcher_cfg = execute_cfg.get("dispatcher", {})
    dpdispatcher_machine = dispatcher_cfg.get("machine", {})
    dpdispatcher_resources = execute_cfg.get("resources", {})
    forward_files = _forward_files_from_backend_input(backend_input)
    backward_files = ["OUT.ABACUS/", "conv"]
    task_list = []

    for index, path in enumerate(sys_paths):
        try:
            atom_data = np.load(f"{path}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(path)
        nframes = atom_data.shape[0]

        for frame_index in range(nframes):
            task_list.append(Task.load_from_dict({
                "command": (
                    f"cd {sys_names[index]}/ABACUS/{frame_index}/ && "
                    f"{run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog} && "
                    f"echo {frame_index}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv && "
                    f"echo {frame_index}`grep -i converge ./OUT.ABACUS/running_scf.log`"
                ),
                "task_work_path": ".",
                "forward_files": [f"./{sys_names[index]}/ABACUS/{frame_index}/"],
                "backward_files": [f"./{sys_names[index]}/ABACUS/{frame_index}/"],
                "outlog": outlog,
                "errlog": errlog,
            }))

    return DPDispatcherTask(
        task_list,
        work_base="systems",
        outlog=outlog,
        machine=dpdispatcher_machine,
        resources=dpdispatcher_resources,
        forward_files=forward_files,
        backward_files=backward_files,
    )


def _build_batch_task(sys_paths, sys_names, abacus_path, run_cmd,
                      task_per_node, group_size, dispatcher, resources,
                      outlog, errlog, backend_input):
    forward_files = _forward_files_from_backend_input(backend_input)
    backward_files = ["OUT.ABACUS/", "conv"]
    batch_tasks = []

    for index, path in enumerate(sys_paths):
        try:
            atom_data = np.load(f"{path}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(path)
        nframes = atom_data.shape[0]

        for frame_index in range(nframes):
            cmd = (
                f"cd {sys_names[index]}/ABACUS/{frame_index}/ && "
                f"{run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog} && "
                f"echo {frame_index}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv && "
                f"echo {frame_index}`grep -i converge ./OUT.ABACUS/running_scf.log`"
            )
            batch_tasks.append(BatchTask(
                cmds=cmd,
                workdir="systems",
                forward_files=[f"./{sys_names[index]}/ABACUS/{frame_index}/"],
                backward_files=[f"./{sys_names[index]}/ABACUS/{frame_index}/"],
            ))

    return GroupBatchTask(
        batch_tasks,
        group_size=group_size,
        workdir="./",
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        forward_files=forward_files,
        backward_files=backward_files,
    )


def execute_sequence(prepare_task, config):
    """Run the prepare + execute sequence for ABACUS SCF."""
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    scf_runtime = runtime.get("scf") if isinstance(runtime.get("scf"), dict) else {}
    execute_cfg = dict(scf_runtime.get("execute", {})) if isinstance(scf_runtime.get("execute"), dict) else {}
    dispatcher_cfg = execute_cfg.get("dispatcher", {})
    dispatcher_batch = dispatcher_cfg.get("batch", "shell")
    resources = execute_cfg.get("resources", {})
    group_size = execute_cfg.get("group_size", 1)

    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_input = dict(backend.get("input", {})) if isinstance(backend.get("input"), dict) else {}
    command_cfg = dict(scf_runtime.get("command", {})) if isinstance(scf_runtime.get("command"), dict) else {}
    abacus_path = command_cfg.get("abacus_path", "abacus")
    run_cmd = command_cfg.get("run_cmd", "mpirun")
    task_per_node = resources.get("task_per_node", 1)
    outlog = "out.log"
    errlog = "err.log"

    sys_paths = [os.path.abspath(s) for s in load_sys_paths([] if data.get("systems") is None else data.get("systems", []))]
    sys_names = [os.path.basename(get_sys_name(s)) for s in sys_paths]

    if dispatcher_batch == "dpdispatcher":
        run_task = _build_dpdispatcher_task(
            sys_paths, sys_names, abacus_path, run_cmd,
            task_per_node, outlog, errlog, backend_input, execute_cfg,
        )
    else:
        run_task = _build_batch_task(
            sys_paths, sys_names, abacus_path, run_cmd,
            task_per_node, group_size, dispatcher_cfg or {"batch": dispatcher_batch},
            resources, outlog, errlog, backend_input,
        )

    Sequence([prepare_task, run_task], workdir=".").run()


def _load_system_geometry(sys_path, lattice_vector, lattice_constant, coord_type):
    try:
        atom_data = np.load(f"{sys_path}/atom.npy")
    except FileNotFoundError:
        atom_data = coord_to_atom(sys_path)

    nframes = atom_data.shape[0]
    if os.path.isfile(f"{sys_path}/box.npy"):
        box_data = np.load(f"{sys_path}/box.npy")
    else:
        box_data = np.array([lattice_vector]).reshape(1, 9).repeat(nframes, axis=0)

    box_data = box_data.reshape(nframes, 3, 3)
    if coord_type == "Direct":
        atom_data[:, :, 1:4] = np.matmul(atom_data[:, :, 1:4], box_data)

    atom_data[:, :, 1:4] *= lattice_constant
    box_data *= lattice_constant
    return atom_data, box_data


def _build_parser_fields(dump_fields, cal_force, cal_stress, deepks_bandgap, deepks_v_delta):
    parser_fields = {"conv"}
    if any(field in dump_fields for field in ("e_tot", "e_base")):
        parser_fields.update({"e_tot", "e_base"})
    if cal_force and any(field in dump_fields for field in ("f_tot", "f_base")):
        parser_fields.update({"f_tot", "f_base"})
    if cal_stress and any(field in dump_fields for field in ("s_tot", "s_base")):
        parser_fields.update({"s_tot", "s_base"})
    if "dm_eig" in dump_fields:
        parser_fields.add("dm_eig")
    if deepks_bandgap and "bandgap" in dump_fields:
        parser_fields.add("bandgap")
    if deepks_v_delta and "v_delta_precondition" in dump_fields:
        parser_fields.add("v_delta")
    return parser_fields


def _initialize_result_buffers(nframes, natoms):
    return {
        "conv": np.full((nframes, 1), False),
        "e_tot": None,
        "e_base": None,
        "f_tot": None,
        "f_base": None,
        "s_tot": None,
        "s_base": None,
        "dm_eig": None,
        "bandgap": None,
        "v_delta_precondition": None,
        "natoms": natoms,
        "nframes": nframes,
    }


def _store_scalar_field(buffers, key, frame_index, value):
    if value is None:
        return
    if buffers[key] is None:
        buffers[key] = np.zeros((buffers["nframes"], 1))
    buffers[key][frame_index] = value


def _store_tensor_field(buffers, key, frame_index, value, shape):
    if value is None:
        return
    if buffers[key] is None:
        buffers[key] = np.zeros((buffers["nframes"],) + shape)
    buffers[key][frame_index] = value


def _collect_system_frames(sys_path, dump_fields, parser_fields, natoms):
    frame_dirs = sorted(
        entry.path
        for entry in os.scandir(os.path.join(sys_path, "ABACUS"))
        if entry.is_dir() and entry.name.isdigit()
    )
    buffers = _initialize_result_buffers(len(frame_dirs), natoms)

    for frame_index, frame_dir in enumerate(frame_dirs):
        parsed = parse_abacus_output(frame_dir, fields=list(parser_fields), natoms=natoms)
        buffers["conv"][frame_index] = parsed.get("convergence", False)
        _store_scalar_field(buffers, "e_tot", frame_index, parsed.get("e_tot"))
        _store_scalar_field(buffers, "e_base", frame_index, parsed.get("e_base"))
        _store_scalar_field(buffers, "bandgap", frame_index, parsed.get("bandgap"))
        _store_tensor_field(buffers, "f_tot", frame_index, parsed.get("f_tot"), (natoms, 3))
        _store_tensor_field(buffers, "f_base", frame_index, parsed.get("f_base"), (natoms, 3))
        _store_tensor_field(buffers, "s_tot", frame_index, parsed.get("s_tot"), (3, 3))
        _store_tensor_field(buffers, "s_base", frame_index, parsed.get("s_base"), (3, 3))
        descriptor = parsed.get("dm_eig")
        if descriptor is not None:
            _store_tensor_field(buffers, "dm_eig", frame_index, descriptor, descriptor.shape)
        v_delta = parsed.get("v_delta")
        if v_delta is not None:
            _store_tensor_field(buffers, "v_delta_precondition", frame_index, v_delta, (natoms,))

    save_results = {"conv": buffers["conv"]}
    for key in (
        "e_tot", "e_base", "f_tot", "f_base", "s_tot", "s_base",
        "dm_eig", "bandgap", "v_delta_precondition",
    ):
        if buffers[key] is not None and key in dump_fields:
            save_results[key] = buffers[key]
    return save_results


def collect_results(config):
    """Collect ABACUS SCF results into dump arrays."""
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    output_settings = dict(backend.get("output", {})) if isinstance(backend.get("output"), dict) else {}
    dump_dir = output_settings["dump_dir"]
    dump_fields = output_settings["dump_fields"]
    backend_input = dict(backend.get("input", {})) if isinstance(backend.get("input"), dict) else {}

    cal_force = backend_input.get("cal_force", 0)
    cal_stress = backend_input.get("cal_stress", 0)
    deepks_bandgap = backend_input.get("deepks_bandgap", 0)
    deepks_v_delta = backend_input.get("deepks_v_delta", 0)
    lattice_constant = backend_input.get("lattice_constant", 1.0)
    coord_type = backend_input.get("coord_type", "Cartesian")
    lattice_vector = backend_input.get("lattice_vector", np.eye(3))

    systems = [] if data.get("systems") is None else data.get("systems", [])
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
    sys_names = [os.path.basename(get_sys_name(s)) for s in sys_paths]
    parser_fields = _build_parser_fields(dump_fields, cal_force, cal_stress, deepks_bandgap, deepks_v_delta)

    os.makedirs(dump_dir, exist_ok=True)
    result = SCFResult(dump_dir=dump_dir)

    for sys_path, sys_name in zip(sys_paths, sys_names):
        sys_dump_dir = os.path.join(dump_dir, sys_name)
        os.makedirs(sys_dump_dir, exist_ok=True)
        atom_data, box_data = _load_system_geometry(
            sys_path,
            lattice_vector=lattice_vector,
            lattice_constant=lattice_constant,
            coord_type=coord_type,
        )
        nframes = atom_data.shape[0]
        natoms = atom_data.shape[1]
        save_results = _collect_system_frames(sys_path, dump_fields, parser_fields, natoms)
        save_results["atom"] = atom_data
        save_results["box"] = box_data.reshape(nframes, 9)

        for key, value in save_results.items():
            np.save(f"{sys_dump_dir}/{key}.npy", value)

        result.systems.append({
            "name": sys_name,
            "path": sys_dump_dir,
            "nframes": nframes,
            "natoms": natoms,
            "converged": int(save_results["conv"].sum()),
        })

    total_frames = sum(system["nframes"] for system in result.systems)
    total_converged = sum(system["converged"] for system in result.systems)
    result.statistics = {
        "total_systems": len(systems),
        "total_frames": total_frames,
        "total_converged": total_converged,
        "convergence_rate": total_converged / total_frames if total_frames > 0 else 0.0,
    }
    return asdict(result)
