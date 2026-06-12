"""ABACUS iterate task builders.

These are ABACUS-specific orchestration task constructors used by the iterate
workflow. Keeping them under ``physics.backends.abacus`` prevents workflow
modules from owning backend execution details.
"""

import os
from glob import glob

import numpy as np

from deepks.orchestration.workflow.task import BatchTask, DPDispatcherTask, GroupBatchTask, PythonTask
from deepks.physics.backends.abacus.constants import CMODEL_FILE
from deepks.physics.backends.abacus.iterate_ops import convert_data, coord_to_atom, gather_stats_abacus
from deepks.io.utils import get_sys_name


def make_convert_scf_abacus(systems_train, systems_test=None,
                            no_model=True, model_file=None, resources=None,
                            check_system_names=None, **pre_args):
    systems_train = [os.path.abspath(s) for s in systems_train]
    systems_test = [os.path.abspath(s) for s in systems_test] if systems_test else []
    link_prev = pre_args.pop("link_prev_files", [])
    if not systems_test:
        systems_test.append(systems_train[-1])
    if check_system_names is not None:
        check_system_names(systems_train)
        check_system_names(systems_test)
    if not no_model:
        assert model_file is not None
        link_prev.append((model_file, "model.pth"))
    task_per_node = 1
    if resources is not None and "task_per_node" in resources:
        task_per_node = resources["task_per_node"]
    pre_args.update(
        systems_train=systems_train,
        systems_test=systems_test,
        model_file=model_file,
        no_model=no_model,
        task_per_node=task_per_node,
        **pre_args,
    )
    return PythonTask(
        convert_data,
        call_kwargs=pre_args,
        outlog="convert.log",
        errlog="err",
        workdir=".",
        link_prev_files=link_prev,
    )


def make_run_scf_abacus(systems_train, systems_test=None,
                        outlog="out.log", errlog="err.log", group_size=1,
                        resources=None, dispatcher=None,
                        share_folder="share", workdir=".", link_systems=True,
                        dpdispatcher_machine=None, dpdispatcher_resources=None,
                        no_model=True, coord_to_atom_fn=coord_to_atom,
                        check_system_names=None, **task_args):
    link_share = task_args.pop("link_share_files", [])
    link_prev = task_args.pop("link_prev_files", [])
    link_abs = task_args.pop("link_abs_files", [])
    forward_files = task_args.pop("forward_files", [])
    backward_files = task_args.pop("backward_files", [])
    if not no_model:
        forward_files.append("../" + CMODEL_FILE)

    systems_train = [os.path.abspath(s) for s in systems_train]
    systems_test = [os.path.abspath(s) for s in systems_test] if systems_test else []
    if not systems_test:
        systems_test.append(systems_train[-1])
    if check_system_names is not None:
        check_system_names(systems_train)
        check_system_names(systems_test)

    sys_train_paths = [os.path.abspath(s) for s in systems_train]
    sys_train_base = [get_sys_name(s) for s in sys_train_paths]
    sys_test_paths = [os.path.abspath(s) for s in systems_test]
    sys_test_base = [get_sys_name(s) for s in sys_test_paths]
    sys_paths = sys_train_paths + sys_test_paths
    sys_name = [os.path.basename(s) for s in (sys_train_base + sys_test_base)]
    if link_systems:
        target_dir = "systems"
        src_files = sum((glob(f"{base}*") for base in (sys_train_base + sys_test_base)), [])
        for fl in src_files:
            dst = os.path.join(target_dir, os.path.basename(fl))
            link_abs.append((fl, dst))

    task_per_node = 1
    if resources is not None and "task_per_node" in resources:
        task_per_node = resources["task_per_node"]
    run_cmd = task_args.pop("run_cmd", "mpirun")
    abacus_path = task_args.pop("abacus_path", None)
    assert abacus_path is not None

    if dispatcher == "dpdispatcher":
        if dpdispatcher_resources is not None and "cpu_per_node" in dpdispatcher_resources:
            assert task_per_node <= dpdispatcher_resources["cpu_per_node"]
        from dpdispatcher import Task

        task_list = []
        singletask = {
            "command": None,
            "task_work_path": "./",
            "forward_files": [],
            "backward_files": [],
            "outlog": outlog,
            "errlog": errlog,
        }
        for i, pth in enumerate(sys_paths):
            try:
                atom_data = np.load(f"{str(pth)}/atom.npy")
            except FileNotFoundError:
                atom_data = coord_to_atom_fn(str(pth))
            nframes = atom_data.shape[0]
            for f in range(nframes):
                singletask["command"] = str(
                    f"cd {sys_name[i]}/ABACUS/{f}/ &&  "
                    f"{run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog}  &&  "
                    f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv  &&  "
                    f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log`"
                )
                singletask["task_work_path"] = "."
                singletask["forward_files"] = [str(f"./{sys_name[i]}/ABACUS/{f}/")]
                singletask["backward_files"] = [str(f"./{sys_name[i]}/ABACUS/{f}/")]
                task_list.append(Task.load_from_dict(singletask))
        return DPDispatcherTask(
            task_list,
            work_base="systems",
            outlog=outlog,
            share_folder=share_folder,
            link_share_files=link_share,
            link_prev_files=link_prev,
            link_abs_files=link_abs,
            machine=dpdispatcher_machine,
            resources=dpdispatcher_resources,
            forward_files=forward_files,
            backward_files=backward_files,
        )

    batch_tasks = []
    for i, pth in enumerate(sys_paths):
        try:
            atom_data = np.load(f"{str(pth)}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom_fn(str(pth))
        nframes = atom_data.shape[0]
        for f in range(nframes):
            batch_tasks.append(BatchTask(
                cmds=str(
                    f"cd {sys_name[i]}/ABACUS/{f}/ &&  "
                    f"{run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog}  &&  "
                    f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv  &&  "
                    f"echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log`"
                ),
                workdir="systems",
                forward_files=[str(f"./{sys_name[i]}/ABACUS/{f}/")],
                backward_files=[str(f"./{sys_name[i]}/ABACUS/{f}/")],
            ))
    return GroupBatchTask(
        batch_tasks,
        group_size=group_size,
        workdir="./",
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        link_abs_files=link_abs,
        forward_files=forward_files,
        backward_files=backward_files,
    )


def make_stat_scf_abacus(systems_train, systems_test=None, *,
                         train_dump="data_train", test_dump="data_test",
                         cal_force=0, cal_stress=0, deepks_bandgap=0,
                         deepks_v_delta=0, deepks_scf=0,
                         workdir=".", outlog="log.data", **stat_args):
    systems_train = [os.path.abspath(s) for s in systems_train]
    systems_test = [os.path.abspath(s) for s in systems_test] if systems_test else []
    if not systems_test:
        systems_test.append(systems_train[-1])
    stat_args.update(
        systems_train=systems_train,
        systems_test=systems_test,
        train_dump=train_dump,
        test_dump=test_dump,
        cal_force=cal_force,
        cal_stress=cal_stress,
        deepks_bandgap=deepks_bandgap,
        deepks_v_delta=deepks_v_delta,
        deepks_scf=deepks_scf,
    )
    return PythonTask(
        gather_stats_abacus,
        call_kwargs=stat_args,
        outlog=outlog,
        errlog="err",
        workdir=workdir,
    )
