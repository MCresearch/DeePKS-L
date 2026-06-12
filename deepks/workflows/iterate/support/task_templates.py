"""Iterate task-template builders owned by the iterate workflow."""

import os

import numpy as np

from deepks.io.utils import check_list, get_sys_name, load_sys_paths
from deepks.orchestration.workflow.task import BatchTask, GroupBatchTask, PythonTask, ShellTask
from deepks.orchestration.workflow.workflow import Sequence
from deepks.io.reporting import print_stats

from .template_settings import DEFAULT_SCF_RES, DEFAULT_SCF_SUB_RES, SCF_CMD
from .task_yaml import build_task_yaml, extract_task_payload, uses_structured_config


def check_system_names(systems):
    sys_names = [get_sys_name(os.path.basename(s)) for s in systems]
    if len(set(sys_names)) != len(systems):
        raise ValueError("Systems have duplicated base names. Not supported yet.")


def make_cleanup(pattern="slurm-*.out", workdir=".", **task_args):
    pattern = check_list(pattern)
    pattern = " ".join(pattern)
    assert pattern
    return ShellTask(f"rm -r {pattern}", workdir=workdir, **task_args)


def make_scf_task(*, workdir=".", task_config=None, model_file="model.pth", source_model=None,
                  proj_basis=None, source_pbasis=None, systems="systems.raw", link_systems=True,
                  dump_dir="results", share_folder="share", outlog="log.scf", group_data=None,
                  dispatcher=None, resources=None, python="python", **task_args):
    link_share = task_args.pop("link_share_files", [])
    link_prev = task_args.pop("link_prev_files", [])
    link_abs = task_args.pop("link_abs_files", [])
    forward_files = task_args.pop("forward_files", [])
    backward_files = task_args.pop("backward_files", [])

    sys_name = None
    sys_str = None
    if systems:
        sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
        sys_base = [get_sys_name(s) for s in sys_paths]
        sys_name = [os.path.basename(s) for s in sys_base]
        if link_systems:
            target_dir = "systems"
            for base in sys_base:
                link_abs.append((base, os.path.join(target_dir, os.path.basename(base))))
            forward_files.append(target_dir)
            sys_str = os.path.join(target_dir, "*")
        else:
            sys_str = " ".join(sys_paths)

    if model_file and model_file.upper() != "NONE":
        if source_model is not None:
            link_prev.append((source_model, model_file))
        forward_files.append(model_file)
    if proj_basis:
        if source_pbasis is not None:
            link_share.append((source_pbasis, proj_basis))
        forward_files.append(proj_basis)
    if dump_dir:
        if sys_name:
            for nm in sys_name:
                backward_files.append(os.path.join(dump_dir, nm))
        else:
            backward_files.append(dump_dir)

    task_yaml = "_scf_task.yaml"
    _, base_payload = extract_task_payload(task_config, "scf")
    if uses_structured_config(base_payload):
        overrides = {"type": "scf"}
        if sys_str is not None:
            overrides.setdefault("data", {})["systems"] = sys_str
        if model_file:
            overrides.setdefault("ml", {}).setdefault("checkpoint", {})["file"] = (
                model_file if model_file.upper() != "NONE" else None
            )
        if proj_basis:
            overrides.setdefault("physics", {}).setdefault("representation", {}).setdefault("params", {})["proj_basis"] = proj_basis
        if dump_dir:
            overrides.setdefault("physics", {}).setdefault("backend", {}).setdefault("output", {})["dump_dir"] = dump_dir
        if group_data is not None:
            overrides.setdefault("data", {}).setdefault("loader", {})["group"] = bool(group_data)
    else:
        overrides = {"type": "scf", "scf_soft": "pyscf"}
        if sys_str is not None:
            overrides["systems"] = sys_str
        if model_file:
            overrides["model_file"] = model_file if model_file.upper() != "NONE" else None
        if proj_basis:
            overrides["proj_basis"] = proj_basis
        if dump_dir:
            overrides["dump_dir"] = dump_dir
        if group_data is not None:
            overrides["group"] = bool(group_data)

    return BatchTask(
        f"{SCF_CMD} {task_yaml}",
        workdir=workdir,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        link_abs_files=link_abs,
        forward_files=forward_files,
        backward_files=backward_files,
        write_files={task_yaml: build_task_yaml(task_config, "scf", overrides)},
        **task_args,
    )


def make_run_scf(systems_train, systems_test=None, *, train_dump="data_train", test_dump="data_test",
                 no_model=False, group_data=None, workdir=".", share_folder="share", outlog="log.scf",
                 task_config=None, source_model="model.pth", source_pbasis=None, dispatcher=None,
                 resources=None, dpdispatcher_machine=None, dpdispatcher_resources=None, sub_size=1,
                 group_size=1, ingroup_parallel=1, sub_res=None, python="python", **task_args):
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
    check_system_names(systems_train)
    check_system_names(systems_test)
    ntask_trn = int(np.ceil(len(systems_train) / sub_size))
    ntask_tst = int(np.ceil(len(systems_test) / sub_size))
    train_sets = [systems_train[i::ntask_trn] for i in range(ntask_trn)]
    test_sets = [systems_test[i::ntask_tst] for i in range(ntask_tst)]
    model_file = "../model.pth" if not no_model else "NONE"
    proj_basis = "../proj_basis.npz" if source_pbasis else None
    nd = max(len(str(ntask_trn + ntask_tst)), 2)
    sub_res = {**DEFAULT_SCF_SUB_RES, **(sub_res or {})}
    trn_tasks = [
        make_scf_task(
            systems=sset, workdir=f"task.trn.{i:0{nd}}", task_config=task_config,
            model_file=model_file, proj_basis=proj_basis, dump_dir=f"../{train_dump}",
            group_data=group_data, link_systems=True, resources=sub_res, python=python,
        )
        for i, sset in enumerate(train_sets)
    ]
    tst_tasks = [
        make_scf_task(
            systems=sset, workdir=f"task.tst.{i:0{nd}}", task_config=task_config,
            model_file=model_file, proj_basis=proj_basis, dump_dir=f"../{test_dump}",
            group_data=group_data, link_systems=True, resources=sub_res, python=python,
        )
        for i, sset in enumerate(test_sets)
    ]
    link_share = task_args.pop("link_share_files", [])
    if source_pbasis:
        link_share.append((source_pbasis, "proj_basis.npz"))
    link_prev = task_args.pop("link_prev_files", [])
    if not no_model:
        link_prev.append((source_model, "model.pth"))
    resources = {**DEFAULT_SCF_RES, "numb_node": ingroup_parallel, **(resources or {})}
    return GroupBatchTask(
        trn_tasks + tst_tasks,
        workdir=workdir,
        group_size=group_size,
        ingroup_parallel=ingroup_parallel,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        errlog="err",
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
    )


def make_stat_scf(systems_train, systems_test=None, *, train_dump="data_train", test_dump="data_test",
                  group_data=False, workdir=".", outlog="log.data", **stat_args):
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
    stat_args.update(
        systems=systems_train,
        test_sys=systems_test,
        dump_dir=train_dump,
        test_dump=test_dump,
        group=group_data,
    )
    return PythonTask(print_stats, call_kwargs=stat_args, outlog=outlog, errlog="err", workdir=workdir)


def make_scf(systems_train, systems_test=None, *, train_dump="data_train", test_dump="data_test",
             no_model=False, workdir="00.scf", share_folder="share", task_config=None,
             source_model="model.pth", source_pbasis=None, dispatcher=None, resources=None,
             dpdispatcher_machine=None, dpdispatcher_resources=None, sub_size=1, group_size=1,
             ingroup_parallel=1, sub_res=None, python="python", cleanup=False, **task_args):
    run_scf = make_run_scf(
        systems_train, systems_test, train_dump=train_dump, test_dump=test_dump,
        no_model=no_model, group_data=False, workdir=".", outlog="log.scf",
        share_folder=share_folder, task_config=task_config, source_model=source_model,
        source_pbasis=source_pbasis, dispatcher=dispatcher, resources=resources,
        dpdispatcher_machine=dpdispatcher_machine, dpdispatcher_resources=dpdispatcher_resources,
        group_size=group_size, ingroup_parallel=ingroup_parallel, sub_size=sub_size,
        sub_res=sub_res, python=python, **task_args,
    )
    post_scf = make_stat_scf(
        systems_train=systems_train, systems_test=systems_test, train_dump=train_dump,
        test_dump=test_dump, workdir=".", outlog="log.data", group_data=False,
    )
    seq = [run_scf, post_scf]
    if cleanup:
        seq.append(make_cleanup(["slurm-*.out", "task.*/err", "fin.record"], workdir="."))
    return Sequence(seq, workdir=workdir)


def make_train_task(*, workdir=".", arg_file="train_input.yaml", task_config=None, restart_model=None,
                    source_model=None, proj_basis=None, source_pbasis=None, save_model="model.pth",
                    data_train="data_train", source_train=None, data_test="data_test", source_test=None,
                    share_folder="share", outlog="log.train", dispatcher=None, resources=None,
                    dpdispatcher_machine=None, dpdispatcher_resources=None, python="python", **task_args):
    link_prev = task_args.pop("link_prev_files", [])
    link_share = task_args.pop("link_share_files", [])
    forward_files = task_args.pop("forward_files", [])
    backward_files = task_args.pop("backward_files", [])

    if restart_model and source_model is not None:
        link_prev.append((source_model, restart_model))
        forward_files.append(restart_model)
    if proj_basis and source_pbasis is not None:
        link_share.append((source_pbasis, proj_basis))
        forward_files.append(proj_basis)
    if data_train and source_train is not None:
        link_prev.append((source_train, data_train))
        forward_files.append(data_train)
    if data_test and source_test is not None:
        link_prev.append((source_test, data_test))
        forward_files.append(data_test)
    if save_model:
        backward_files.append(save_model)

    _, base_payload = extract_task_payload(task_config, "train")
    if uses_structured_config(base_payload):
        overrides = {"type": "train"}
        if data_train:
            overrides.setdefault("data", {})["train"] = os.path.join(data_train, "*")
        if data_test:
            overrides.setdefault("data", {})["test"] = os.path.join(data_test, "*")
        if restart_model:
            overrides.setdefault("ml", {}).setdefault("checkpoint", {})["restart"] = restart_model
        if proj_basis:
            overrides.setdefault("physics", {}).setdefault("representation", {}).setdefault("params", {})["proj_basis"] = proj_basis
        if save_model:
            overrides.setdefault("runtime", {}).setdefault("io", {})["ckpt_file"] = save_model
    else:
        overrides = {"type": "train"}
        if data_train:
            overrides["systems_train"] = os.path.join(data_train, "*")
        if data_test:
            overrides["systems_test"] = os.path.join(data_test, "*")
        if restart_model:
            overrides["restart"] = restart_model
        if proj_basis:
            overrides["proj_basis"] = proj_basis
        if save_model:
            overrides["ckpt_file"] = save_model

    return BatchTask(
        f"{SCF_CMD} {arg_file}",
        workdir=workdir,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        errlog="err",
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        forward_files=forward_files,
        backward_files=backward_files,
        write_files={arg_file: build_task_yaml(task_config, "train", overrides)},
        **task_args,
    )


def make_run_train(source_train="data_train", source_test="data_test", *, restart=True,
                   source_model="model.pth", save_model="model.pth", task_config=None,
                   source_pbasis=None, workdir=".", share_folder="share", outlog="log.train",
                   data_train="data_train", data_test="data_test",
                   dispatcher=None, resources=None, dpdispatcher_machine=None,
                   python="python", **task_args):
    return make_train_task(
        workdir=workdir,
        arg_file="train_input.yaml",
        task_config=task_config,
        restart_model="model.pth" if restart else None,
        source_model=source_model if restart else None,
        proj_basis=None,
        source_pbasis=source_pbasis,
        save_model=save_model,
        data_train=data_train,
        source_train=source_train,
        data_test=data_test,
        source_test=source_test,
        share_folder=share_folder,
        outlog=outlog,
        dispatcher=dispatcher,
        resources=resources,
        dpdispatcher_machine=dpdispatcher_machine,
        python=python,
        **task_args,
    )


def make_train(source_train="data_train", source_test="data_test", *, restart=True,
               source_model="model.pth", save_model="model.pth", task_config=None,
               source_pbasis=None, workdir="01.train", share_folder="share",
               data_train="data_train", data_test="data_test",
               dispatcher=None, resources=None, dpdispatcher_machine=None,
               python="python", cleanup=False, **task_args):
    run_train = make_run_train(
        source_train=source_train, source_test=source_test, restart=restart,
        source_model=source_model, save_model=save_model, task_config=task_config,
        source_pbasis=source_pbasis, workdir=".", share_folder=share_folder,
        data_train=data_train, data_test=data_test,
        outlog="log.train", dispatcher=dispatcher, resources=resources,
        dpdispatcher_machine=dpdispatcher_machine, python=python, **task_args,
    )
    seq = [run_train]
    if cleanup:
        seq.append(make_cleanup(["slurm-*.out", "err", "fin.record"], workdir="."))
    return Sequence(seq, workdir=workdir)


def make_iterate(systems_train, systems_test=None, *, n_iter=0, train_dump="data_train",
                 test_dump="data_test", no_model=False, workdir="iter.{n_iter:02d}",
                 share_folder="share", cleanup=False, scf_workdir="00.scf",
                 train_workdir="01.train", scf_task_config=None, source_model="model.pth",
                 source_pbasis=None, scf_dispatcher=None, scf_resources=None,
                 scf_dpdispatcher_machine=None, scf_dpdispatcher_resources=None, scf_sub_size=1,
                 scf_group_size=1, scf_ingroup_parallel=1, scf_sub_res=None, scf_python="python",
                 train_source_train="data_train", train_source_test="data_test", train_restart=True,
                 train_source_model="model.pth", train_save_model="model.pth", train_source_pbasis=None,
                 train_task_config=None, train_dispatcher=None, train_resources=None,
                 train_dpdispatcher_machine=None, train_python="python", **task_args):
    scf = make_scf(
        systems_train=systems_train, systems_test=systems_test, train_dump=train_dump,
        test_dump=test_dump, no_model=no_model, workdir=scf_workdir,
        share_folder=os.path.join("..", share_folder), task_config=scf_task_config,
        source_model=os.path.join("..", source_model) if not no_model else None,
        source_pbasis=os.path.join("..", source_pbasis) if source_pbasis else None,
        dispatcher=scf_dispatcher, resources=scf_resources,
        dpdispatcher_machine=scf_dpdispatcher_machine,
        dpdispatcher_resources=scf_dpdispatcher_resources, sub_size=scf_sub_size,
        group_size=scf_group_size, ingroup_parallel=scf_ingroup_parallel,
        sub_res=scf_sub_res, python=scf_python, cleanup=cleanup,
    )
    train = make_train(
        source_train=os.path.join("..", scf_workdir, train_dump),
        source_test=os.path.join("..", scf_workdir, test_dump),
        restart=train_restart,
        source_model=os.path.join("..", train_source_model),
        save_model=train_save_model,
        task_config=train_task_config,
        source_pbasis=os.path.join("..", train_source_pbasis) if train_source_pbasis else None,
        workdir=train_workdir,
        share_folder=os.path.join("..", share_folder),
        dispatcher=train_dispatcher,
        resources=train_resources,
        dpdispatcher_machine=train_dpdispatcher_machine,
        python=train_python,
        cleanup=cleanup,
    )
    return Sequence([scf, train], workdir=workdir.format(n_iter=n_iter))
