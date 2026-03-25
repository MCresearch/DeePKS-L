import os
import sys
import numpy as np
from glob import glob
from deepks.workflows.defaults import SCF_CMD, DEFAULT_SCF_RES, DEFAULT_SCF_SUB_RES, DEFAULT_TRN_RES, DEFAULT_DPDISPATCHER_RES
from deepks.io.utils import check_list
from deepks.io.utils import flat_file_list
from deepks.io.utils import get_sys_name, load_sys_paths, dump_yaml_str
from deepks.orchestration.workflow.task import PythonTask, ShellTask
from deepks.orchestration.workflow.task import BatchTask, GroupBatchTask, DPDispatcherTask
from deepks.orchestration.workflow.workflow import Sequence

def check_system_names(systems):
    sys_names = [get_sys_name(os.path.basename(s)) for s in systems]
    if len(set(sys_names)) != len(systems):
        raise ValueError("Systems have duplicated base names. Not supported yet.")


def make_cleanup(pattern="slurm-*.out", workdir=".", **task_args):
    pattern = check_list(pattern)
    pattern = " ".join(pattern)
    assert pattern
    return ShellTask(
        f"rm -r {pattern}",
        workdir=workdir,
        **task_args
    )


def make_scf_task(*, workdir=".",
                  arg_file="scf_input.yaml", source_arg=None,
                  model_file="model.pth", source_model=None,
                  proj_basis=None, source_pbasis=None,
                  systems="systems.raw", link_systems=True,
                  dump_dir="results", share_folder="share",
                  outlog="log.scf", group_data=None,
                  dispatcher=None, resources=None,
                  python="python", **task_args):
    """Create a per-task SCF BatchTask that runs via the unified 'deepks' CLI.

    The task YAML (_scf_task.yaml) is written to the workdir during preprocess()
    (before execute()), so no shell one-liner is needed at runtime.
    """
    link_share = task_args.pop("link_share_files", [])
    link_prev  = task_args.pop("link_prev_files",  [])
    link_abs   = task_args.pop("link_abs_files",   [])
    forward_files  = task_args.pop("forward_files",  [])
    backward_files = task_args.pop("backward_files", [])

    # --- systems: link abs paths into task dir, build sys_str ---
    sys_name = None
    sys_str  = None
    if systems:
        sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
        sys_base  = [get_sys_name(s) for s in sys_paths]
        sys_name  = [os.path.basename(s) for s in sys_base]
        if link_systems:
            target_dir = "systems"
            src_files = sum((glob(f"{base}*") for base in sys_base), [])
            for fl in src_files:
                dst = os.path.join(target_dir, os.path.basename(fl))
                link_abs.append((fl, dst))
            forward_files.append(target_dir)
            sys_str = os.path.join(target_dir, "*")
        else:
            sys_str = " ".join(sys_paths)

    # --- file links ---
    if arg_file and source_arg is not None:
        link_share.append((source_arg, arg_file))
    if arg_file:
        forward_files.append(arg_file)
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

    # --- build per-task override dict ---
    TASK_YAML = "_scf_task.yaml"
    overrides = {"type": "scf_task"}
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

    # Write the YAML content at construction time; BatchTask.preprocess() will
    # write it to disk before the shell command runs.
    task_yaml_content = dump_yaml_str(overrides)
    command = f"{SCF_CMD} {TASK_YAML}"

    return BatchTask(
        command,
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
        write_files={TASK_YAML: task_yaml_content},
        **task_args
    )


def make_run_scf(systems_train, systems_test=None, *,
                 train_dump="data_train", test_dump="data_test",
                 no_model=False, group_data=None,
                 workdir='.', share_folder='share', outlog="log.scf",
                 source_arg="scf_input.yaml", source_model="model.pth",
                 source_pbasis=None, dispatcher=None, resources=None,
                 sub_size=1, group_size=1, ingroup_parallel=1,
                 sub_res=None, python='python', **task_args):
    # if no test systems, use last one in train systems
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
    check_system_names(systems_train)
    check_system_names(systems_test)
    # split systems into groups
    nsys_trn = len(systems_train)
    nsys_tst = len(systems_test)
    ntask_trn = int(np.ceil(nsys_trn / sub_size))
    ntask_tst = int(np.ceil(nsys_tst / sub_size))
    train_sets = [systems_train[i::ntask_trn] for i in range(ntask_trn)]
    test_sets = [systems_test[i::ntask_tst] for i in range(ntask_tst)]
    # make subtasks
    model_file = "../model.pth" if not no_model else "NONE"
    proj_basis = "../proj_basis.npz" if source_pbasis else None
    nd = max(len(str(ntask_trn+ntask_tst)), 2)
    if sub_res is None:
        sub_res = {}
    sub_res = {**DEFAULT_SCF_SUB_RES, **sub_res}
    trn_tasks = [
        make_scf_task(systems=sset, workdir=f"task.trn.{i:0{nd}}",
                      arg_file="../scf_input.yaml", source_arg=None,
                      model_file=model_file, source_model=None,
                      proj_basis=proj_basis, source_pbasis=None,
                      dump_dir=f"../{train_dump}", group_data=group_data,
                      link_systems=True, resources=sub_res, python=python)
        for i, sset in enumerate(train_sets)
    ]
    tst_tasks = [
        make_scf_task(systems=sset, workdir=f"task.tst.{i:0{nd}}",
                      arg_file="../scf_input.yaml", source_arg=None,
                      model_file=model_file, source_model=None,
                      proj_basis=proj_basis, source_pbasis=None,
                      dump_dir=f"../{test_dump}", group_data=group_data,
                      link_systems=True, resources=sub_res, python=python)
        for i, sset in enumerate(test_sets)
    ]
    # set up optional args
    link_share = task_args.pop("link_share_files", [])
    link_share.append((source_arg, "scf_input.yaml"))
    if source_pbasis:
        link_share.append((source_pbasis, "proj_basis.npz"))
    link_prev = task_args.pop("link_prev_files", [])
    if not no_model:
        link_prev.append((source_model, "model.pth"))
    if resources is None:
        resources = {}
    resources = {**DEFAULT_SCF_RES, "numb_node": ingroup_parallel, **resources}
    # make task
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
        link_prev_files=link_prev
    )


def make_stat_scf(systems_train, systems_test=None, *,
                  train_dump="data_train", test_dump="data_test", group_data=False,
                  workdir='.', outlog="log.data", **stat_args):
    # follow same convention for systems as run_scf
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
    # load stats function
    from deepks.physics.backends.stats import print_stats
    stat_args.update(
        systems=systems_train,
        test_sys=systems_test,
        dump_dir=train_dump,
        test_dump=test_dump,
        group=group_data)
    # make task
    return PythonTask(
        print_stats,
        call_kwargs=stat_args,
        outlog=outlog,
        errlog="err",
        workdir=workdir
    )


def make_scf(systems_train, systems_test=None, *,
             train_dump="data_train", test_dump="data_test",
             no_model=False, workdir='00.scf', share_folder='share',
             source_arg="scf_input.yaml", source_model="model.pth",
             source_pbasis=None, dispatcher=None, resources=None,
             dpdispatcher_machine=None, dpdispatcher_resources=None,
             sub_size=1, group_size=1, ingroup_parallel=1,
             sub_res=None, python='python',
             cleanup=False, **task_args):
    run_scf = make_run_scf(
        systems_train, systems_test,
        train_dump=train_dump, test_dump=test_dump,
        no_model=no_model, group_data=False,
        workdir=".", outlog="log.scf", share_folder=share_folder,
        source_arg=source_arg, source_model=source_model, source_pbasis=source_pbasis,
        dispatcher=dispatcher, resources=resources,
        dpdispatcher_machine=dpdispatcher_machine,
        dpdispatcher_resources=dpdispatcher_resources,
        group_size=group_size, ingroup_parallel=ingroup_parallel,
        sub_size=sub_size, sub_res=sub_res, python=python, **task_args
    )
    post_scf = make_stat_scf(
        systems_train=systems_train, systems_test=systems_test,
        train_dump=train_dump, test_dump=test_dump, workdir=".",
        outlog="log.data", group_data=False
    )
    # concat
    seq = [run_scf, post_scf]
    if cleanup:
        clean_scf = make_cleanup(
            ["slurm-*.out", "task.*/err", "fin.record"],
            workdir=".")
        seq.append(clean_scf)
    # make sequence
    return Sequence(
        seq,
        workdir=workdir
    )


def make_train_task(*, workdir=".",
                    arg_file="train_input.yaml", source_arg=None,
                    restart_model=None, source_model=None,
                    proj_basis=None, source_pbasis=None,
                    save_model="model.pth", group_data=False,
                    data_train="data_train", source_train=None,
                    data_test="data_test", source_test=None,
                    share_folder="share", outlog="log.train",
                    dispatcher=None, resources=None,
                    dpdispatcher_machine=None,
                    dpdispatcher_resources=None,
                    python="python", **task_args):
    """Create a training task as a BatchTask running via the unified 'deepks' CLI.

    All training parameters are folded into a per-task YAML (_train_task.yaml)
    with type: train_task.  The YAML is written to disk during preprocess()
    (not via a shell one-liner), so the command is simply:
        deepks _train_task.yaml
    """
    link_share = task_args.pop("link_share_files", [])
    link_prev  = task_args.pop("link_prev_files",  [])
    forward_files  = task_args.pop("forward_files",  [])
    backward_files = task_args.pop("backward_files", [])

    # --- file links ---
    if arg_file and source_arg is not None:
        link_share.append((source_arg, arg_file))
    if arg_file:
        forward_files.append(arg_file)
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

    # --- build per-task YAML overrides ---
    TASK_YAML = "_train_task.yaml"
    overrides = {"type": "train_task"}
    if data_train:
        overrides["train_paths"] = os.path.join(data_train, "*")
    if data_test:
        overrides["test_paths"] = os.path.join(data_test, "*")
    if restart_model:
        overrides["restart"] = restart_model
    if proj_basis:
        overrides["proj_basis"] = proj_basis
    if save_model:
        overrides.setdefault("train_args", {})["ckpt_file"] = save_model

    # Write the YAML at construction time; BatchTask.preprocess() writes it to
    # disk before the shell command runs — no fragile python -c one-liner needed.
    task_yaml_content = dump_yaml_str(overrides)
    command = f"{SCF_CMD} {TASK_YAML}"

    return BatchTask(
        command,
        workdir=workdir,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        errlog='err',
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        forward_files=forward_files,
        backward_files=backward_files,
        write_files={TASK_YAML: task_yaml_content},
        **task_args
    )


def make_run_train(source_train="data_train", source_test="data_test", *,
                   restart=True, source_model="model.pth", save_model="model.pth",
                   source_pbasis=None, source_arg="train_input.yaml",
                   workdir=".", share_folder="share", outlog="log.train",
                   dispatcher=None, resources=None,
                   dpdispatcher_machine=None,
                   python="python", **task_args):
    # just add some presetted arguments of make_train_task
    # have not implement parrallel training for now
    return make_train_task(
        workdir=workdir,
        arg_file="train_input.yaml",
        source_arg=source_arg,
        restart_model="model.pth" if restart else None,
        source_model=source_model if restart else None,
        proj_basis=None,
        source_pbasis=source_pbasis,
        save_model=save_model,
        group_data=False,
        data_train="data_train",
        source_train=source_train,
        data_test="data_test",
        source_test=source_test,
        share_folder=share_folder,
        outlog=outlog,
        dispatcher=dispatcher,
        resources=resources,
        dpdispatcher_machine=dpdispatcher_machine,
        python=python,
        **task_args
    )


def make_train(source_train="data_train", source_test="data_test", *,
               restart=True, source_model="model.pth", save_model="model.pth",
               source_pbasis=None, source_arg="train_input.yaml",
               workdir="01.train", share_folder="share",
               dispatcher=None, resources=None,
               dpdispatcher_machine=None,
               python="python", cleanup=False, **task_args):
    run_train = make_run_train(
        source_train=source_train, source_test=source_test,
        restart=restart, source_model=source_model, save_model=save_model,
        source_pbasis=source_pbasis, source_arg=source_arg,
        workdir=".", share_folder=share_folder, outlog="log.train",
        dispatcher=dispatcher, resources=resources,
        dpdispatcher_machine=dpdispatcher_machine,
        python=python, **task_args
    )
    seq = [run_train]
    if cleanup:
        clean_train = make_cleanup(
            ["slurm-*.out", "err", "fin.record"],
            workdir=".")
        seq.append(clean_train)
    return Sequence(
        seq,
        workdir=workdir
    )


def make_iterate(systems_train, systems_test=None, *,
                 n_iter=0, train_dump="data_train", test_dump="data_test",
                 no_model=False, workdir="iter.{n_iter:02d}",
                 share_folder="share", cleanup=False,
                 scf_workdir="00.scf", train_workdir="01.train",
                 source_arg="scf_input.yaml", source_model="model.pth",
                 source_pbasis=None,
                 scf_dispatcher=None, scf_resources=None,
                 scf_dpdispatcher_machine=None, scf_dpdispatcher_resources=None,
                 scf_sub_size=1, scf_group_size=1, scf_ingroup_parallel=1,
                 scf_sub_res=None, scf_python="python",
                 train_source_train="data_train", train_source_test="data_test",
                 train_restart=True, train_source_model="model.pth",
                 train_save_model="model.pth", train_source_pbasis=None,
                 train_source_arg="train_input.yaml",
                 train_dispatcher=None, train_resources=None,
                 train_dpdispatcher_machine=None,
                 train_python="python",
                 **task_args):
    scf = make_scf(
        systems_train=systems_train, systems_test=systems_test,
        train_dump=train_dump, test_dump=test_dump,
        no_model=no_model, workdir=scf_workdir,
        share_folder=os.path.join("..", share_folder),
        source_arg=os.path.join("..", source_arg),
        source_model=os.path.join("..", source_model) if not no_model else None,
        source_pbasis=os.path.join("..", source_pbasis) if source_pbasis else None,
        dispatcher=scf_dispatcher, resources=scf_resources,
        dpdispatcher_machine=scf_dpdispatcher_machine,
        dpdispatcher_resources=scf_dpdispatcher_resources,
        sub_size=scf_sub_size, group_size=scf_group_size,
        ingroup_parallel=scf_ingroup_parallel,
        sub_res=scf_sub_res, python=scf_python,
        cleanup=cleanup
    )
    train = make_train(
        source_train=os.path.join("..", scf_workdir, train_dump),
        source_test=os.path.join("..", scf_workdir, test_dump),
        restart=train_restart,
        source_model=os.path.join("..", train_source_model),
        save_model=train_save_model,
        source_pbasis=os.path.join("..", train_source_pbasis) if train_source_pbasis else None,
        source_arg=os.path.join("..", train_source_arg),
        workdir=train_workdir,
        share_folder=os.path.join("..", share_folder),
        dispatcher=train_dispatcher, resources=train_resources,
        dpdispatcher_machine=train_dpdispatcher_machine,
        python=train_python,
        cleanup=cleanup
    )
    actual_workdir = workdir.format(n_iter=n_iter)
    return Sequence(
        [scf, train],
        workdir=actual_workdir
    )
