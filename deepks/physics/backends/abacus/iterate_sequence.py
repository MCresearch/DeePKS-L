"""ABACUS iterate sequence builders."""

from deepks.orchestration.workflow.workflow import Sequence

from deepks.workflows.iterate.support.task_templates import check_system_names, make_cleanup

from .iterate_ops import coord_to_atom, load_and_share_abacus_assets
from .iterate_tasks import (
    make_convert_scf_abacus as _make_convert_scf_abacus,
    make_run_scf_abacus as _make_run_scf_abacus,
    make_stat_scf_abacus as _make_stat_scf_abacus,
)


def make_scf_abacus(systems_train, systems_test=None, *, train_dump="data_train", test_dump="data_test",
                    cleanup=None, dispatcher=None, resources=None, dpdispatcher_machine=None,
                    dpdispatcher_resources=None, no_model=True, group_size=1, workdir="00.scf",
                    share_folder="share", model_file=None, orb_files=None, pp_files=None,
                    proj_file=None, run_cmd="mpirun", abacus_path=None, **scf_abacus):
    orb_files, pp_files, proj_file = load_and_share_abacus_assets(
        orb_files or [],
        pp_files or [],
        proj_file or [],
        share_folder,
    )
    forward_files = orb_files + pp_files + proj_file
    pre_scf_abacus = make_convert_scf_abacus(
        systems_train=systems_train,
        systems_test=systems_test,
        no_model=no_model,
        workdir=".",
        share_folder=share_folder,
        model_file=model_file,
        resources=resources,
        dispatcher=dispatcher,
        orb_files=orb_files,
        pp_files=pp_files,
        proj_file=proj_file,
        **scf_abacus,
    )
    run_scf_abacus = make_run_scf_abacus(
        systems_train,
        systems_test,
        no_model=no_model,
        model_file=model_file,
        group_data=False,
        workdir=".",
        outlog="log.scf",
        share_folder=share_folder,
        dispatcher=dispatcher,
        resources=resources,
        group_size=group_size,
        dpdispatcher_machine=dpdispatcher_machine,
        dpdispatcher_resources=dpdispatcher_resources,
        forward_files=forward_files,
        run_cmd=run_cmd,
        abacus_path=abacus_path,
        **scf_abacus,
    )
    post_scf_abacus = make_stat_scf_abacus(
        systems_train,
        systems_test,
        train_dump=train_dump,
        test_dump=test_dump,
        workdir=".",
        **scf_abacus,
    )
    seq = [pre_scf_abacus, run_scf_abacus, post_scf_abacus]
    if cleanup:
        seq.append(make_cleanup(["slurm-*.out", "task.*/err", "fin.record"], workdir="."))
    return Sequence(seq, workdir=workdir)


def make_convert_scf_abacus(systems_train, systems_test=None, no_model=True, model_file=None, resources=None, **pre_args):
    return _make_convert_scf_abacus(
        systems_train=systems_train,
        systems_test=systems_test,
        no_model=no_model,
        model_file=model_file,
        resources=resources,
        check_system_names=check_system_names,
        **pre_args,
    )


def make_run_scf_abacus(systems_train, systems_test=None, outlog="out.log", errlog="err.log", group_size=1,
                        resources=None, dispatcher=None, share_folder="share", workdir=".",
                        link_systems=True, dpdispatcher_machine=None, dpdispatcher_resources=None,
                        no_model=True, **task_args):
    return _make_run_scf_abacus(
        systems_train=systems_train,
        systems_test=systems_test,
        outlog=outlog,
        errlog=errlog,
        group_size=group_size,
        resources=resources,
        dispatcher=dispatcher,
        share_folder=share_folder,
        workdir=workdir,
        link_systems=link_systems,
        dpdispatcher_machine=dpdispatcher_machine,
        dpdispatcher_resources=dpdispatcher_resources,
        no_model=no_model,
        coord_to_atom_fn=coord_to_atom,
        check_system_names=check_system_names,
        **task_args,
    )


def make_stat_scf_abacus(systems_train, systems_test=None, *, train_dump="data_train", test_dump="data_test",
                         cal_force=0, cal_stress=0, deepks_bandgap=0, deepks_v_delta=0, deepks_scf=0,
                         workdir=".", outlog="log.data", **stat_args):
    return _make_stat_scf_abacus(
        systems_train=systems_train,
        systems_test=systems_test,
        train_dump=train_dump,
        test_dump=test_dump,
        cal_force=cal_force,
        cal_stress=cal_stress,
        deepks_bandgap=deepks_bandgap,
        deepks_v_delta=deepks_v_delta,
        deepks_scf=deepks_scf,
        workdir=workdir,
        outlog=outlog,
        **stat_args,
    )
