"""Import coverage for canonical orchestration/pipeline packages."""

import importlib


def test_task_package_exports():
    task_pkg = importlib.import_module("deepks.orchestration.workflow.task")
    task_impl = importlib.import_module("deepks.orchestration.workflow.task")
    workflow_impl = importlib.import_module("deepks.orchestration.workflow.workflow")

    assert task_pkg.PythonTask is task_impl.PythonTask
    assert task_pkg.GroupBatchTask.__module__ == "deepks.orchestration.workflow.task"
    assert workflow_impl.Sequence.__module__ == "deepks.orchestration.workflow.workflow"


def test_task_job_package_exports():
    job_pkg = importlib.import_module("deepks.orchestration.scheduler.job")
    dispatcher_mod = importlib.import_module("deepks.orchestration.scheduler.job.dispatcher")

    assert job_pkg.__name__ == "deepks.orchestration.scheduler.job"
    assert dispatcher_mod.Dispatcher.__module__ == "deepks.orchestration.scheduler.job.dispatcher"
    assert dispatcher_mod._split_tasks.__module__ == "deepks.orchestration.scheduler.job.dispatcher"


def test_iterate_package_exports():
    iterate_pkg = importlib.import_module("deepks.pipelines.iterate")
    iterate_impl = importlib.import_module("deepks.pipelines.iterate.iterate")

    assert iterate_pkg.__all__ == ["iterate", "template", "template_abacus", "generator_abacus"]
    assert iterate_pkg.make_iterate is iterate_impl.make_iterate
    assert iterate_pkg.make_scf is iterate_impl.make_scf
    assert iterate_pkg.make_train is iterate_impl.make_train
    assert iterate_pkg.make_scf_abacus is iterate_impl.make_scf_abacus


def test_iterate_package_submodule_lookup():
    iterate_pkg = importlib.import_module("deepks.pipelines.iterate")

    assert iterate_pkg.make_iterate.__module__ == "deepks.pipelines.iterate.iterate"
    assert iterate_pkg.make_scf.__module__ == "deepks.pipelines.iterate.template"
    assert iterate_pkg.make_scf_abacus.__module__ == "deepks.pipelines.iterate.template_abacus"
