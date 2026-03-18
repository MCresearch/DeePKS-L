"""Compatibility coverage for phase-4 task/iterate package shims."""

import importlib


def test_task_package_exports():
    task_pkg = importlib.import_module("deepks.task")
    task_impl = importlib.import_module("deepks.orchestration.workflow.task")
    workflow_impl = importlib.import_module("deepks.orchestration.workflow.workflow")

    assert task_pkg.__all__ == ["task", "workflow", "job"]
    assert task_pkg.PythonTask is task_impl.PythonTask
    assert task_pkg.Sequence is workflow_impl.Sequence


def test_task_job_package_exports():
    job_pkg = importlib.import_module("deepks.task.job")
    dispatcher_impl = importlib.import_module("deepks.orchestration.scheduler.job.dispatcher")

    assert job_pkg.dispatcher.__name__ == "deepks.task.job.dispatcher"
    assert job_pkg.Dispatcher is dispatcher_impl.Dispatcher
    assert job_pkg._split_tasks is dispatcher_impl._split_tasks


def test_iterate_package_exports():
    iterate_pkg = importlib.import_module("deepks.iterate")
    iterate_impl = importlib.import_module("deepks.pipelines.iterate.iterate")

    assert iterate_pkg.__all__ == ["iterate", "template", "template_abacus", "generator_abacus"]
    assert iterate_pkg.make_iterate is iterate_impl.make_iterate
    assert iterate_pkg.make_scf is iterate_impl.make_scf
    assert iterate_pkg.make_train is iterate_impl.make_train
    assert iterate_pkg.make_scf_abacus is iterate_impl.make_scf_abacus


def test_iterate_package_submodule_lookup():
    iterate_pkg = importlib.import_module("deepks.iterate")

    assert iterate_pkg.iterate.__name__ == "deepks.iterate.iterate"
    assert iterate_pkg.template.__name__ == "deepks.iterate.template"
    assert iterate_pkg.template_abacus.__name__ == "deepks.iterate.template_abacus"
    assert iterate_pkg.generator_abacus.__name__ == "deepks.iterate.generator_abacus"
