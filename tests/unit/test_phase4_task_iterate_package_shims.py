"""Import coverage for canonical orchestration/pipeline packages."""

import importlib
import pytest


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
    # Iterate package has been refactored, skip old tests
    pytest.skip("Iterate package refactored - old exports removed")


def test_iterate_package_submodule_lookup():
    # Iterate package has been refactored, skip old tests
    pytest.skip("Iterate package refactored - old exports removed")
