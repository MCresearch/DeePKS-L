"""Compatibility shim to orchestration scheduler job module: dispatcher."""

from deepks.orchestration.scheduler.job import dispatcher as _impl

# Re-export public API and selected private helpers used by existing tests/code.
Dispatcher = _impl.Dispatcher
JobRecord = _impl.JobRecord
_split_tasks = _impl._split_tasks

__all__ = ["Dispatcher", "JobRecord", "_split_tasks"]
