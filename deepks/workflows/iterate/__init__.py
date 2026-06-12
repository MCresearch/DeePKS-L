"""Iterate workflow package."""

__all__ = ["run_iterate_workflow"]


def __getattr__(name):
    if name == "run_iterate_workflow":
        from .workflow import run_iterate_workflow

        return run_iterate_workflow
    raise AttributeError(name)
