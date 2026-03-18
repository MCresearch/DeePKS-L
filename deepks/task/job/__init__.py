"""Compatibility shim package for scheduler job modules."""

import importlib


_SUBMODULES = (
    "batch",
    "dispatcher",
    "job_status",
    "lazy_local_context",
    "local_context",
    "pbs",
    "shell",
    "slurm",
    "ssh_context",
)

__all__ = list(_SUBMODULES)


def _load_submodule(name):
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return importlib.import_module(f"{__name__}.{name}")


def __getattr__(name):
    if name in _SUBMODULES:
        return _load_submodule(name)

    for module_name in _SUBMODULES:
        module = _load_submodule(module_name)
        if hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    exported = set(globals()) | set(_SUBMODULES)
    for module_name in _SUBMODULES:
        try:
            module = _load_submodule(module_name)
        except Exception:
            continue
        exported.update(getattr(module, "__all__", [n for n in dir(module) if not n.startswith("_")]))
    return sorted(exported)
