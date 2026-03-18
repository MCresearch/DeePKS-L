"""Compatibility shim for the legacy scf grad module."""

import importlib


_IMPL_MODULE = "deepks.core.physics.pyscf.grad"


def _load_impl():
    return importlib.import_module(_IMPL_MODULE)


def _exported_names():
    try:
        impl = _load_impl()
    except Exception:
        return []
    return getattr(impl, "__all__", [name for name in dir(impl) if not name.startswith("_")])


def __getattr__(name):
    if name == "__all__":
        return _exported_names()
    return getattr(_load_impl(), name)


def __dir__():
    return sorted(set(globals()) | set(_exported_names()))
