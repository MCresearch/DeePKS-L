"""Compatibility shim for the legacy model test module."""

import importlib


_impl = importlib.import_module("deepks.core.ml.eval.test")


def __getattr__(name):
	return getattr(_impl, name)


def __dir__():
	return sorted(set(globals()) | set(__all__))


__all__ = [name for name in dir(_impl) if not name.startswith("_")]
