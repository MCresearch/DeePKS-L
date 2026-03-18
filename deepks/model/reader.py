"""Compatibility shim for the legacy model reader module."""

from deepks.io.readers import GroupReader, Reader, SimpleReader

__all__ = ["Reader", "GroupReader", "SimpleReader"]
