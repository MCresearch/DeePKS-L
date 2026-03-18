"""Compatibility exports for legacy reader module path."""

from deepks.io.readers.grouped_reader import GroupReader
from deepks.io.readers.reader import Reader
from deepks.io.readers.simple_reader import SimpleReader

__all__ = ["Reader", "GroupReader", "SimpleReader"]