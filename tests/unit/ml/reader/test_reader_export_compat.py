"""
Reader exports canonical coverage.

Tests:
- `test_reader_canonical_exports`
"""

from deepks.io.readers import GroupReader, Reader, SimpleReader
from deepks.io.readers.grouped_reader import GroupReader as GroupReaderImpl
from deepks.io.readers.reader import Reader as ReaderImpl
from deepks.io.readers.simple_reader import SimpleReader as SimpleReaderImpl


def test_reader_canonical_exports():
    """Canonical exports should point to implementation classes."""
    assert Reader is ReaderImpl
    assert GroupReader is GroupReaderImpl
    assert SimpleReader is SimpleReaderImpl
