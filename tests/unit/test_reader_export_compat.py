"""
Reader exports compatibility coverage.

Tests:
- `test_reader_export_identity`
"""

from deepks.io.readers import GroupReader as GroupReaderPkg
from deepks.io.readers import Reader as ReaderPkg
from deepks.io.readers import SimpleReader as SimpleReaderPkg
from deepks.io.readers.group_reader import GroupReader as GroupReaderLegacy
from deepks.io.readers.group_reader import Reader as ReaderLegacy
from deepks.io.readers.group_reader import SimpleReader as SimpleReaderLegacy
from deepks.io.readers import GroupReader as GroupReaderModelShim
from deepks.io.readers import Reader as ReaderModelShim
from deepks.io.readers import SimpleReader as SimpleReaderModelShim


def test_reader_export_identity():
    """Canonical exports and compatibility shims should point to identical class objects."""
    assert ReaderPkg is ReaderLegacy
    assert GroupReaderPkg is GroupReaderLegacy
    assert SimpleReaderPkg is SimpleReaderLegacy

    assert ReaderPkg is ReaderModelShim
    assert GroupReaderPkg is GroupReaderModelShim
    assert SimpleReaderPkg is SimpleReaderModelShim
