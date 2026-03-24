"""Data objects for the SCF workflow."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SCFTask:
    system: str
    system_name: str
    workdir: str
    frame_dirs: List[str] = field(default_factory=list)
    backend: str = 'abacus'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SCFResult:
    dump_dir: str
    systems: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
