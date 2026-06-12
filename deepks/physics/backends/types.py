"""Backend result data objects."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SCFResult:
    dump_dir: str
    systems: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
