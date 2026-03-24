"""Shared physical result types for DeePKS workflows."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PhysicalResult:
    """Minimal shared result container for physics calculations."""

    metadata: Dict[str, Any] = field(default_factory=dict)
