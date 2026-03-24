"""Data objects for the iterative DeePKS workflow."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class IterationState:
    workdir: str
    iteration: int = 0
    final_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
