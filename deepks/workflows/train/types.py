"""Data objects for the training workflow."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainResult:
    model_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    train_stats: Optional[Dict[str, Any]] = None
