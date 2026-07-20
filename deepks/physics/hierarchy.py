"""Hierarchy helpers for level-wise staged supervision."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class HierarchyLevel:
    """One nested output level in a hierarchical target."""

    name: str
    output_dim: int
    target_shape: tuple[int, ...] | None = None


class HierarchyScheme:
    """Describe nested output spaces and per-level target shapes."""

    def __init__(self, levels: Sequence[HierarchyLevel | Dict[str, Any]]):
        parsed_levels: List[HierarchyLevel] = []
        for raw_level in levels:
            if isinstance(raw_level, HierarchyLevel):
                level = raw_level
            elif isinstance(raw_level, dict):
                level = HierarchyLevel(
                    name=str(raw_level.get("name", f"level_{len(parsed_levels)}")),
                    output_dim=int(raw_level["output_dim"]),
                    target_shape=tuple(int(v) for v in raw_level.get("target_shape", ()))
                    if raw_level.get("target_shape") is not None
                    else None,
                )
            else:
                raise TypeError(f"Unsupported hierarchy level: {type(raw_level)!r}")
            if level.output_dim <= 0:
                raise ValueError(f"Hierarchy level '{level.name}' must have positive output_dim")
            if level.target_shape is not None and len(level.target_shape) == 0:
                level = HierarchyLevel(name=level.name, output_dim=level.output_dim, target_shape=None)
            parsed_levels.append(level)
        if not parsed_levels:
            raise ValueError("HierarchyScheme requires at least one level")
        dims = [level.output_dim for level in parsed_levels]
        if dims != sorted(dims):
            raise ValueError(f"Hierarchy output dims must be non-decreasing, got {dims}")
        self._validate_target_shapes(parsed_levels)
        self.levels = tuple(parsed_levels)
        self._max_output_dim = dims[-1]
        self._masks = tuple(
            self._build_mask(level.output_dim, max_dim=self._max_output_dim) for level in self.levels
        )

    def num_levels(self) -> int:
        return len(self.levels)

    def max_output_dim(self) -> int:
        return self._max_output_dim

    def level_names(self) -> List[str]:
        return [level.name for level in self.levels]

    def level_mask(self, level_index: int, *, device=None) -> torch.Tensor:
        try:
            mask = self._masks[level_index]
        except IndexError as exc:
            raise IndexError(f"Hierarchy level index out of range: {level_index}") from exc
        if device is not None:
            return mask.to(device=device)
        return mask

    def level_output_dim(self, level_index: int) -> int:
        return self.levels[level_index].output_dim

    def level_target_shape(self, level_index: int) -> tuple[int, ...]:
        level = self.levels[level_index]
        if level.target_shape is not None:
            return level.target_shape
        return (level.output_dim, level.output_dim)

    def map_level_indices(self, level_index: int, row: torch.Tensor, col: torch.Tensor):
        """Map level-local sparse indices to the global max-dimension space."""

        level_dim = self.level_output_dim(level_index)
        if torch.any(row >= level_dim) or torch.any(col >= level_dim):
            raise ValueError(
                f"Sparse target indices exceed level-{level_index} dimension {level_dim}"
            )
        return row, col

    def embed_target(self, raw_target: torch.Tensor, level_index: int) -> torch.Tensor:
        level_dim = self.levels[level_index].output_dim
        max_dim = self._max_output_dim
        if raw_target.shape[-2:] == (max_dim, max_dim):
            return raw_target
        if raw_target.shape[-2:] != (level_dim, level_dim):
            raise ValueError(
                f"Target shape {tuple(raw_target.shape[-2:])} does not match level-{level_index} "
                f"dim {level_dim} or max dim {max_dim}"
            )
        padded = raw_target.new_zeros(*raw_target.shape[:-2], max_dim, max_dim)
        padded[..., :level_dim, :level_dim] = raw_target
        return padded

    @staticmethod
    def _build_mask(output_dim: int, *, max_dim: int | None = None) -> torch.Tensor:
        size = output_dim if max_dim is None else max_dim
        mask = torch.zeros(size, size, dtype=torch.bool)
        mask[:output_dim, :output_dim] = True
        return mask

    def __iter__(self) -> Iterable[HierarchyLevel]:
        return iter(self.levels)

    @staticmethod
    def _validate_target_shapes(levels: Sequence[HierarchyLevel]) -> None:
        prev_nr = None
        for level in levels:
            target_shape = level.target_shape
            if target_shape is None:
                continue
            if len(target_shape) not in {1, 2, 5}:
                raise ValueError(
                    f"Hierarchy level '{level.name}' target_shape must have rank 1, 2 or 5, got {target_shape}"
                )
            if len(target_shape) == 1:
                if int(target_shape[0]) <= 0:
                    raise ValueError(
                        f"Hierarchy level '{level.name}' 1D target_shape must be positive, got {target_shape}"
                    )
                continue
            if len(target_shape) == 2:
                if tuple(target_shape) != (level.output_dim, level.output_dim):
                    raise ValueError(
                        f"Hierarchy level '{level.name}' 2D target_shape must equal "
                        f"({level.output_dim}, {level.output_dim}), got {target_shape}"
                    )
                continue
            if tuple(target_shape[-2:]) != (level.output_dim, level.output_dim):
                raise ValueError(
                    f"Hierarchy level '{level.name}' target_shape tail must equal "
                    f"({level.output_dim}, {level.output_dim}), got {target_shape[-2:]}"
                )
            nr = tuple(int(v) for v in target_shape[:3])
            if prev_nr is not None and any(cur < prev for cur, prev in zip(nr, prev_nr)):
                raise ValueError(
                    f"Hierarchy nR ranges must be non-decreasing across levels; "
                    f"got previous {prev_nr} then {nr} at level '{level.name}'"
                )
            prev_nr = nr


def build_hierarchy_scheme(levels: Sequence[HierarchyLevel | Dict[str, Any]]) -> HierarchyScheme:
    """Build a hierarchy scheme from config-level metadata."""

    return HierarchyScheme(levels)
