"""Model-sidecar artifact helpers kept outside the ML model classes."""

import os
from typing import Optional, Tuple

from deepks.io.utils import load_elem_table, save_elem_table


ElemTable = Tuple[object, object]


def elem_table_sidecar_path(model_file: str) -> str:
    """Return the sidecar path used for element-reference offsets."""

    return f"{model_file}.elemtab"


def save_elem_table_sidecar(model_file: str, elem_table: Optional[ElemTable]) -> None:
    """Persist element-reference offsets next to a model file."""

    if elem_table is None:
        return
    save_elem_table(elem_table_sidecar_path(model_file), elem_table)


def load_elem_table_sidecar(model_file: str) -> Optional[ElemTable]:
    """Load element-reference offsets if a sidecar exists."""

    sidecar = elem_table_sidecar_path(model_file)
    if not os.path.exists(sidecar):
        return None
    return load_elem_table(sidecar)
