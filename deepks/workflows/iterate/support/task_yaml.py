"""Helpers for iterate task-yaml materialization."""

from typing import Any, Dict, Optional, Tuple

from deepks.config.packager import INTERNAL_PACKED_MARKER, get_payload_key, is_packed_config
from deepks.io.utils import deep_update, dump_yaml_str


def uses_structured_config(config: Any) -> bool:
    return isinstance(config, dict) and any(key in config for key in ("data", "physics", "ml", "runtime"))


def extract_task_payload(task_config: Optional[Dict[str, Any]], expected_type: str) -> Tuple[Optional[str], Dict[str, Any]]:
    if is_packed_config(task_config):
        payload_key = get_payload_key(expected_type)
        payload = task_config.get(payload_key)
        if not isinstance(payload, dict):
            raise ValueError(f"Packed task config missing '{payload_key}'")
        return payload_key, dict(payload)
    return None, dict(task_config or {})


def rebuild_task_config(task_config: Optional[Dict[str, Any]], expected_type: str,
                        payload_key: Optional[str], merged_payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload_key is None:
        return merged_payload
    merged = dict(task_config or {})
    merged[INTERNAL_PACKED_MARKER] = True
    merged["type"] = expected_type
    merged[payload_key] = merged_payload
    return merged


def build_task_yaml(task_config: Optional[Dict[str, Any]], expected_type: str,
                    overrides: Dict[str, Any]) -> str:
    payload_key, base_payload = extract_task_payload(task_config, expected_type)
    merged_payload = deep_update(base_payload, overrides)
    merged = rebuild_task_config(task_config, expected_type, payload_key, merged_payload)
    return dump_yaml_str(merged)
