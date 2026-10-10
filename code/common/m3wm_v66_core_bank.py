from __future__ import annotations
from typing import Any
from common.m3wm_identity_event_artifact import identity_event_payload_sha256

SCHEMA = "m3wm_v66_faci_core_bank"
VERSION = 1
DEFAULT_CAPACITY = 64


def _i(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {name}")
    return int(value)


def empty_core_bank(*, capacity: int = DEFAULT_CAPACITY) -> dict[str, Any]:
    cap = _i(capacity, "CORE capacity", 1)
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "capacity": cap,
        "contexts": {},
        "task_blind": True,
        "append_only": True,
        "may_create": False,
        "may_recall": False,
        "may_mutate_context_memory": False,
        "may_mutate_replay": False,
    }
    return {**payload, "payload_sha256": identity_event_payload_sha256(payload)}
