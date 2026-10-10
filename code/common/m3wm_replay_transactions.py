from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StagedReplayRow:
    sequence_id: int
    segment_id: int
    trajectory_id: int
    payload: Any
