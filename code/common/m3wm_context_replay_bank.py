from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextReplayRow:
    transition_id: int
    behavior_context_id: int
    assigned_context_id: int
    episode_id: int
    segment_id: int
    sequence_id: int
    terminated: bool
    truncated: bool
    payload: Any


class ContextReplayBank:
    schema = "m3wm-context-replay-bank-v1"

    def __init__(self, *, max_rows: int = 0):
        self.max_rows = int(max_rows)
        if self.max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        self._rows: dict[int, list[ContextReplayRow]] = defaultdict(list)
        self._next_transition_id = 0
        self._active_context_id = 0
        self._segment_id = 0
        self._sequence_by_episode: dict[tuple[int, int], int] = {}
        self._switch_count = 0

    @property
    def active_context_id(self) -> int:
        return self._active_context_id

    @property
    def segment_id(self) -> int:
        return self._segment_id

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def switch(self, target_context_id: int) -> dict[str, int]:
        target = int(target_context_id)
        if target < 0:
            raise ValueError("Context id must be non-negative")
        source = self._active_context_id
        if source == target:
            raise ValueError("Context switch requires distinct source and target")
        self._active_context_id = target
        self._segment_id += 1
        self._switch_count += 1
        return {
            "source_context_id": source,
            "target_context_id": target,
            "segment_id": self._segment_id,
        }

    def append(
        self,
        payload: Any,
        *,
        behavior_context_id: int,
        assigned_context_id: int | None = None,
        episode_id: int,
        terminated: bool = False,
        truncated: bool = False,
    ) -> ContextReplayRow:
        behavior = int(behavior_context_id)
        assigned = behavior if assigned_context_id is None else int(assigned_context_id)
        if behavior < 0 or assigned < 0 or int(episode_id) < 0:
            raise ValueError("Context and episode ids must be non-negative")
        if assigned != self._active_context_id:
            raise ValueError("assigned Context must be the committed active Context")
        key = (int(episode_id), self._segment_id)
        sequence = self._sequence_by_episode.get(key, 0)
        row = ContextReplayRow(
            transition_id=self._next_transition_id,
            behavior_context_id=behavior,
            assigned_context_id=assigned,
            episode_id=int(episode_id),
            segment_id=self._segment_id,
            sequence_id=sequence,
            terminated=bool(terminated),
            truncated=bool(truncated),
            payload=None,
        )
        self._sequence_by_episode[key] = sequence + 1
        self._next_transition_id += 1
        self._rows[assigned].append(row)
        if self.max_rows:
            while self.total_rows > self.max_rows:
                oldest_context = min(
                    (context for (context, rows) in self._rows.items() if rows),
                    key=lambda context: self._rows[context][0].transition_id,
                )
                self._rows[oldest_context].pop(0)
        return row

    @property
    def total_rows(self) -> int:
        return sum((len(rows) for rows in self._rows.values()))

    def rows(self, context_id: int) -> tuple[ContextReplayRow, ...]:
        return tuple(self._rows.get(int(context_id), ()))

    def discard_contexts_from(self, first_context_id: int) -> tuple[int, ...]:
        first = int(first_context_id)
        if first <= 0:
            raise ValueError("cannot discard Context 0 or use a negative Context id")
        discarded = tuple(
            sorted((context for context in self._rows if context >= first))
        )
        for context in discarded:
            self._rows.pop(context, None)
        return discarded

    def sample_sequences(
        self, context_id: int, *, length: int
    ) -> list[tuple[ContextReplayRow, ...]]:
        length = int(length)
        if length <= 0:
            raise ValueError("sequence length must be positive")
        rows = self.rows(context_id)
        sequences: list[tuple[ContextReplayRow, ...]] = []
        for start in range(0, max(0, len(rows) - length + 1)):
            candidate = rows[start : start + length]
            first = candidate[0]
            if any(
                (
                    row.assigned_context_id != first.assigned_context_id
                    or row.segment_id != first.segment_id
                    or row.episode_id != first.episode_id
                    or (row.sequence_id != first.sequence_id + index)
                    for (index, row) in enumerate(candidate)
                )
            ):
                continue
            sequences.append(candidate)
        return sequences

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "max_rows": self.max_rows,
            "next_transition_id": self._next_transition_id,
            "active_context_id": self._active_context_id,
            "segment_id": self._segment_id,
            "switch_count": self._switch_count,
            "sequence_by_episode": dict(self._sequence_by_episode),
            "rows": {context: tuple(rows) for (context, rows) in self._rows.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != self.schema:
            raise ValueError("Unsupported Context replay schema")
        self.max_rows = int(state["max_rows"])
        self._next_transition_id = int(state["next_transition_id"])
        self._active_context_id = int(state["active_context_id"])
        self._segment_id = int(state["segment_id"])
        self._switch_count = int(state["switch_count"])
        self._sequence_by_episode = {
            tuple(key): int(value)
            for (key, value) in state["sequence_by_episode"].items()
        }
        self._rows = defaultdict(
            list,
            {int(context): list(rows) for (context, rows) in state["rows"].items()},
        )
