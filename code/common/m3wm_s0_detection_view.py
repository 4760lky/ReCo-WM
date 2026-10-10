from __future__ import annotations
import copy
from collections import deque
from typing import Any, Mapping
import torch


class DetectionTrainingView:
    schema = "m3wm-s0-detection-training-view-v1"

    def __init__(
        self,
        *,
        recent_capacity: int = 1024,
        reservoir_capacity: int = 4096,
        seed: int = 0,
    ) -> None:
        if int(recent_capacity) <= 0 or int(reservoir_capacity) <= 0:
            raise ValueError("detection view capacities must be positive")
        self.recent_capacity = int(recent_capacity)
        self.reservoir_capacity = int(reservoir_capacity)
        self.seed = int(seed)
        self._recent: deque[dict[str, Any]] = deque(maxlen=self.recent_capacity)
        self._reservoir: list[dict[str, Any]] = []
        self._seen = 0
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._last_sample_stats: dict[str, int] = {
            "requested": 0,
            "recent": 0,
            "reservoir": 0,
            "fill": 0,
            "unique": 0,
        }

    def __len__(self) -> int:
        return self._seen

    @property
    def seen(self) -> int:
        return self._seen

    def recent_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(tuple(self._recent)))

    def reservoir_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._reservoir))

    @property
    def last_sample_stats(self) -> dict[str, int]:
        return dict(self._last_sample_stats)

    def add(self, row: Mapping[str, Any]) -> None:
        stored = copy.deepcopy(dict(row))
        transition_id = int(stored["transition_id"])
        if self._recent and transition_id <= int(self._recent[-1]["transition_id"]):
            raise ValueError("detection transition ids must increase strictly")
        self._seen += 1
        self._recent.append(stored)
        if len(self._reservoir) < self.reservoir_capacity:
            self._reservoir.append(copy.deepcopy(stored))
            return
        replacement = int(
            torch.randint(0, self._seen, (1,), generator=self._generator).item()
        )
        if replacement < self.reservoir_capacity:
            self._reservoir[replacement] = copy.deepcopy(stored)

    def _choose(
        self,
        rows: list[dict[str, Any]],
        count: int,
        *,
        generator: torch.Generator | None = None,
        copy_rows: bool = True,
    ) -> list[dict[str, Any]]:
        count = min(max(0, int(count)), len(rows))
        if count == 0:
            return []
        if count == len(rows):
            return [copy.deepcopy(row) for row in rows] if copy_rows else list(rows)
        order = torch.randperm(
            len(rows), generator=self._generator if generator is None else generator
        )[:count].tolist()
        selected = [rows[index] for index in order]
        return copy.deepcopy(selected) if copy_rows else selected

    def _select_training_rows(
        self, *, max_rows: int, generator: torch.Generator | None, copy_rows: bool
    ) -> list[dict[str, Any]]:
        requested = int(max_rows)
        if requested <= 0:
            raise ValueError("training view size must be positive")
        recent = list(self._recent)
        reservoir = list(self._reservoir)
        recent_quota = (requested + 1) // 2
        reservoir_quota = requested // 2
        selected = self._choose(
            recent, recent_quota, generator=generator, copy_rows=copy_rows
        )
        used = {int(row["transition_id"]) for row in selected}
        reservoir_available = [
            row for row in reservoir if int(row["transition_id"]) not in used
        ]
        reservoir_selected = self._choose(
            reservoir_available,
            reservoir_quota,
            generator=generator,
            copy_rows=copy_rows,
        )
        selected.extend(reservoir_selected)
        used.update((int(row["transition_id"]) for row in reservoir_selected))
        fill_candidates = [
            row for row in recent + reservoir if int(row["transition_id"]) not in used
        ]
        fill = self._choose(
            fill_candidates,
            requested - len(selected),
            generator=generator,
            copy_rows=copy_rows,
        )
        selected.extend(fill)
        self._last_sample_stats = {
            "requested": requested,
            "recent": min(recent_quota, len(recent)),
            "reservoir": len(reservoir_selected),
            "fill": len(fill),
            "unique": len(selected),
        }
        return selected

    def training_rows(
        self, *, max_rows: int, generator: torch.Generator | None = None
    ) -> list[dict[str, Any]]:
        return self._select_training_rows(
            max_rows=max_rows, generator=generator, copy_rows=True
        )

    def _borrow_training_rows(
        self, *, max_rows: int, generator: torch.Generator | None = None
    ) -> list[dict[str, Any]]:
        return self._select_training_rows(
            max_rows=max_rows, generator=generator, copy_rows=False
        )

    def all_unique_rows(self) -> list[dict[str, Any]]:
        by_id: dict[int, dict[str, Any]] = {}
        for row in list(self._reservoir) + list(self._recent):
            by_id[int(row["transition_id"])] = row
        return [copy.deepcopy(by_id[key]) for key in sorted(by_id)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "recent_capacity": self.recent_capacity,
            "reservoir_capacity": self.reservoir_capacity,
            "seed": self.seed,
            "seen": self._seen,
            "recent": copy.deepcopy(tuple(self._recent)),
            "reservoir": copy.deepcopy(tuple(self._reservoir)),
            "generator_state": self._generator.get_state().clone(),
            "last_sample_stats": dict(self._last_sample_stats),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.schema:
            raise ValueError("invalid detection training view schema")
        if int(state["recent_capacity"]) != self.recent_capacity:
            raise ValueError("recent detection capacity changed")
        if int(state["reservoir_capacity"]) != self.reservoir_capacity:
            raise ValueError("reservoir detection capacity changed")
        self.seed = int(state["seed"])
        self._seen = int(state["seen"])
        recent = copy.deepcopy(tuple(state["recent"]))
        reservoir = copy.deepcopy(list(state["reservoir"]))
        if (
            len(recent) > self.recent_capacity
            or len(reservoir) > self.reservoir_capacity
        ):
            raise ValueError("detection view payload exceeds configured capacity")
        self._recent = deque(recent, maxlen=self.recent_capacity)
        self._reservoir = reservoir
        self._generator.set_state(state["generator_state"].clone())
        self._last_sample_stats = dict(state.get("last_sample_stats", {}))
