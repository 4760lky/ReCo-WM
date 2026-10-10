from __future__ import annotations
from collections import deque
from copy import deepcopy
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "m3wm_context_evidence_stream"
VERSION = 1


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and (not isinstance(value, bool))


def _clone(value: Any) -> Any:
    if value is None:
        return None
    clone = getattr(value, "detach", None)
    if callable(clone):
        try:
            return value.detach().clone().cpu()
        except (AttributeError, RuntimeError, TypeError):
            pass
    clone = getattr(value, "copy", None)
    if callable(clone):
        try:
            return value.copy()
        except (AttributeError, RuntimeError, TypeError):
            pass
    return deepcopy(value)


def _clone_transition(value: Any) -> Any:
    if not isinstance(value, (tuple, list)):
        return _clone(value)
    return tuple((_clone(item) for item in value))


class ContextEvidenceStream:
    schema = SCHEMA
    version = VERSION

    def __init__(
        self, capacity: int = 4096, *, episode_id: int = 0, context_generation: int = 0
    ) -> None:
        if not _is_integer(capacity) or int(capacity) < 1:
            raise ValueError("capacity must be a positive integer")
        if not _is_integer(episode_id) or int(episode_id) < 0:
            raise ValueError("episode_id must be a non-negative integer")
        if not _is_integer(context_generation) or int(context_generation) < 0:
            raise ValueError("context_generation must be a non-negative integer")
        self.capacity = int(capacity)
        self._records: deque[dict[str, Any]] = deque(maxlen=self.capacity)
        self._episode_id = int(episode_id)
        self._context_generation = int(context_generation)
        self._next_transition_id = 0
        self._last_step: int | None = None
        self._reset_boundary_pending = True
        self._pending_from_step: int | None = None
        self._revision = 0

    @property
    def episode_id(self) -> int:
        return int(self._episode_id)

    @property
    def context_generation(self) -> int:
        return int(self._context_generation)

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def valid_transition_count(self) -> int:
        return len(self._records)

    @property
    def pending_count(self) -> int:
        return sum((bool(row["pending"]) for row in self._records))

    @property
    def revision(self) -> int:
        return int(self._revision)

    def set_context_generation(
        self, generation: int, *, from_step: int | None = None
    ) -> None:
        if not _is_integer(generation) or int(generation) < 0:
            raise ValueError("context generation must be non-negative")
        generation = int(generation)
        if from_step is None:
            if self._context_generation == generation:
                return
            self._context_generation = generation
            self._revision += 1
            return
        if not _is_integer(from_step) or int(from_step) < 1:
            raise ValueError("from_step must be a positive integer")
        from_step = int(from_step)
        for row in self._records:
            if int(row["step"]) >= from_step and (not bool(row["pending"])):
                row["context_generation"] = generation
        self._context_generation = generation
        self._revision += 1

    def mark_reset(self, episode_id: int | None = None) -> int:
        if episode_id is None:
            episode_id = self._episode_id + 1
        if not _is_integer(episode_id) or int(episode_id) < 0:
            raise ValueError("episode_id must be a non-negative integer")
        episode_id = int(episode_id)
        if episode_id <= self._episode_id:
            raise ValueError("episode_id must increase at reset")
        self._episode_id = episode_id
        self._reset_boundary_pending = True
        self._revision += 1
        return episode_id

    def append(
        self,
        *,
        step: int,
        transition: Sequence[Any],
        feature: Any = None,
        episode_id: int | None = None,
        context_generation: int | None = None,
        pending: bool = False,
        terminal: bool = False,
        copy_payload: bool = True,
        return_copy: bool = True,
    ) -> dict[str, Any]:
        if not _is_integer(step) or int(step) < 1:
            raise ValueError("step must be a positive integer")
        step = int(step)
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("transition steps must be strictly increasing")
        if not isinstance(transition, (tuple, list)) or len(transition) != 6:
            raise ValueError("transition must contain six fields")
        if episode_id is not None:
            if not _is_integer(episode_id) or int(episode_id) < 0:
                raise ValueError("episode_id must be non-negative")
            if int(episode_id) != self._episode_id:
                raise ValueError("transition episode does not match stream")
        if context_generation is None:
            context_generation = self._context_generation
        if not _is_integer(context_generation) or int(context_generation) < 0:
            raise ValueError("context_generation must be non-negative")
        record = {
            "schema": SCHEMA,
            "version": VERSION,
            "valid_transition_id": int(self._next_transition_id),
            "step": step,
            "episode_id": int(self._episode_id),
            "context_generation": int(context_generation),
            "transition": (
                _clone_transition(transition) if copy_payload else tuple(transition)
            ),
            "feature": (
                _clone(feature)
                if copy_payload
                else (
                    feature.detach()
                    if callable(getattr(feature, "detach", None))
                    else feature
                )
            ),
            "terminal": bool(terminal),
            "reset_boundary": bool(self._reset_boundary_pending),
            "pending": bool(pending),
        }
        self._records.append(record)
        self._next_transition_id += 1
        self._last_step = step
        self._reset_boundary_pending = False
        if bool(pending) and self._pending_from_step is None:
            self._pending_from_step = step
        self._revision += 1
        return self._copy_record(record) if return_copy else record

    def mark_pending_from(
        self, proposal_step: int, *, generation: int | None = None
    ) -> int:
        if not _is_integer(proposal_step) or int(proposal_step) < 1:
            raise ValueError("proposal_step must be positive")
        proposal_step = int(proposal_step)
        if generation is not None and (
            not _is_integer(generation) or int(generation) < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        changed = 0
        for row in self._records:
            if (
                int(row["step"]) >= proposal_step
                and (
                    generation is None
                    or int(row["context_generation"]) == int(generation)
                )
                and (not row["pending"])
            ):
                row["pending"] = True
                changed += 1
        if changed or self._pending_from_step is None:
            self._pending_from_step = proposal_step
        if changed:
            self._revision += 1
        return changed

    def promote_pending(self, generation: int) -> int:
        if not _is_integer(generation) or int(generation) < 0:
            raise ValueError("generation must be non-negative")
        generation = int(generation)
        changed = 0
        for row in self._records:
            if row["pending"]:
                row["pending"] = False
                row["context_generation"] = generation
                changed += 1
        self._context_generation = generation
        self._pending_from_step = None
        if changed:
            self._revision += 1
        return changed

    def discard_pending(self) -> int:
        before = len(self._records)
        if before == 0:
            self._pending_from_step = None
            return 0
        self._records = deque(
            (row for row in self._records if not bool(row["pending"])),
            maxlen=self.capacity,
        )
        self._last_step = None if not self._records else int(self._records[-1]["step"])
        if self._records:
            self._reset_boundary_pending = False
        self._pending_from_step = None
        if before != len(self._records):
            self._revision += 1
        return int(before - len(self._records))

    def readiness(self, required_rows: int) -> bool:
        if not _is_integer(required_rows) or int(required_rows) < 1:
            raise ValueError("required_rows must be positive")
        return bool(self.valid_transition_count >= int(required_rows))

    @staticmethod
    def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: _clone_transition(value) if key == "transition" else _clone(value)
            for (key, value) in record.items()
        }

    def records(
        self,
        *,
        count: int | None = None,
        before_step: int | None = None,
        after_step: int | None = None,
        generation: int | None = None,
        include_pending: bool = False,
        copy_records: bool = True,
    ) -> list[dict[str, Any]]:
        if count is not None:
            if not _is_integer(count) or int(count) < 0:
                raise ValueError("count must be a non-negative integer")
            if int(count) == 0:
                return []
        if count is not None and before_step is None and (after_step is None):
            rows = []
            for row in reversed(self._records):
                if generation is not None and int(row["context_generation"]) != int(
                    generation
                ):
                    continue
                if not include_pending and bool(row["pending"]):
                    continue
                rows.append(self._copy_record(row) if copy_records else row)
                if len(rows) >= int(count):
                    break
            rows.reverse()
            return rows
        rows = []
        for row in self._records:
            step = int(row["step"])
            if before_step is not None and step >= int(before_step):
                continue
            if after_step is not None and step <= int(after_step):
                continue
            if generation is not None and int(row["context_generation"]) != int(
                generation
            ):
                continue
            if not include_pending and bool(row["pending"]):
                continue
            rows.append(self._copy_record(row) if copy_records else row)
        if count is not None:
            rows = rows[-int(count) :]
        return rows

    def clean_reference(
        self,
        *,
        proposal_step: int,
        count: int = 256,
        excluded_tail_steps: int = 64,
        generation: int | None = None,
    ) -> list[dict[str, Any]]:
        if not _is_integer(proposal_step) or int(proposal_step) < 1:
            raise ValueError("proposal_step must be positive")
        if not _is_integer(count) or int(count) < 1:
            raise ValueError("count must be positive")
        if not _is_integer(excluded_tail_steps) or int(excluded_tail_steps) < 0:
            raise ValueError("excluded_tail_steps must be non-negative")
        cutoff = int(proposal_step) - int(excluded_tail_steps)
        return self.records(
            count=int(count),
            before_step=cutoff + 1,
            generation=generation,
            include_pending=False,
        )

    def chunks(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
        *,
        copy_records: bool = True,
    ) -> list[list[dict[str, Any]]]:
        source = (
            self.records(include_pending=False, copy_records=copy_records)
            if rows is None
            else (
                list(rows)
                if not copy_records
                else [self._copy_record(row) for row in rows]
            )
        )
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        previous_episode = None
        previous_step = None
        for row in source:
            episode = int(row["episode_id"])
            step = int(row["step"])
            if current and (
                episode != previous_episode
                or previous_step is None
                or step != previous_step + 1
            ):
                groups.append(current)
                current = []
            current.append(row)
            previous_episode = episode
            previous_step = step
        if current:
            groups.append(current)
        return groups

    def feature_chunks(
        self,
        *,
        count: int | None = None,
        generation: int | None = None,
        include_pending: bool = False,
        copy_records: bool = True,
    ) -> list[list[dict[str, Any]]]:
        return self.chunks(
            self.records(
                count=count,
                generation=generation,
                include_pending=include_pending,
                copy_records=copy_records,
            ),
            copy_records=copy_records,
        )

    def panel_partition(
        self,
        *,
        proposal_step: int,
        panel_sizes: Sequence[int] = (32, 64, 64),
        generation: int | None = None,
        include_pending: bool = True,
    ) -> dict[str, list[dict[str, Any]]] | None:
        if not _is_integer(proposal_step) or int(proposal_step) < 1:
            raise ValueError("proposal_step must be positive")
        sizes = tuple((int(size) for size in panel_sizes))
        if not sizes or any((size < 1 for size in sizes)):
            raise ValueError("panel_sizes must contain positive integers")
        rows = self.records(
            after_step=int(proposal_step),
            generation=generation,
            include_pending=bool(include_pending),
        )
        required = sum(sizes)
        if len(rows) < required:
            return None
        rows = rows[:required]
        result: dict[str, list[dict[str, Any]]] = {}
        offset = 0
        names = ("selection", "validation_a", "validation_b")
        for index, size in enumerate(sizes):
            name = names[index] if index < len(names) else f"panel_{index}"
            result[name] = rows[offset : offset + size]
            offset += size
        for name, panel in tuple(result.items()):
            result[f"{name}_chunks"] = self.chunks(panel)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "capacity": int(self.capacity),
            "episode_id": int(self._episode_id),
            "context_generation": int(self._context_generation),
            "next_transition_id": int(self._next_transition_id),
            "last_step": self._last_step,
            "reset_boundary_pending": bool(self._reset_boundary_pending),
            "pending_from_step": self._pending_from_step,
            "records": [self._copy_record(row) for row in self._records],
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("invalid ContextEvidenceStream state")
        if state.get("schema") != SCHEMA or state.get("version") != VERSION:
            raise ValueError("ContextEvidenceStream schema/version mismatch")
        capacity = state.get("capacity")
        if not _is_integer(capacity) or int(capacity) < 1:
            raise ValueError("invalid ContextEvidenceStream capacity")
        records = state.get("records")
        if not isinstance(records, (list, tuple)) or len(records) > int(capacity):
            raise ValueError("invalid ContextEvidenceStream records")
        rebuilt = ContextEvidenceStream(
            int(capacity), episode_id=0, context_generation=0
        )
        previous_id = None
        for raw in records:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid ContextEvidenceStream record")
            required = {
                "schema",
                "version",
                "valid_transition_id",
                "step",
                "episode_id",
                "context_generation",
                "transition",
                "feature",
                "terminal",
                "reset_boundary",
                "pending",
            }
            if set(raw) != required:
                raise ValueError("ContextEvidenceStream record keys changed")
            raw_id = raw["valid_transition_id"]
            if (
                not _is_integer(raw_id)
                or (previous_id is not None and int(raw_id) <= previous_id)
                or (not _is_integer(raw["episode_id"]))
                or (int(raw["episode_id"]) < rebuilt.episode_id)
            ):
                raise ValueError("invalid ContextEvidenceStream record order")
            if int(raw["episode_id"]) > rebuilt.episode_id:
                rebuilt.mark_reset(int(raw["episode_id"]))
            record = rebuilt.append(
                step=int(raw["step"]),
                transition=raw["transition"],
                feature=raw["feature"],
                context_generation=int(raw["context_generation"]),
                pending=bool(raw["pending"]),
                terminal=bool(raw["terminal"]),
            )
            rebuilt._records[-1]["valid_transition_id"] = int(raw_id)
            record["reset_boundary"] = bool(raw["reset_boundary"])
            rebuilt._records[-1]["reset_boundary"] = bool(raw["reset_boundary"])
            previous_id = int(raw_id)
        episode_id = state.get("episode_id")
        generation = state.get("context_generation")
        if (
            not _is_integer(episode_id)
            or int(episode_id) < rebuilt.episode_id
            or (not _is_integer(generation))
            or (int(generation) < 0)
        ):
            raise ValueError("invalid ContextEvidenceStream state coordinates")
        if int(episode_id) > rebuilt.episode_id:
            rebuilt.mark_reset(int(episode_id))
        rebuilt._context_generation = int(generation)
        next_id = state.get("next_transition_id")
        if (
            not _is_integer(next_id)
            or int(next_id) < 0
            or (previous_id is not None and int(next_id) <= previous_id)
            or (previous_id is None and int(next_id) < 0)
        ):
            raise ValueError("invalid ContextEvidenceStream transition id")
        rebuilt._next_transition_id = int(next_id)
        last_step = state.get("last_step")
        if last_step is not None and (
            not _is_integer(last_step)
            or (records and int(last_step) != int(records[-1]["step"]))
        ):
            raise ValueError("invalid ContextEvidenceStream last step")
        rebuilt._last_step = None if last_step is None else int(last_step)
        rebuilt._reset_boundary_pending = bool(
            state.get("reset_boundary_pending", False)
        )
        pending_from = state.get("pending_from_step")
        if pending_from is not None and (
            not _is_integer(pending_from) or int(pending_from) < 1
        ):
            raise ValueError("invalid ContextEvidenceStream pending marker")
        rebuilt._pending_from_step = pending_from
        self.__dict__.update(rebuilt.__dict__)
        self._revision = max(1, int(getattr(self, "_revision", 0)))


__all__ = ["ContextEvidenceStream", "SCHEMA", "VERSION"]
