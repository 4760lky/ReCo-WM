from __future__ import annotations
import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import torch


def _clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone().cpu()
    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone().cpu()
        except (AttributeError, TypeError):
            return clone()
    return copy.deepcopy(value)


def _flat_cpu_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value).detach().clone().cpu().flatten()


def _same_observation(left: Any, right: Any) -> bool:
    lhs = _flat_cpu_tensor(left)
    rhs = _flat_cpu_tensor(right)
    return bool(lhs.shape == rhs.shape and torch.equal(lhs, rhs))


def _origin_ids_sha256(values: Sequence[int]) -> str:
    body = "\n".join((str(int(value)) for value in values)).encode("ascii")
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class ControlTransferRow:
    transition_id: int
    episode_id: int
    behavior_context_id: int
    prev_obs: torch.Tensor
    next_obs: torch.Tensor
    command_action: torch.Tensor
    transition_td: Any
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ControlTransferPlan:
    transaction_id: str
    source_context_id: int
    nomination_transition_id: int
    source_revision: int
    init_capsule_hash: str
    commit_transition_id: int
    segments: tuple[tuple[ControlTransferRow, ...], ...]
    tail_connects_commit: bool
    origin_transition_ids: tuple[int, ...]
    origin_ids_sha256: str

    @property
    def prefill_count(self) -> int:
        return len(self.origin_transition_ids)

    def audit_segments(self) -> list[dict[str, Any]]:
        return [
            {
                "segment_index": index,
                "episode_id": int(rows[0].episode_id),
                "first_transition_id": int(rows[0].transition_id),
                "last_transition_id": int(rows[-1].transition_id),
                "transition_count": len(rows),
                "terminated": bool(rows[-1].terminated),
                "truncated": bool(rows[-1].truncated),
            }
            for (index, rows) in enumerate(self.segments)
        ]


@dataclass
class PendingControlTransfer:
    transaction_id: str
    source_context_id: int
    nomination_transition_id: int
    source_revision: int
    init_capsule_hash: str
    max_rows: int = 4096
    rows: list[ControlTransferRow] = field(default_factory=list)
    skipped_rows: list[dict[str, Any]] = field(default_factory=list)
    schema = "m3wm-pending-control-transfer-v1"

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ValueError("control transfer requires a transaction id")
        if self.source_context_id < 0:
            raise ValueError("control transfer source Context must be non-negative")
        if self.nomination_transition_id < 0:
            raise ValueError("control transfer nomination id must be non-negative")
        if self.source_revision < 0:
            raise ValueError("control transfer source revision must be non-negative")
        if not self.init_capsule_hash:
            raise ValueError("control transfer requires an InitCapsule hash")
        if self.max_rows <= 0:
            raise ValueError("control transfer max_rows must be positive")

    def matches(
        self,
        *,
        transaction_id: str,
        source_context_id: int,
        nomination_transition_id: int,
        source_revision: int,
        init_capsule_hash: str,
    ) -> bool:
        return bool(
            self.transaction_id == str(transaction_id)
            and self.source_context_id == int(source_context_id)
            and (self.nomination_transition_id == int(nomination_transition_id))
            and (self.source_revision == int(source_revision))
            and (self.init_capsule_hash == str(init_capsule_hash))
        )

    def record(
        self,
        transition: Mapping[str, Any],
        control_payload: Mapping[str, Any] | None,
        *,
        behavior_context_id: int,
    ) -> bool:
        transition_id = int(transition["transition_id"])
        if transition_id < self.nomination_transition_id:
            raise RuntimeError("control transfer row predates its nomination")
        if behavior_context_id != self.source_context_id:
            self.skipped_rows.append(
                {
                    "transition_id": transition_id,
                    "behavior_context_id": int(behavior_context_id),
                    "reason": "behavior_not_committed_source",
                }
            )
            return False
        if control_payload is None:
            raise RuntimeError("source-owned control transfer row has no payload")
        required = {"transition_td", "prev_obs", "episode_id"}
        missing = sorted(required.difference(control_payload))
        if missing:
            raise RuntimeError(f"control transfer payload missing fields: {missing}")
        if len(self.rows) >= self.max_rows:
            raise RuntimeError("control transfer exceeded its bounded row budget")
        if self.rows and transition_id <= self.rows[-1].transition_id:
            raise RuntimeError("control transfer transition ids are not increasing")
        if any((row.transition_id == transition_id for row in self.rows)):
            raise RuntimeError("control transfer contains a duplicate transition")
        transition_td = _clone(control_payload["transition_td"])
        td_action = _flat_cpu_tensor(transition_td.get("action"))
        command_action = _flat_cpu_tensor(transition["action"])
        if td_action.shape != command_action.shape or not torch.equal(
            td_action, command_action
        ):
            raise RuntimeError(
                "control transfer payload does not contain the executed command action"
            )
        row = ControlTransferRow(
            transition_id=transition_id,
            episode_id=int(control_payload["episode_id"]),
            behavior_context_id=int(behavior_context_id),
            prev_obs=_flat_cpu_tensor(control_payload["prev_obs"]),
            next_obs=_flat_cpu_tensor(transition["next_obs"]),
            command_action=command_action,
            transition_td=transition_td,
            terminated=bool(transition.get("terminated", False)),
            truncated=bool(transition.get("truncated", False)),
        )
        self.rows.append(row)
        return True

    def build_plan(
        self, *, commit_transition_id: int, commit_prev_obs: Any
    ) -> ControlTransferPlan:
        commit_id = int(commit_transition_id)
        if commit_id <= self.nomination_transition_id:
            raise RuntimeError("NEW commit does not follow its nomination")
        if not self.rows:
            raise RuntimeError("confirmed NEW has no legal source control rows")
        if any((row.transition_id >= commit_id for row in self.rows)):
            raise RuntimeError("NEW commit row was duplicated into the prefill ledger")
        segments: list[list[ControlTransferRow]] = []
        for row in self.rows:
            start_new = not segments
            if segments:
                previous = segments[-1][-1]
                start_new = bool(
                    row.episode_id != previous.episode_id
                    or row.transition_id != previous.transition_id + 1
                    or previous.terminated
                    or previous.truncated
                )
                if not start_new and (
                    not _same_observation(previous.next_obs, row.prev_obs)
                ):
                    raise RuntimeError(
                        "contiguous control transfer rows do not share an observation boundary"
                    )
            if start_new:
                segments.append([row])
            else:
                segments[-1].append(row)
        tail = segments[-1][-1]
        tail_connects = bool(
            not tail.terminated
            and (not tail.truncated)
            and (tail.transition_id == commit_id - 1)
            and _same_observation(tail.next_obs, commit_prev_obs)
        )
        if not tail_connects and (not (tail.terminated or tail.truncated)):
            raise RuntimeError(
                "prefill tail does not connect to the NEW commit observation"
            )
        origin_ids = tuple((row.transition_id for row in self.rows))
        return ControlTransferPlan(
            transaction_id=self.transaction_id,
            source_context_id=self.source_context_id,
            nomination_transition_id=self.nomination_transition_id,
            source_revision=self.source_revision,
            init_capsule_hash=self.init_capsule_hash,
            commit_transition_id=commit_id,
            segments=tuple((tuple(rows) for rows in segments)),
            tail_connects_commit=tail_connects,
            origin_transition_ids=origin_ids,
            origin_ids_sha256=_origin_ids_sha256(origin_ids),
        )

    def audit_record(self, *, event: str, reason: str | None = None) -> dict[str, Any]:
        origin_ids = tuple((row.transition_id for row in self.rows))
        return {
            "schema": self.schema,
            "event": str(event),
            "reason": reason,
            "transaction_id": self.transaction_id,
            "source_context_id": self.source_context_id,
            "nomination_transition_id": self.nomination_transition_id,
            "source_revision": self.source_revision,
            "init_capsule_hash": self.init_capsule_hash,
            "recorded_count": len(origin_ids),
            "origin_ids_sha256": _origin_ids_sha256(origin_ids),
            "first_transition_id": origin_ids[0] if origin_ids else None,
            "last_transition_id": origin_ids[-1] if origin_ids else None,
            "skipped_rows": copy.deepcopy(self.skipped_rows),
        }
