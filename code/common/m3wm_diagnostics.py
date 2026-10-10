from __future__ import annotations
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import struct
from typing import Any, Dict, Iterable, Optional
import torch

DIAGNOSTIC_SCHEMA = "m3wm_transaction_diagnostic"
DIAGNOSTIC_SCHEMA_VERSION = 3
_QUARANTINE_RETRY_STEPS = 128
GUARD_COUNTER_NAMES = (
    "router_pending_blend",
    "proposal_edge_theta_reset",
    "retrigger_drop",
    "infer_statistics",
    "same_slot_theta",
    "continuous_prototype",
    "stable_step_observe",
    "stable_episode_observe",
    "online_theta_writeback",
    "episode_reset_theta",
)
COMMIT_COUNTER_NAMES = (
    "dismiss_prototype",
    "confirmed_recall_observe",
    "slot_create",
    "infer_statistics",
    "same_slot_theta",
    "continuous_prototype",
    "stable_step_observe",
    "stable_episode_observe",
    "online_theta_writeback",
    "episode_reset_theta",
)
_DUE_EVIDENCE_DEFAULTS = {
    "feature_ready": None,
    "capacity_available": None,
    "identity_action": None,
    "active_context_id_before": None,
    "context_count_before": None,
    "best_existing_context_id": None,
    "best_existing_log_posterior": None,
    "active_log_posterior": None,
    "new_log_posterior": None,
    "new_advantage": None,
    "recall_advantage": None,
    "regime_d_z": None,
    "regime_s_post": None,
    "authoritative_assignment_diagnostics": None,
    "feature_group_contributions": None,
    "reward_residual": None,
    "slot_digests_due_pre": None,
    "theta_before_activation": None,
}


def runtime_step_coordinates(runtime_step: int) -> Dict[str, int]:
    step = int(runtime_step)
    if step < 1:
        raise ValueError("runtime_step must be 1-based and positive")
    return {"runtime_step": step, "env_step": step - 1}


def _scheduled_due_coordinates(
    proposal: Dict[str, Any], confirmation_due_step: Optional[int] = None
) -> Dict[str, Any]:
    raw_step = (
        proposal.get("confirmation_due_step")
        if confirmation_due_step is None
        else confirmation_due_step
    )
    try:
        numeric = float(raw_step)
        step = int(raw_step)
        valid = math.isfinite(numeric) and numeric == float(step) and (step >= 1)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        return {
            "runtime_step": None,
            "env_step": None,
            "scheduled_confirmation_due_step": None,
            "coordinate_valid": False,
            "coordinate_reason": "invalid_confirmation_due_step",
        }
    return {
        **runtime_step_coordinates(step),
        "scheduled_confirmation_due_step": step,
        "coordinate_valid": True,
        "coordinate_reason": None,
    }


def _due_payload_defaults() -> Dict[str, Any]:
    return deepcopy(_DUE_EVIDENCE_DEFAULTS)


def _strict_record_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _validate_record_coordinates(
    payload: Dict[str, Any], *, phase: str, expected_runtime_step: Optional[int] = None
) -> int:
    runtime_step = _strict_record_integer(
        payload.get("runtime_step"), f"{phase}.runtime_step"
    )
    env_step = _strict_record_integer(payload.get("env_step"), f"{phase}.env_step")
    if runtime_step < 1 or env_step != runtime_step - 1:
        raise ValueError(f"{phase} has invalid runtime/env step coordinates")
    if expected_runtime_step is not None and runtime_step != int(expected_runtime_step):
        raise ValueError(
            f"{phase}.runtime_step must equal the scheduled confirmation step"
        )
    return runtime_step


def _record_confirmation_due_steps(
    record: Dict[str, Any],
) -> tuple[int, Optional[int], int]:
    original = int(record["proposal"]["confirmation_due_step"])
    quarantine = record.get("quarantine")
    retry = (
        None if quarantine is None else int(quarantine["retry_confirmation_due_step"])
    )
    checkpoints = record.get("sequential_identity_checkpoints", [])
    if retry is not None and checkpoints:
        raise RuntimeError(
            "quarantine and sequential identity checkpoints cannot coexist"
        )
    if checkpoints:
        current = int(
            checkpoints[-1]["sequential_identity_checkpoint"][
                "next_confirmation_due_step"
            ]
        )
    else:
        current = original if retry is None else retry
    return (original, retry, current)


def strict_json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    )


def canonical_json_sha256(value: Any) -> str:
    cloned = strict_json_clone(value)
    serialized = json.dumps(
        cloned,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def tensor_snapshot(value: Optional[torch.Tensor]) -> Dict[str, Any]:
    if value is None:
        return {
            "valid": False,
            "reason": "unavailable",
            "dtype": None,
            "shape": None,
            "values": None,
        }
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if tensor.numel() and (not bool(torch.isfinite(tensor).all())):
        return {
            "valid": False,
            "reason": "nonfinite_tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": None,
        }
    return {
        "valid": True,
        "reason": None,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "values": tensor.tolist(),
    }


def signed_reward_residual_summary(
    values: Iterable[Any], *, window: int = 128
) -> Dict[str, Any]:
    if window < 1:
        raise ValueError("window must be positive")
    selected = list(values)[-int(window) :]
    normalized = [finite_or_none(value) for value in selected]
    finite = [value for value in normalized if value is not None]
    n_total = len(normalized)
    n_finite = len(finite)
    if not finite:
        return {
            "valid": False,
            "reason": "no_finite_samples",
            "window": int(window),
            "n": n_finite,
            "n_total": n_total,
            "n_finite": n_finite,
            "signed_mean": None,
            "rms": None,
            "median": None,
            "mad": None,
        }

    def median(items: list[float]) -> float:
        ordered = sorted(items)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return float(0.5 * (ordered[middle - 1] + ordered[middle]))

    center = median(finite)
    valid = n_total == n_finite
    return {
        "valid": valid,
        "reason": None if valid else "nonfinite_sample_in_window",
        "window": int(window),
        "n": n_finite,
        "n_total": n_total,
        "n_finite": n_finite,
        "signed_mean": float(sum(finite) / n_finite),
        "rms": float(math.sqrt(sum((value * value for value in finite)) / n_finite)),
        "median": center,
        "mad": median([abs(value - center) for value in finite]),
    }


def _hash_bytes(hasher: "hashlib._Hash", value: bytes) -> None:
    hasher.update(struct.pack(">Q", len(value)))
    hasher.update(value)


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    return bytes(value.view(torch.uint8).reshape(-1).tolist())


def canonical_slot_digest(
    slot_id: int, count: int, tensors: Iterable[tuple[str, torch.Tensor]]
) -> str:
    hasher = hashlib.sha256()
    _hash_bytes(hasher, b"M3WM_CANONICAL_SLOT_V1")
    hasher.update(struct.pack(">q", int(slot_id)))
    hasher.update(struct.pack(">q", int(count)))
    for name, raw_tensor in tensors:
        tensor = torch.as_tensor(raw_tensor).detach().cpu().contiguous()
        _hash_bytes(hasher, str(name).encode("utf-8"))
        _hash_bytes(hasher, str(tensor.dtype).encode("ascii"))
        hasher.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            hasher.update(struct.pack(">q", int(dimension)))
        _hash_bytes(hasher, _tensor_raw_bytes(tensor))
    return hasher.hexdigest()


def canonical_context_slot_digests(memory: Any) -> Dict[str, Any]:
    slots = [] if memory is None else list(getattr(memory, "slots", ()))
    records = []
    memory_hasher = hashlib.sha256()
    _hash_bytes(memory_hasher, b"M3WM_CONTEXT_MEMORY_DIGEST_V1")
    memory_hasher.update(struct.pack(">Q", len(slots)))
    for slot_id, slot in enumerate(slots):
        count = int(slot.count)
        prototype = canonical_slot_digest(
            slot_id, count, (("mean", slot.mean), ("m2", slot.m2))
        )
        theta = canonical_slot_digest(slot_id, count, (("theta", slot.theta),))
        full = canonical_slot_digest(
            slot_id,
            count,
            (("mean", slot.mean), ("m2", slot.m2), ("theta", slot.theta)),
        )
        memory_hasher.update(bytes.fromhex(full))
        records.append(
            {
                "context_id": slot_id,
                "count": count,
                "prototype_sha256": prototype,
                "theta_sha256": theta,
                "full_slot_sha256": full,
            }
        )
    return {
        "schema": "m3wm_context_slot_digests",
        "version": 1,
        "num_contexts": len(slots),
        "memory_sha256": memory_hasher.hexdigest(),
        "slots": records,
    }


class M3wmDiagnosticRecorder:

    def __init__(self, enabled: bool = False, reward_window: int = 128):
        if reward_window < 1:
            raise ValueError("reward_window must be positive")
        self.enabled = bool(enabled)
        self.reward_window = int(reward_window)
        self._reward_residuals = deque(maxlen=self.reward_window)
        self._queue: list[Dict[str, Any]] = []
        self._active: Optional[Dict[str, Any]] = None
        self._guard_hits = {name: 0 for name in GUARD_COUNTER_NAMES}
        self._commits = {name: 0 for name in COMMIT_COUNTER_NAMES}

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._active is not None)

    def observe_reward_residual(self, value: Any) -> None:
        if not self.enabled:
            return
        numeric = finite_or_none(value)
        self._reward_residuals.append(numeric)

    def reward_residual_summary(self) -> Dict[str, Any]:
        return signed_reward_residual_summary(
            self._reward_residuals, window=self.reward_window
        )

    def increment_guard(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        if name not in self._guard_hits:
            raise KeyError(f"Unknown diagnostic guard counter: {name}")
        if int(amount) < 0:
            raise ValueError("counter increment must be non-negative")
        self._guard_hits[name] += int(amount)

    def increment_commit(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        if name not in self._commits:
            raise KeyError(f"Unknown diagnostic commit counter: {name}")
        if int(amount) < 0:
            raise ValueError("counter increment must be non-negative")
        self._commits[name] += int(amount)

    def begin(self, proposal: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._active is not None:
            raise RuntimeError("Diagnostic transaction already active")
        step = int(proposal["runtime_step"])
        source = str(proposal["source"])
        self._active = {
            "schema": DIAGNOSTIC_SCHEMA,
            "version": DIAGNOSTIC_SCHEMA_VERSION,
            "transaction_id": f"tx-{step:09d}",
            "source": source,
            "proposal": strict_json_clone(proposal),
            "quarantine": None,
            "sequential_identity_checkpoints": [],
            "due": None,
            "final": None,
            "outcome": None,
            "reason": None,
            "guard_counters": {
                "baseline": deepcopy(self._guard_hits),
                "final": None,
                "delta": None,
            },
            "commit_counters": {
                "baseline": deepcopy(self._commits),
                "final": None,
                "delta": None,
            },
            "retriggers": {
                "drop_count": 0,
                "first_runtime_step": None,
                "last_runtime_step": None,
            },
            "restore_reconciliations": [],
        }

    def capture_proposal_runtime(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._active is None:
            raise RuntimeError("No active diagnostic transaction")
        self._active["proposal"]["runtime"] = strict_json_clone(payload)

    def note_retrigger(self, runtime_step: int) -> None:
        if not self.enabled:
            return
        self.increment_guard("retrigger_drop")
        if self._active is None:
            raise RuntimeError("Retrigger without active diagnostic transaction")
        retriggers = self._active["retriggers"]
        retriggers["drop_count"] += 1
        step = int(runtime_step)
        if retriggers["first_runtime_step"] is None:
            retriggers["first_runtime_step"] = step
        retriggers["last_runtime_step"] = step

    def record_quarantine(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._active is None:
            raise RuntimeError("Quarantine record without active transaction")
        if self._active["due"] is not None:
            raise RuntimeError("Quarantine recorded after authoritative due")
        if self._active["quarantine"] is not None:
            raise RuntimeError("Diagnostic transaction quarantined twice")
        if not isinstance(payload, dict):
            raise TypeError("Quarantine payload must be a dict")
        normalized = strict_json_clone(payload)
        original_due, _retry_due, _current_due = _record_confirmation_due_steps(
            self._active
        )
        _validate_record_coordinates(
            normalized, phase="quarantine", expected_runtime_step=original_due
        )
        retry_due = _strict_record_integer(
            normalized.get("retry_confirmation_due_step"),
            "quarantine.retry_confirmation_due_step",
        )
        if retry_due != original_due + _QUARANTINE_RETRY_STEPS:
            raise ValueError(
                "quarantine retry must be exactly 128 steps after the original confirmation due step"
            )
        normalized.update(
            {
                "observed": True,
                "valid": True,
                "reason": normalized.get("reason"),
                "stage": "quarantine",
                "retry_count": 1,
                "scheduled_confirmation_due_step": original_due,
                "original_confirmation_due_step": original_due,
                "retry_confirmation_due_step": retry_due,
                "current_confirmation_due_step": retry_due,
                "coordinate_valid": True,
                "coordinate_reason": None,
            }
        )
        self._active["quarantine"] = strict_json_clone(normalized)

    def record_due(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._active is None:
            raise RuntimeError("Due record without active transaction")
        if self._active["due"] is not None:
            raise RuntimeError("Diagnostic transaction due recorded twice")
        if not isinstance(payload, dict):
            raise TypeError("Due payload must be a dict")
        payload_snapshot = strict_json_clone(payload)
        original_due, retry_due, current_due = _record_confirmation_due_steps(
            self._active
        )
        sequential_checkpoints = self._active["sequential_identity_checkpoints"]
        if retry_due is not None or sequential_checkpoints:
            _validate_record_coordinates(
                payload_snapshot, phase="due", expected_runtime_step=current_due
            )
        normalized = _due_payload_defaults()
        normalized.update(payload_snapshot)
        normalized.update(
            {
                "observed": True,
                "valid": True,
                "reason": None,
                "stage": (
                    "retry_due"
                    if retry_due is not None
                    else (
                        "sequential_final_due"
                        if sequential_checkpoints
                        else "initial_due"
                    )
                ),
                "scheduled_confirmation_due_step": current_due,
                "original_confirmation_due_step": original_due,
                "retry_confirmation_due_step": retry_due,
                "current_confirmation_due_step": current_due,
                "coordinate_valid": True,
                "coordinate_reason": None,
            }
        )
        self._active["due"] = strict_json_clone(normalized)

    def record_sequential_identity_checkpoint(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._active is None:
            raise RuntimeError(
                "Sequential checkpoint without active diagnostic transaction"
            )
        if self._active["due"] is not None:
            raise RuntimeError("Sequential checkpoint recorded after authoritative due")
        if self._active["quarantine"] is not None:
            raise RuntimeError("Sequential checkpoint cannot coexist with quarantine")
        if not isinstance(payload, dict):
            raise TypeError("Sequential checkpoint payload must be a dict")
        normalized = strict_json_clone(payload)
        metadata = normalized.get("sequential_identity_checkpoint")
        if not isinstance(metadata, dict) or set(metadata) != {
            "proposal_step",
            "checkpoint_index",
            "horizon_steps",
            "interval_steps",
            "max_horizon_steps",
            "next_confirmation_due_step",
        }:
            raise ValueError("Invalid sequential identity checkpoint metadata")
        proposal_step = _strict_record_integer(
            metadata["proposal_step"], "sequential_checkpoint.proposal_step"
        )
        checkpoint_index = _strict_record_integer(
            metadata["checkpoint_index"], "sequential_checkpoint.checkpoint_index"
        )
        horizon = _strict_record_integer(
            metadata["horizon_steps"], "sequential_checkpoint.horizon_steps"
        )
        interval = _strict_record_integer(
            metadata["interval_steps"], "sequential_checkpoint.interval_steps"
        )
        max_horizon = _strict_record_integer(
            metadata["max_horizon_steps"], "sequential_checkpoint.max_horizon_steps"
        )
        next_due = _strict_record_integer(
            metadata["next_confirmation_due_step"],
            "sequential_checkpoint.next_confirmation_due_step",
        )
        expected_proposal = int(self._active["proposal"]["runtime_step"])
        expected_index = len(self._active["sequential_identity_checkpoints"]) + 1
        expected_next_due = proposal_step + min(horizon + interval, max_horizon)
        if (
            proposal_step != expected_proposal
            or checkpoint_index != expected_index
            or interval < 1
            or (max_horizon < interval)
            or (horizon != checkpoint_index * interval)
            or (horizon >= max_horizon)
            or (next_due != expected_next_due)
        ):
            raise ValueError(
                "Sequential identity checkpoint violates fixed progression"
            )
        _validate_record_coordinates(
            normalized,
            phase="sequential_checkpoint",
            expected_runtime_step=proposal_step + horizon,
        )
        normalized.update(
            {
                "observed": True,
                "valid": True,
                "reason": None,
                "stage": "sequential_identity_checkpoint",
                "coordinate_valid": True,
                "coordinate_reason": None,
            }
        )
        self._active["sequential_identity_checkpoints"].append(
            strict_json_clone(normalized)
        )

    def note_restore_reconciliation(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._active is None:
            raise RuntimeError("Restore reconciliation without active transaction")
        self._active["restore_reconciliations"].append(strict_json_clone(payload))

    def finalize(self, outcome: str, reason: str, final: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._active is None:
            raise RuntimeError("Finalize without active diagnostic transaction")
        if self._active["due"] is None:
            raise RuntimeError("Diagnostic transaction finalized without due")
        record = deepcopy(self._active)
        record["outcome"] = str(outcome)
        record["reason"] = str(reason)
        record["final"] = strict_json_clone(final)
        proposal_step = int(record["proposal"]["runtime_step"])
        final_step = int(record["final"]["runtime_step"])
        original_due, retry_due, current_due = _record_confirmation_due_steps(record)
        record["original_confirmation_due_step"] = original_due
        record["retry_confirmation_due_step"] = retry_due
        record["current_confirmation_due_step"] = current_due
        record["original_configured_delay"] = original_due - proposal_step
        record["configured_delay"] = current_due - proposal_step
        record["observed_delay"] = final_step - proposal_step
        record["guard_counters"]["final"] = deepcopy(self._guard_hits)
        record["guard_counters"]["delta"] = {
            name: self._guard_hits[name] - record["guard_counters"]["baseline"][name]
            for name in GUARD_COUNTER_NAMES
        }
        record["commit_counters"]["final"] = deepcopy(self._commits)
        record["commit_counters"]["delta"] = {
            name: self._commits[name] - record["commit_counters"]["baseline"][name]
            for name in COMMIT_COUNTER_NAMES
        }
        closed = strict_json_clone(record)
        self._queue.append(closed)
        self._active = None

    def orphan(
        self, runtime_step: int, reason: str, final: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.enabled or self._active is None:
            return
        original_due, retry_due, current_due = _record_confirmation_due_steps(
            self._active
        )
        closure_runtime_step = _strict_record_integer(
            runtime_step, "orphan.runtime_step"
        )
        if reason == "run_end_right_censored" and (
            closure_runtime_step >= current_due
            or (retry_due is not None and closure_runtime_step < original_due)
        ):
            raise RuntimeError(
                "Run-end right-censor must follow observed quarantine and precede the authoritative due step"
            )
        pending_due = None
        if self._active["due"] is None:
            due = _due_payload_defaults()
            due.update(
                _scheduled_due_coordinates(self._active["proposal"], current_due)
            )
            due.update(
                {
                    "observed": False,
                    "valid": False,
                    "reason": (
                        "run_end_right_censored"
                        if reason == "run_end_right_censored"
                        else "not_reached"
                    ),
                    "closure_reason": str(reason),
                    "closure_stage": (
                        "quarantine"
                        if retry_due is not None
                        else "pending_confirmation"
                    ),
                    "stage": "retry_due" if retry_due is not None else "initial_due",
                    "original_confirmation_due_step": original_due,
                    "retry_confirmation_due_step": retry_due,
                    "current_confirmation_due_step": current_due,
                }
            )
            pending_due = strict_json_clone(due)
        payload = dict(runtime_step_coordinates(closure_runtime_step))
        payload.update(
            {
                "active_context_id_after": None,
                "context_count_after": None,
                "slot_digests_after": None,
                "theta": None,
                "reward_residual": self.reward_residual_summary(),
            }
        )
        if final:
            payload.update(final)
        payload = strict_json_clone(payload)
        int(payload["runtime_step"])
        if pending_due is not None:
            self._active["due"] = pending_due
        self.finalize("orphan_cancel", reason, payload)

    def drain(self) -> list[Dict[str, Any]]:
        if not self.enabled:
            return []
        records = strict_json_clone(self._queue)
        self._queue.clear()
        return records

    def peek_active(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._active is None:
            return None
        return strict_json_clone(self._active)

    def active_identity(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._active is None:
            return None
        proposal = self._active["proposal"]
        original_due, retry_due, current_due = _record_confirmation_due_steps(
            self._active
        )
        return {
            "source": str(self._active["source"]),
            "proposal_step": int(proposal["runtime_step"]),
            "confirmation_due_step": current_due,
            "original_confirmation_due_step": original_due,
            "retry_confirmation_due_step": retry_due,
            "current_confirmation_due_step": current_due,
            "stage": "quarantine" if retry_due is not None else "pending_due",
        }
