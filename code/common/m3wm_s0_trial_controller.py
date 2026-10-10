from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TrialLease:
    trial_id: int
    source_context_id: int
    candidate_context_id: int
    start_step: int
    reserved_steps: int
    source_bundle_hash: str
    candidate_bundle_hash: str
    selector_state: Any
    predictor_versions: Any
    rows: list[dict[str, Any]] = field(default_factory=list)
    control_payloads: list[Any] = field(default_factory=list)
    identity_payloads: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def used_steps(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class TrialResolution:
    trial_id: int
    outcome: str
    source_context_id: int
    candidate_context_id: int
    start_step: int
    end_step: int
    used_steps: int
    assigned_context_id: int | None
    rows: tuple[dict[str, Any], ...]
    control_payloads: tuple[Any, ...]
    reason: str
    selector_state: Any = None
    checkpoint_count: int = 0
    trial_mean_llr: float | None = None
    evidence_cleared: bool = False
    trial_mean_reward: float | None = None


class TrialController:
    schema = "m3wm-s0-trial-controller-v2-evidence-checkpoints"

    def __init__(
        self,
        *,
        enabled: bool = True,
        check_interval: int = 32,
        max_trial_steps: int = 128,
        cooldown_steps: int = 1024,
        total_ratio: float = 0.02,
        absolute_limit: int = 4096,
    ) -> None:
        if int(check_interval) <= 0 or int(max_trial_steps) < int(check_interval):
            raise ValueError("invalid trial checkpoint or duration")
        if int(cooldown_steps) < 0 or int(absolute_limit) <= 0:
            raise ValueError("invalid trial cooldown or absolute limit")
        if not math.isfinite(float(total_ratio)) or not 0.0 < float(total_ratio) <= 1.0:
            raise ValueError("trial ratio must be in (0, 1]")
        self.enabled = bool(enabled)
        self.check_interval = int(check_interval)
        self.max_trial_steps = int(max_trial_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.total_ratio = float(total_ratio)
        self.absolute_limit = int(absolute_limit)
        self.trial_steps_total = 0
        self.paused_control_updates = 0
        self.paused_theta_updates = 0
        self.paused_predictor_updates = 0
        self.discarded_training_rows = 0
        self._next_trial_id = 1
        self._last_end_step: int | None = None
        self._lease: TrialLease | None = None
        self._working_bundle: Any = None
        self._records: list[dict[str, Any]] = []
        self._latest_checkpoint: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self._lease is not None

    @property
    def behavior_context_id(self) -> int | None:
        return None if self._lease is None else self._lease.candidate_context_id

    @property
    def committed_context_id(self) -> int | None:
        return None if self._lease is None else self._lease.source_context_id

    @property
    def working_bundle(self) -> Any:
        return self._working_bundle

    def allowed_total(self, env_steps_seen: int) -> int:
        return min(
            self.absolute_limit,
            int(math.floor(self.total_ratio * max(0, int(env_steps_seen)))),
        )

    def remaining_budget(self, env_steps_seen: int) -> int:
        return max(0, self.allowed_total(env_steps_seen) - self.trial_steps_total)

    def can_start(self, *, env_steps_seen: int) -> tuple[bool, str, int]:
        step = int(env_steps_seen)
        if not self.enabled:
            return (False, "trial_disabled", 0)
        if self.active:
            return (False, "trial_already_active", 0)
        if (
            self._last_end_step is not None
            and step - self._last_end_step < self.cooldown_steps
        ):
            return (False, "trial_cooldown", 0)
        reserve = min(self.max_trial_steps, self.remaining_budget(step))
        if reserve < self.check_interval:
            return (False, "trial_budget_below_checkpoint", reserve)
        return (True, "trial_eligible", reserve)

    def record_rejection(
        self,
        *,
        env_steps_seen: int,
        source_context_id: int,
        candidate_context_id: int,
        reason: str,
    ) -> None:
        self._records.append(
            {
                "event": "TRIAL_REJECTED",
                "step": int(env_steps_seen),
                "source_context_id": int(source_context_id),
                "candidate_context_id": int(candidate_context_id),
                "reason": str(reason),
                "remaining_budget": int(self.remaining_budget(env_steps_seen)),
            }
        )

    def start(
        self,
        *,
        env_steps_seen: int,
        source_context_id: int,
        candidate_context_id: int,
        source_bundle_hash: str,
        candidate_bundle_hash: str,
        selector_state: Any,
        predictor_versions: Any,
        working_bundle: Any,
    ) -> TrialLease | None:
        eligible, reason, reserve = self.can_start(env_steps_seen=env_steps_seen)
        source = int(source_context_id)
        candidate = int(candidate_context_id)
        if candidate == source:
            reason = "candidate_is_committed_source"
            eligible = False
        if working_bundle is None:
            reason = "candidate_bundle_unavailable"
            eligible = False
        if not source_bundle_hash or not candidate_bundle_hash:
            reason = "bundle_hash_missing"
            eligible = False
        if not eligible:
            self._records.append(
                {
                    "event": "TRIAL_REJECTED",
                    "step": int(env_steps_seen),
                    "source_context_id": source,
                    "candidate_context_id": candidate,
                    "reason": reason,
                    "remaining_budget": int(self.remaining_budget(env_steps_seen)),
                }
            )
            return None
        lease = TrialLease(
            trial_id=self._next_trial_id,
            source_context_id=source,
            candidate_context_id=candidate,
            start_step=int(env_steps_seen) + 1,
            reserved_steps=int(reserve),
            source_bundle_hash=str(source_bundle_hash),
            candidate_bundle_hash=str(candidate_bundle_hash),
            selector_state=copy.deepcopy(selector_state),
            predictor_versions=copy.deepcopy(predictor_versions),
        )
        self._next_trial_id += 1
        self._lease = lease
        self._working_bundle = working_bundle
        self._records.append(
            {
                "event": "TRIAL_STARTED",
                "trial_id": lease.trial_id,
                "step": int(env_steps_seen),
                "behavior_start_step": lease.start_step,
                "source_context_id": source,
                "candidate_context_id": candidate,
                "reserved_steps": lease.reserved_steps,
                "source_bundle_hash": lease.source_bundle_hash,
                "candidate_bundle_hash": lease.candidate_bundle_hash,
            }
        )
        return lease

    def record_transition(
        self,
        row: Mapping[str, Any],
        *,
        control_payload: Any = None,
        identity_payload: Mapping[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        if self._lease is None:
            raise RuntimeError("no active trial lease")
        self._lease.rows.append(copy.deepcopy(dict(row)))
        self._lease.control_payloads.append(copy.deepcopy(control_payload))
        self._lease.identity_payloads.append(
            {} if identity_payload is None else copy.deepcopy(dict(identity_payload))
        )
        self.trial_steps_total += 1
        self.paused_control_updates += 1
        self.paused_theta_updates += 1
        self.paused_predictor_updates += 1
        used = self._lease.used_steps
        checkpoint_due = used % self.check_interval == 0
        if checkpoint_due:
            self._record_identity_checkpoint()
        if used >= self._lease.reserved_steps:
            return (True, "reserved_budget_exhausted")
        if used >= self.max_trial_steps:
            return (True, "trial_hard_limit")
        return (checkpoint_due, None)

    def _record_identity_checkpoint(self) -> None:
        if self._lease is None:
            raise RuntimeError("no active trial lease")
        payloads = self._lease.identity_payloads[-self.check_interval :]
        rows = self._lease.rows[-self.check_interval :]
        complete = len(payloads) == self.check_interval and all(payloads)
        mean_llr = None
        historical_scores_zero = False
        new_score_zero = False
        identity_matches = False
        if complete:
            source = int(self._lease.source_context_id)
            candidate = int(self._lease.candidate_context_id)
            identity_matches = all(
                (
                    int(item.get("source_context_id", -1)) == source
                    and int(item.get("candidate_context_id", -1)) == candidate
                    and (item.get("source_log_prob") is not None)
                    and (item.get("candidate_log_prob") is not None)
                    for item in payloads
                )
            )
            if identity_matches:
                mean_llr = sum(
                    (
                        float(item["candidate_log_prob"])
                        - float(item["source_log_prob"])
                        for item in payloads
                    )
                ) / len(payloads)
                last = payloads[-1]
                scores = {
                    int(key): float(value)
                    for (key, value) in dict(last.get("scores", {})).items()
                }
                historical_scores_zero = all(
                    (
                        value <= 0.0
                        for (context_id, value) in scores.items()
                        if context_id != source
                    )
                )
                new_score_zero = float(last.get("new_score", math.inf)) <= 0.0
        row_ids = [int(row["transition_id"]) for row in rows]
        record = {
            "event": "TRIAL_CHECKPOINT",
            "trial_id": int(self._lease.trial_id),
            "checkpoint_count": int(len(self._lease.checkpoint_records) + 1),
            "source_context_id": int(self._lease.source_context_id),
            "candidate_context_id": int(self._lease.candidate_context_id),
            "block_first_step": None if not row_ids else row_ids[0],
            "block_last_step": None if not row_ids else row_ids[-1],
            "observed_rows": int(len(row_ids)),
            "row_ids": row_ids,
            "trial_mean_llr": mean_llr,
            "historical_scores_zero": bool(historical_scores_zero),
            "new_score_zero": bool(new_score_zero),
            "evidence_cleared": bool(
                complete
                and identity_matches
                and historical_scores_zero
                and new_score_zero
            ),
            "identity_matches": bool(identity_matches),
        }
        self._lease.checkpoint_records.append(copy.deepcopy(record))
        self._latest_checkpoint = copy.deepcopy(record)
        self._records.append(copy.deepcopy(record))

    def evidence_dissipated_at_checkpoint(self) -> bool:
        record = self._latest_checkpoint
        if self._lease is None or record is None:
            return False
        return bool(
            int(record.get("trial_id", -1)) == int(self._lease.trial_id)
            and record.get("evidence_cleared", False)
            and (record.get("trial_mean_llr") is not None)
            and (float(record["trial_mean_llr"]) <= 0.0)
        )

    def resolve(
        self,
        *,
        end_step: int,
        outcome: str,
        reason: str,
        assigned_context_id: int | None = None,
    ) -> TrialResolution:
        if self._lease is None:
            raise RuntimeError("no active trial lease")
        outcome = str(outcome)
        if outcome not in {"RETURN", "NEW", "INCONCLUSIVE", "ABORTED"}:
            raise ValueError("invalid trial outcome")
        if outcome in {"RETURN", "NEW"} and assigned_context_id is None:
            raise ValueError("successful trial resolution requires an assignment")
        lease = self._lease
        latest_checkpoint = (
            lease.checkpoint_records[-1] if lease.checkpoint_records else None
        )
        mean_reward = (
            None
            if not lease.rows
            else sum((float(row.get("reward", 0.0)) for row in lease.rows))
            / len(lease.rows)
        )
        resolution = TrialResolution(
            trial_id=lease.trial_id,
            outcome=outcome,
            source_context_id=lease.source_context_id,
            candidate_context_id=lease.candidate_context_id,
            start_step=lease.start_step,
            end_step=int(end_step),
            used_steps=lease.used_steps,
            assigned_context_id=(
                None if assigned_context_id is None else int(assigned_context_id)
            ),
            rows=tuple(copy.deepcopy(lease.rows)),
            control_payloads=tuple(copy.deepcopy(lease.control_payloads)),
            reason=str(reason),
            selector_state=copy.deepcopy(lease.selector_state),
            checkpoint_count=int(len(lease.checkpoint_records)),
            trial_mean_llr=(
                None
                if latest_checkpoint is None
                else latest_checkpoint.get("trial_mean_llr")
            ),
            evidence_cleared=bool(
                latest_checkpoint is not None
                and latest_checkpoint.get("evidence_cleared", False)
            ),
            trial_mean_reward=mean_reward,
        )
        if outcome in {"INCONCLUSIVE", "ABORTED"}:
            self.discarded_training_rows += lease.used_steps
        self._records.append(
            {
                "event": f"TRIAL_{outcome}",
                "trial_id": lease.trial_id,
                "step": int(end_step),
                "source_context_id": lease.source_context_id,
                "candidate_context_id": lease.candidate_context_id,
                "assigned_context_id": resolution.assigned_context_id,
                "used_steps": lease.used_steps,
                "reason": str(reason),
                "checkpoint_count": int(resolution.checkpoint_count),
                "trial_mean_llr": resolution.trial_mean_llr,
                "evidence_cleared": bool(resolution.evidence_cleared),
                "trial_mean_reward": resolution.trial_mean_reward,
            }
        )
        self._last_end_step = int(end_step)
        self._lease = None
        self._working_bundle = None
        self._latest_checkpoint = None
        return resolution

    def state_dict(self) -> dict[str, Any]:
        lease = None
        if self._lease is not None:
            lease = {
                key: copy.deepcopy(value) for (key, value) in vars(self._lease).items()
            }
        return {
            "schema": self.schema,
            "enabled": self.enabled,
            "check_interval": self.check_interval,
            "max_trial_steps": self.max_trial_steps,
            "cooldown_steps": self.cooldown_steps,
            "total_ratio": self.total_ratio,
            "absolute_limit": self.absolute_limit,
            "trial_steps_total": self.trial_steps_total,
            "paused_control_updates": self.paused_control_updates,
            "paused_theta_updates": self.paused_theta_updates,
            "paused_predictor_updates": self.paused_predictor_updates,
            "discarded_training_rows": self.discarded_training_rows,
            "next_trial_id": self._next_trial_id,
            "last_end_step": self._last_end_step,
            "lease": lease,
            "records": copy.deepcopy(self._records),
            "latest_checkpoint": copy.deepcopy(self._latest_checkpoint),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.schema:
            raise ValueError("invalid trial controller schema")
        for key in (
            "enabled",
            "check_interval",
            "max_trial_steps",
            "cooldown_steps",
            "total_ratio",
            "absolute_limit",
        ):
            if state[key] != getattr(self, key):
                raise ValueError(f"trial configuration changed: {key}")
        self.trial_steps_total = int(state["trial_steps_total"])
        self.paused_control_updates = int(state["paused_control_updates"])
        self.paused_theta_updates = int(state["paused_theta_updates"])
        self.paused_predictor_updates = int(state["paused_predictor_updates"])
        self.discarded_training_rows = int(state["discarded_training_rows"])
        self._next_trial_id = int(state["next_trial_id"])
        self._last_end_step = state["last_end_step"]
        lease = state.get("lease")
        self._lease = None if lease is None else TrialLease(**copy.deepcopy(lease))
        self._working_bundle = None
        self._records = copy.deepcopy(state["records"])
        self._latest_checkpoint = copy.deepcopy(state.get("latest_checkpoint"))

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._records))

    def audit_metrics(self) -> dict[str, int | float]:
        events = [str(record.get("event", "")) for record in self._records]
        return {
            "trial_started_count": events.count("TRIAL_STARTED"),
            "trial_return_count": events.count("TRIAL_RETURN"),
            "trial_new_count": events.count("TRIAL_NEW"),
            "trial_inconclusive_count": events.count("TRIAL_INCONCLUSIVE"),
            "trial_aborted_count": events.count("TRIAL_ABORTED"),
            "trial_rejected_count": events.count("TRIAL_REJECTED"),
            "trial_steps_total": int(self.trial_steps_total),
            "trial_discarded_training_rows": int(self.discarded_training_rows),
            "trial_paused_control_updates": int(self.paused_control_updates),
            "trial_paused_theta_updates": int(self.paused_theta_updates),
            "trial_paused_predictor_updates": int(self.paused_predictor_updates),
        }
