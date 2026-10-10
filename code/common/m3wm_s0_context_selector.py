from __future__ import annotations
import copy
import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping
import torch
from common.m3wm_s0_detection_view import DetectionTrainingView


def _finite_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True)
class S0Decision:
    action: str
    source_context_id: int
    target_context_id: int | None
    transition_id: int
    ready: bool
    reason: str
    scores: dict[int, float] = field(default_factory=dict)
    new_score: float = 0.0
    log_probs: dict[int, float] = field(default_factory=dict)
    active_log_prob: float | None = None
    behavior_context_id: int | None = None
    active_disagreement: float | None = None
    context_disagreements: dict[int, float] = field(default_factory=dict)
    active_uncertain_diagnostic: bool = False
    evidence_peak: float = 0.0
    investigation_active: bool = False
    identity_row_disposition: str = "TRAIN"
    comparison_performed: bool = False
    raw_action: str | None = None
    hold_reason: str | None = None
    settle_until: int | None = None
    candidate_context_id: int | None = None
    confirmation_block_count: int = 0
    raw_scores: dict[int, float] = field(default_factory=dict)
    confirmation_block: dict[str, Any] | None = None
    new_increment: float | None = None
    grouped_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    predictor_revisions: dict[int, int] = field(default_factory=dict)
    applicability_shadow: dict[str, Any] = field(default_factory=dict)
    new_confirmation_active: bool = False
    new_transaction_id: str | None = None
    new_nomination_step: int | None = None
    new_source_revision: int | None = None
    confirmation_method: str | None = None
    natural_passed: bool | None = None
    effective_passed: bool | None = None
    residual_reference_ready: bool = False
    residual_reference_count: int = 0
    residual_center: float | None = None
    residual_tail_rank: float | None = None
    residual_direction: int = 0
    automatic_new_success: bool = False

    def to_record(self) -> dict[str, Any]:

        def finite(value):
            return float(value) if value is not None and math.isfinite(value) else None

        return {
            "action": str(self.action),
            "source_context_id": int(self.source_context_id),
            "target_context_id": (
                None if self.target_context_id is None else int(self.target_context_id)
            ),
            "transition_id": int(self.transition_id),
            "ready": bool(self.ready),
            "reason": str(self.reason),
            "scores": {str(k): finite(v) for (k, v) in self.scores.items()},
            "new_score": finite(self.new_score),
            "log_probs": {str(k): finite(v) for (k, v) in self.log_probs.items()},
            "active_log_prob": finite(self.active_log_prob),
            "behavior_context_id": (
                None
                if self.behavior_context_id is None
                else int(self.behavior_context_id)
            ),
            "active_disagreement": finite(self.active_disagreement),
            "context_disagreements": {
                str(k): finite(v) for (k, v) in self.context_disagreements.items()
            },
            "active_uncertain_diagnostic": bool(self.active_uncertain_diagnostic),
            "evidence_peak": finite(self.evidence_peak),
            "investigation_active": bool(self.investigation_active),
            "identity_row_disposition": str(self.identity_row_disposition),
            "comparison_performed": bool(self.comparison_performed),
            "raw_action": self.raw_action,
            "admitted_action": str(self.action),
            "hold_reason": self.hold_reason,
            "settle_until": self.settle_until,
            "candidate_context_id": self.candidate_context_id,
            "confirmation_block_count": int(self.confirmation_block_count),
            "raw_scores": {str(k): finite(v) for (k, v) in self.raw_scores.items()},
            "confirmation_block": copy.deepcopy(self.confirmation_block),
            "new_increment": finite(self.new_increment),
            "grouped_stats": {
                str(context_id): {
                    str(key): finite(value) for (key, value) in stats.items()
                }
                for (context_id, stats) in self.grouped_stats.items()
            },
            "predictor_revisions": {
                str(key): int(value)
                for (key, value) in self.predictor_revisions.items()
            },
            "applicability_shadow": copy.deepcopy(self.applicability_shadow),
            "new_confirmation_active": bool(self.new_confirmation_active),
            "new_transaction_id": self.new_transaction_id,
            "new_nomination_step": self.new_nomination_step,
            "new_source_revision": self.new_source_revision,
            "confirmation_method": self.confirmation_method,
            "natural_passed": self.natural_passed,
            "effective_passed": self.effective_passed,
            "residual_reference_ready": bool(self.residual_reference_ready),
            "residual_reference_count": int(self.residual_reference_count),
            "residual_center": finite(self.residual_center),
            "residual_tail_rank": finite(self.residual_tail_rank),
            "residual_direction": int(self.residual_direction),
            "automatic_new_success": bool(self.automatic_new_success),
        }


@dataclass
class _PredictorSlot:
    context_id: int
    predictor: Any
    training_view: DetectionTrainingView
    fitted_rows: int = 0
    initial_fit_complete: bool = False
    optimizer_updates: int = 0
    maintenance_rounds: int = 0

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return self.training_view.recent_rows()


class S0PNNEvidenceContextSelector:
    schema = "reco-wm-s0-pnn-evidence-context-selector-v1"
    allowed_transition_fields = {
        "transition_id",
        "obs",
        "action",
        "reward",
        "next_obs",
        "terminated",
        "truncated",
    }

    def __init__(
        self,
        *,
        initial_predictor: Any,
        predictor_factory: Callable[[int], Any],
        threshold: float,
        new_log_prob: Callable[[Any, Mapping[str, Any]], float] | None = None,
        initial_fit_updates: int = 200,
        maintenance_every: int = 250,
        maintenance_updates: int = 20,
        batch_size: int = 256,
        min_ready_rows: int = 512,
        max_contexts: int = 8,
        max_disagreement: float = 0.5,
        detection_recent_capacity: int = 1024,
        detection_reservoir_capacity: int = 4096,
        detection_seed: int = 9017,
        trial_requests_enabled: bool = False,
        trial_probe_fraction: float = 0.5,
        settle_after_new_steps: int = 0,
        settle_after_return_steps: int = 0,
        new_min_maintenance_rounds: int = 0,
        return_confirmation_enabled: bool = False,
        return_confirm_block_size: int = 32,
        return_confirm_blocks: int = 2,
        failed_candidate_hold_steps: int = 0,
        post_new_failed_candidate_hold_steps: int = 2048,
        post_new_return_confirm_blocks: int = 3,
        new_confirmation_enabled: bool = False,
        new_confirm_block_size: int = 32,
        new_confirm_blocks: int = 2,
        residual_confirmation_enabled: bool = False,
        residual_block_size: int = 32,
        residual_reference_min_blocks: int = 128,
        residual_reference_max_blocks: int = 256,
        residual_tail_rank_alpha: float = 0.01,
    ) -> None:
        if threshold <= 0 or not math.isfinite(threshold):
            raise ValueError("threshold must be finite and positive")
        if min_ready_rows <= 0 or maintenance_every <= 0:
            raise ValueError("row budgets must be positive")
        self.threshold = float(threshold)
        self.predictor_factory = predictor_factory
        self.new_log_prob = new_log_prob
        self.initial_fit_updates = int(initial_fit_updates)
        self.maintenance_every = int(maintenance_every)
        self.maintenance_updates = int(maintenance_updates)
        self.batch_size = int(batch_size)
        self.min_ready_rows = int(min_ready_rows)
        self.max_contexts = int(max_contexts)
        self.max_disagreement = float(max_disagreement)
        self.detection_recent_capacity = int(detection_recent_capacity)
        self.detection_reservoir_capacity = int(detection_reservoir_capacity)
        self.detection_seed = int(detection_seed)
        if not 0.0 < float(trial_probe_fraction) < 1.0:
            raise ValueError("trial probe fraction must be in (0, 1)")
        self.trial_requests_enabled = bool(trial_requests_enabled)
        self.trial_probe_fraction = float(trial_probe_fraction)
        self.settle_after_new_steps = int(settle_after_new_steps)
        self.settle_after_return_steps = int(settle_after_return_steps)
        self.new_min_maintenance_rounds = int(new_min_maintenance_rounds)
        self.return_confirmation_enabled = bool(return_confirmation_enabled)
        self.return_confirm_block_size = int(return_confirm_block_size)
        self.return_confirm_blocks = int(return_confirm_blocks)
        self.failed_candidate_hold_steps = int(failed_candidate_hold_steps)
        self.post_new_failed_candidate_hold_steps = int(
            post_new_failed_candidate_hold_steps
        )
        self.post_new_return_confirm_blocks = int(post_new_return_confirm_blocks)
        self.new_confirmation_enabled = bool(new_confirmation_enabled)
        self.new_confirm_block_size = int(new_confirm_block_size)
        self.new_confirm_blocks = int(new_confirm_blocks)
        self.residual_confirmation_enabled = bool(residual_confirmation_enabled)
        self.residual_block_size = int(residual_block_size)
        self.residual_reference_min_blocks = int(residual_reference_min_blocks)
        self.residual_reference_max_blocks = int(residual_reference_max_blocks)
        self.residual_tail_rank_alpha = float(residual_tail_rank_alpha)
        if not 0.0 < self.residual_tail_rank_alpha < 1.0:
            raise ValueError("residual tail rank alpha must be in (0,1)")
        if (
            min(
                self.settle_after_new_steps,
                self.settle_after_return_steps,
                self.new_min_maintenance_rounds,
                self.return_confirm_block_size,
                self.return_confirm_blocks,
                self.failed_candidate_hold_steps,
                self.post_new_failed_candidate_hold_steps,
                self.post_new_return_confirm_blocks,
                self.new_confirm_block_size,
                self.new_confirm_blocks,
            )
            < 0
            or min(
                self.return_confirm_block_size,
                self.return_confirm_blocks,
                self.new_confirm_block_size,
                self.new_confirm_blocks,
                self.residual_block_size,
                self.residual_reference_min_blocks,
                self.residual_reference_max_blocks,
            )
            == 0
        ):
            raise ValueError("invalid S0 identity-stability configuration")
        self.active_context_id = 0
        self._slots = {0: self._new_slot(0, initial_predictor)}
        self._scores = {0: 0.0}
        self._raw_scores = {0: 0.0}
        self._new_score = 0.0
        self._pending: S0Decision | None = None
        self._last_transition_id = -1
        self._decision_records: list[dict[str, Any]] = []
        self.train_wall_ms = 0.0
        self.investigating = False
        self.investigation_source_context_id: int | None = None
        self.pending_identity_rows: list[dict[str, Any]] = []
        self.investigation_count = 0
        self.investigation_timeout_count = 0
        self._pending_investigation_snapshot: dict[str, Any] | None = None
        self.ready_steps = 0
        self.active_uncertain_diagnostic_count = 0
        self.uncertain_but_scored_count = 0
        self.history_logprob_call_count = 0
        self.new_alt_call_count = 0
        self.investigation_steps = 0
        self.pending_identity_rows_peak = 0
        self.identity_updates_during_investigation = 0
        self.trial_request_count = 0
        self.new_count = 0
        self.return_count = 0
        self.failed_trial_selector_rollback_count = 0
        self.last_switch_transition_id: int | None = None
        self.last_switch_action: str | None = None
        self.settle_until = -1
        self._newborn_context_id: int | None = None
        self._newborn_maintenance_baseline = 0
        self._candidate_holds: dict[tuple[int, int], int] = {}
        self._post_new_return_requirements: dict[tuple[int, int], int] = {}
        self._rejected_candidates_pending_new: set[tuple[int, int]] = set()
        self._return_confirmation: dict[str, Any] | None = None
        self._last_confirmation_block: dict[str, Any] | None = None
        self.return_confirmation_started_count = 0
        self.return_confirmation_passed_count = 0
        self.return_confirmation_rejected_count = 0
        self._new_confirmation: dict[str, Any] | None = None
        self.new_confirmation_started_count = 0
        self.new_confirmation_passed_count = 0
        self.new_confirmation_rejected_count = 0
        self._fit_records: list[dict[str, Any]] = []
        self._residual_reference_blocks: dict[int, list[dict[str, Any]]] = {0: []}
        self._residual_current_block: dict[int, list[dict[str, Any]]] = {0: []}
        self._residual_frozen: dict[str, dict[str, Any]] = {}
        self._reference_candidates: dict[int, dict[str, Any]] = {}

    def _new_slot(self, context_id: int, predictor: Any) -> _PredictorSlot:
        return _PredictorSlot(
            context_id=int(context_id),
            predictor=predictor,
            training_view=DetectionTrainingView(
                recent_capacity=self.detection_recent_capacity,
                reservoir_capacity=self.detection_reservoir_capacity,
                seed=self.detection_seed + int(context_id) * 1000003,
            ),
        )

    @staticmethod
    def _clean_row(transition: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(transition) - S0PNNEvidenceContextSelector.allowed_transition_fields
        if unknown:
            raise ValueError(f"forbidden selector fields: {sorted(unknown)}")
        missing = {
            "transition_id",
            "obs",
            "action",
            "reward",
            "next_obs",
            "terminated",
            "truncated",
        } - set(transition)
        if missing:
            raise ValueError(f"missing selector fields: {sorted(missing)}")
        row = {key: copy.deepcopy(transition[key]) for key in sorted(transition)}
        transition_id = int(row["transition_id"])
        if transition_id < 0:
            raise ValueError("transition_id must be non-negative")
        for key in ("obs", "action", "next_obs"):
            value = torch.as_tensor(row[key])
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite {key}")
        if not math.isfinite(float(row["reward"])):
            raise FloatingPointError("non-finite reward")
        return row

    def _slot_ready(self, slot: _PredictorSlot) -> bool:
        return bool(
            slot.initial_fit_complete and getattr(slot.predictor, "fitted", False)
        )

    @property
    def ready(self) -> bool:
        return self._slot_ready(self._slots[self.active_context_id])

    def _train_if_due(self, slot: _PredictorSlot) -> None:
        started = time.perf_counter()
        fit_kind = None
        before_updates = int(slot.optimizer_updates)
        try:
            accepted_rows = len(slot.training_view)
            if not slot.initial_fit_complete and accepted_rows >= self.min_ready_rows:
                slot.predictor.fit_initial(
                    slot.training_view,
                    updates=self.initial_fit_updates,
                    batch_size=self.batch_size,
                )
                slot.initial_fit_complete = bool(
                    getattr(slot.predictor, "fitted", False)
                )
                slot.fitted_rows = accepted_rows
                slot.optimizer_updates += self.initial_fit_updates
                fit_kind = "initial"
            elif (
                slot.initial_fit_complete
                and accepted_rows - slot.fitted_rows >= self.maintenance_every
            ):
                slot.predictor.fit_incremental(
                    slot.training_view,
                    updates=self.maintenance_updates,
                    batch_size=self.batch_size,
                )
                slot.fitted_rows = accepted_rows
                slot.optimizer_updates += self.maintenance_updates
                slot.maintenance_rounds += 1
                fit_kind = "maintenance"
        finally:
            self.train_wall_ms += (time.perf_counter() - started) * 1000.0
        if fit_kind is not None:
            recent = slot.training_view.recent_rows()
            ids = [int(row["transition_id"]) for row in recent]
            self._fit_records.append(
                {
                    "event": "PNN_FIT",
                    "context_id": int(slot.context_id),
                    "fit_kind": fit_kind,
                    "accepted_rows": int(len(slot.training_view)),
                    "recent_first_transition_id": None if not ids else min(ids),
                    "recent_last_transition_id": None if not ids else max(ids),
                    "recent_row_count": int(len(ids)),
                    "recent_row_ids_sha256": hashlib.sha256(
                        ",".join((str(value) for value in ids)).encode("ascii")
                    ).hexdigest(),
                    "sample_stats": slot.training_view.last_sample_stats,
                    "optimizer_updates_before": before_updates,
                    "optimizer_updates_after": int(slot.optimizer_updates),
                    "maintenance_rounds": int(slot.maintenance_rounds),
                }
            )

    def _reset_evidence(self) -> None:
        self._scores = {context_id: 0.0 for context_id in self._slots}
        self._raw_scores = {context_id: 0.0 for context_id in self._slots}
        self._new_score = 0.0
        self._return_confirmation = None
        self._new_confirmation = None
        self.investigating = False
        self.investigation_source_context_id = None

    @staticmethod
    def _predictor_revision(slot: _PredictorSlot) -> int:
        return int(getattr(slot.predictor, "total_updates", slot.optimizer_updates))

    def _settling_reason(self, transition_id: int) -> str | None:
        if self.last_switch_action is None:
            return None
        if int(transition_id) < int(self.settle_until):
            return f"settle_after_{self.last_switch_action.lower()}"
        if self.last_switch_action == "NEW" and self._newborn_context_id is not None:
            slot = self._slots[self._newborn_context_id]
            gained = int(slot.maintenance_rounds) - int(
                self._newborn_maintenance_baseline
            )
            if gained < self.new_min_maintenance_rounds:
                return "new_context_maintenance_pending"
        self.last_switch_action = None
        self.settle_until = -1
        self._newborn_context_id = None
        self._reset_evidence()
        return None

    def _candidate_held(self, candidate_id: int, transition_id: int) -> bool:
        key = (int(self.active_context_id), int(candidate_id))
        until = int(self._candidate_holds.get(key, -1))
        if until >= 0 and int(transition_id) >= until:
            self._candidate_holds.pop(key, None)
            return False
        return until >= 0

    def _required_return_blocks(self, candidate_id: int, transition_id: int) -> int:
        key = (int(self.active_context_id), int(candidate_id))
        if key in self._post_new_return_requirements:
            return max(self.return_confirm_blocks, self.post_new_return_confirm_blocks)
        return self.return_confirm_blocks

    def can_start_trial(
        self, candidate_id: int, transition_id: int
    ) -> tuple[bool, str]:
        reason = self._settling_reason(int(transition_id))
        if reason is not None:
            return (False, reason)
        if self._candidate_held(int(candidate_id), int(transition_id)):
            return (False, "failed_candidate_hold")
        return (True, "selector_trial_eligible")

    def register_failed_candidate(
        self,
        *,
        source_context_id: int,
        candidate_context_id: int,
        end_transition_id: int,
    ) -> None:
        key = (int(source_context_id), int(candidate_context_id))
        self._candidate_holds[key] = (
            int(end_transition_id) + self.failed_candidate_hold_steps
        )
        self._rejected_candidates_pending_new.add(key)
        self._scores[int(candidate_context_id)] = 0.0
        if self._return_confirmation is not None and int(
            self._return_confirmation["candidate_context_id"]
        ) == int(candidate_context_id):
            self._return_confirmation = None

    def _grouped_stats(
        self, slot, row, prediction, log_prob: float
    ) -> dict[str, float]:
        fn = getattr(slot.predictor, "grouped_stats_from_prediction", None)
        if prediction is not None and callable(fn):
            return dict(fn(row, *prediction))
        return {
            "logp_joint": float(log_prob),
            "logp_dynamics": float(log_prob),
            "logp_reward": 0.0,
            "q_dynamics_mean": 0.0,
            "q_reward": 0.0,
            "reward_residual": 0.0,
            "reward_total_variance": 1.0,
        }

    def _residual_reference_ready(self, context_id: int) -> bool:
        return bool(
            self.residual_confirmation_enabled
            and len(self._residual_reference_blocks.get(int(context_id), ()))
            >= self.residual_reference_min_blocks
        )

    @staticmethod
    def _residual_direction(value: float, center: float) -> int:
        delta = float(value) - float(center)
        return 1 if delta > 0.0 else -1 if delta < 0.0 else 0

    def _freeze_residual_reference(
        self, *, source_context_id: int, transaction_id: str
    ) -> dict[str, Any]:
        blocks = copy.deepcopy(
            self._residual_reference_blocks.get(int(source_context_id), [])
        )
        values = [float(item["mean_residual"]) for item in blocks]
        center = (
            float(torch.tensor(values, dtype=torch.float64).median().item())
            if values
            else float("nan")
        )
        payload = {
            "source_context_id": int(source_context_id),
            "reference_blocks": blocks,
            "reference_count": int(len(blocks)),
            "reference_ready": bool(len(blocks) >= self.residual_reference_min_blocks),
            "center": center,
        }
        payload["reference_hash"] = hashlib.sha256(
            repr(payload).encode("utf-8")
        ).hexdigest()
        self._residual_frozen[str(transaction_id)] = copy.deepcopy(payload)
        return payload

    def _record_reference_row(self, row: Mapping[str, Any]) -> None:
        if not self.residual_confirmation_enabled:
            return
        transition_id = int(row["transition_id"])
        item = self._reference_candidates.pop(transition_id, None)
        if item is None or int(item["context_id"]) != int(self.active_context_id):
            return
        if not math.isfinite(float(item["residual"])):
            return
        context_id = int(item["context_id"])
        current = self._residual_current_block.setdefault(context_id, [])
        if current and transition_id != int(current[-1]["transition_id"]) + 1:
            current.clear()
        current.append(
            {
                "transition_id": transition_id,
                "residual": float(item["residual"]),
                "revision": int(item["revision"]),
            }
        )
        if len(current) < self.residual_block_size:
            return
        values = [float(entry["residual"]) for entry in current]
        block = {
            "block_first_step": int(current[0]["transition_id"]),
            "block_last_step": int(current[-1]["transition_id"]),
            "row_count": int(len(current)),
            "mean_residual": float(sum(values) / len(values)),
            "revision_first": int(current[0]["revision"]),
            "revision_last": int(current[-1]["revision"]),
            "row_ids_sha256": hashlib.sha256(
                ",".join((str(entry["transition_id"]) for entry in current)).encode(
                    "ascii"
                )
            ).hexdigest(),
        }
        blocks = self._residual_reference_blocks.setdefault(context_id, [])
        blocks.append(block)
        if len(blocks) > self.residual_reference_max_blocks:
            del blocks[: -self.residual_reference_max_blocks]
        current.clear()

    def _advance_return_confirmation(
        self,
        *,
        candidate_id: int,
        row: Mapping[str, Any],
        log_probs: Mapping[int, float],
        prediction_cache: Mapping[int, tuple[Any, Any]],
    ) -> str:
        source = int(self.active_context_id)
        candidate = int(candidate_id)
        candidate_revision = self._predictor_revision(self._slots[candidate])
        source_revision = self._predictor_revision(self._slots[source])
        state = self._return_confirmation
        if state is None:
            self._return_confirmation = {
                "source_context_id": source,
                "candidate_context_id": candidate,
                "candidate_revision": candidate_revision,
                "source_revision": source_revision,
                "nomination_step": int(row["transition_id"]),
                "last_observed_transition_id": int(row["transition_id"]),
                "current_block": [],
                "passed_blocks": 0,
                "block_results": [],
                "completed": False,
                "required_blocks": self._required_return_blocks(
                    candidate, int(row["transition_id"])
                ),
            }
            self.return_confirmation_started_count += 1
            return "pending"
        if (
            int(state.get("source_context_id", source)) != source
            or int(state["candidate_context_id"]) != candidate
            or int(state["candidate_revision"]) != candidate_revision
            or (int(state["source_revision"]) != source_revision)
        ):
            self._last_confirmation_block = {
                "event": "confirmation_invalidated",
                "nomination_step": int(state.get("nomination_step", -1)),
                "block_first_step": None,
                "block_last_step": int(row["transition_id"]),
                "observed_rows": 0,
                "missing_rows": [],
                "candidate_context_id": int(state["candidate_context_id"]),
                "source_context_id": int(state.get("source_context_id", source)),
                "candidate_revision": int(state["candidate_revision"]),
                "source_revision": int(state["source_revision"]),
                "passed": False,
                "admit_reason": "source_candidate_or_revision_changed",
                "reject_reason": ["source_candidate_or_revision_changed"],
            }
            self._return_confirmation = None
            return "invalidated"
        if bool(state.get("completed", False)):
            return "passed"
        transition_id = int(row["transition_id"])
        expected_id = int(state["last_observed_transition_id"]) + 1
        if transition_id != expected_id:
            missing = (
                list(range(expected_id, transition_id))
                if transition_id > expected_id
                else []
            )
            self._last_confirmation_block = {
                "event": "confirmation_rejected",
                "nomination_step": int(state["nomination_step"]),
                "block_first_step": (
                    None
                    if not state["current_block"]
                    else int(state["current_block"][0]["transition_id"])
                ),
                "block_last_step": transition_id,
                "observed_rows": int(len(state["current_block"])),
                "missing_rows": missing,
                "candidate_context_id": candidate,
                "source_context_id": source,
                "candidate_revision": candidate_revision,
                "source_revision": source_revision,
                "passed": False,
                "admit_reason": "noncontiguous_future_rows",
                "reject_reason": ["noncontiguous_future_rows"],
            }
            self.return_confirmation_rejected_count += 1
            self.register_failed_candidate(
                source_context_id=source,
                candidate_context_id=candidate,
                end_transition_id=transition_id,
            )
            return "rejected"
        state["last_observed_transition_id"] = transition_id
        candidate_stats = self._grouped_stats(
            self._slots[candidate],
            row,
            prediction_cache.get(candidate),
            float(log_probs[candidate]),
        )
        source_stats = self._grouped_stats(
            self._slots[source],
            row,
            prediction_cache.get(source),
            float(log_probs[source]),
        )
        state["current_block"].append(
            {
                "transition_id": transition_id,
                "llr": float(
                    candidate_stats["logp_joint"] - source_stats["logp_joint"]
                ),
                "candidate_logp_dynamics": float(candidate_stats["logp_dynamics"]),
                "candidate_logp_reward": float(candidate_stats["logp_reward"]),
                "candidate_q_dynamics": float(candidate_stats["q_dynamics_mean"]),
                "candidate_q_reward": float(candidate_stats["q_reward"]),
                "source_logp_dynamics": float(source_stats["logp_dynamics"]),
                "source_logp_reward": float(source_stats["logp_reward"]),
                "source_q_dynamics": float(source_stats["q_dynamics_mean"]),
                "source_q_reward": float(source_stats["q_reward"]),
            }
        )
        if len(state["current_block"]) < self.return_confirm_block_size:
            return "pending"
        block = state["current_block"]
        row_ids = [int(item["transition_id"]) for item in block]
        missing_rows = sorted(
            set(range(row_ids[0], row_ids[-1] + 1)).difference(row_ids)
        )
        result = {
            "event": "confirmation_block",
            "block_index": int(len(state["block_results"])),
            "nomination_step": int(state["nomination_step"]),
            "block_first_step": row_ids[0],
            "block_last_step": row_ids[-1],
            "row_ids": row_ids,
            "row_ids_sha256": hashlib.sha256(
                ",".join((str(value) for value in row_ids)).encode("ascii")
            ).hexdigest(),
            "row_count": int(len(block)),
            "observed_rows": int(len(block)),
            "missing_rows": missing_rows,
            "candidate_context_id": candidate,
            "source_context_id": source,
            "candidate_revision": candidate_revision,
            "source_revision": source_revision,
            "mean_llr": sum((item["llr"] for item in block)) / len(block),
            "candidate_mean_logp_dynamics": sum(
                (item["candidate_logp_dynamics"] for item in block)
            )
            / len(block),
            "candidate_mean_logp_reward": sum(
                (item["candidate_logp_reward"] for item in block)
            )
            / len(block),
            "candidate_mean_q_dynamics": sum(
                (item["candidate_q_dynamics"] for item in block)
            )
            / len(block),
            "candidate_mean_q_reward": sum(
                (item["candidate_q_reward"] for item in block)
            )
            / len(block),
            "candidate_q_dynamics_max": max(
                (item["candidate_q_dynamics"] for item in block)
            ),
            "candidate_q_reward_max": max(
                (item["candidate_q_reward"] for item in block)
            ),
            "source_mean_logp_dynamics": sum(
                (item["source_logp_dynamics"] for item in block)
            )
            / len(block),
            "source_mean_logp_reward": sum(
                (item["source_logp_reward"] for item in block)
            )
            / len(block),
            "source_mean_q_dynamics": sum((item["source_q_dynamics"] for item in block))
            / len(block),
            "source_mean_q_reward": sum((item["source_q_reward"] for item in block))
            / len(block),
            "finite_count": int(len(block)),
            "thresholds": {
                "mean_llr_min": 0.0,
                "candidate_mean_q_dynamics_max": 4.0,
                "candidate_mean_q_reward_max": 4.0,
            },
        }
        result["passed"] = bool(
            result["mean_llr"] > 0.0
            and result["candidate_mean_q_dynamics"] <= 4.0
            and (result["candidate_mean_q_reward"] <= 4.0)
            and (not missing_rows)
        )
        reject_reasons = []
        if missing_rows:
            reject_reasons.append("noncontiguous_future_rows")
        if not math.isfinite(result["mean_llr"]):
            reject_reasons.append("nonfinite_statistics")
        elif result["mean_llr"] <= 0.0:
            reject_reasons.append("llr_not_positive")
        if not math.isfinite(result["candidate_mean_q_dynamics"]):
            reject_reasons.append("nonfinite_statistics")
        elif result["candidate_mean_q_dynamics"] > 4.0:
            reject_reasons.append("dynamics_residual_over_threshold")
        if not math.isfinite(result["candidate_mean_q_reward"]):
            reject_reasons.append("nonfinite_statistics")
        elif result["candidate_mean_q_reward"] > 4.0:
            reject_reasons.append("reward_residual_over_threshold")
        result["reject_reason"] = sorted(set(reject_reasons))
        result["admit_reason"] = (
            "continuous_future_block_passed"
            if result["passed"]
            else "continuous_future_block_rejected"
        )
        state["block_results"].append(result)
        state["current_block"] = []
        self._last_confirmation_block = dict(result)
        if not result["passed"]:
            self.return_confirmation_rejected_count += 1
            self.register_failed_candidate(
                source_context_id=source,
                candidate_context_id=candidate,
                end_transition_id=int(row["transition_id"]),
            )
            return "rejected"
        state["passed_blocks"] += 1
        if int(state["passed_blocks"]) >= int(
            state.get("required_blocks", self.return_confirm_blocks)
        ):
            state["completed"] = True
            self.return_confirmation_passed_count += 1
            return "passed"
        return "pending"

    def _advance_new_confirmation(
        self,
        *,
        row: Mapping[str, Any],
        new_increment: float,
        new_log_prob: float,
        source_stats: Mapping[str, float],
    ) -> str:
        source = int(self.active_context_id)
        revision = self._predictor_revision(self._slots[source])
        transition_id = int(row["transition_id"])
        state = self._new_confirmation
        if state is None:
            transaction_id = f"new:{source}:{transition_id}:{revision}"
            self._new_confirmation = {
                "kind": "NEW",
                "source_context_id": source,
                "source_revision": revision,
                "nomination_step": transition_id,
                "last_observed_transition_id": transition_id,
                "current_block": [],
                "passed_blocks": 0,
                "block_results": [],
                "completed": False,
                "required_blocks": self.new_confirm_blocks,
                "transaction_id": transaction_id,
                "residual_reference": (
                    self._freeze_residual_reference(
                        source_context_id=source, transaction_id=transaction_id
                    )
                    if self.residual_confirmation_enabled
                    else {
                        "reference_ready": False,
                        "reference_count": 0,
                        "center": float("nan"),
                        "reference_blocks": [],
                        "reference_hash": "",
                    }
                ),
                "residual_block_results": [],
                "residual_current_block": [],
            }
            self.new_confirmation_started_count += 1
            return "pending"
        if (
            int(state["source_context_id"]) != source
            or int(state["source_revision"]) != revision
        ):
            self._last_confirmation_block = {
                "kind": "NEW",
                "event": "new_confirmation_invalidated",
                "nomination_step": int(state["nomination_step"]),
                "block_first_step": None,
                "block_last_step": transition_id,
                "observed_rows": int(len(state["current_block"])),
                "missing_rows": [],
                "source_context_id": int(state["source_context_id"]),
                "source_revision": int(state["source_revision"]),
                "passed": False,
                "admit_reason": "source_or_revision_changed",
                "reject_reason": ["source_or_revision_changed"],
            }
            self._new_confirmation = None
            self.new_confirmation_rejected_count += 1
            return "rejected"
        if bool(state.get("completed", False)):
            return "passed"
        expected = int(state["last_observed_transition_id"]) + 1
        if transition_id != expected:
            missing = (
                list(range(expected, transition_id)) if transition_id > expected else []
            )
            self._last_confirmation_block = {
                "kind": "NEW",
                "event": "new_confirmation_rejected",
                "nomination_step": int(state["nomination_step"]),
                "block_first_step": (
                    None
                    if not state["current_block"]
                    else int(state["current_block"][0]["transition_id"])
                ),
                "block_last_step": transition_id,
                "observed_rows": int(len(state["current_block"])),
                "missing_rows": missing,
                "source_context_id": source,
                "source_revision": revision,
                "passed": False,
                "admit_reason": "noncontiguous_future_rows",
                "reject_reason": ["noncontiguous_future_rows"],
            }
            self._new_confirmation = None
            self.new_confirmation_rejected_count += 1
            return "rejected"
        state["last_observed_transition_id"] = transition_id
        state["current_block"].append(
            {
                "transition_id": transition_id,
                "new_increment": float(new_increment),
                "new_log_prob": float(new_log_prob),
                "source_logp_joint": float(source_stats["logp_joint"]),
                "source_logp_dynamics": float(source_stats["logp_dynamics"]),
                "source_logp_reward": float(source_stats["logp_reward"]),
                "source_q_dynamics": float(source_stats["q_dynamics_mean"]),
                "source_q_reward": float(source_stats["q_reward"]),
                "source_reward_residual": float(source_stats["reward_residual"]),
                "source_reward_total_variance": float(
                    source_stats["reward_total_variance"]
                ),
                "source_reward_residual": float(
                    source_stats.get("reward_residual", float("nan"))
                ),
            }
        )
        residual_ref = state.get("residual_reference", {})
        if bool(residual_ref.get("reference_ready", False)):
            state.setdefault("residual_current_block", []).append(
                {
                    "transition_id": transition_id,
                    "residual": float(
                        source_stats.get("reward_residual", float("nan"))
                    ),
                    "revision": revision,
                }
            )
        if len(state["current_block"]) < self.new_confirm_block_size:
            return "pending"
        block = state["current_block"]
        row_ids = [int(item["transition_id"]) for item in block]
        missing = sorted(set(range(row_ids[0], row_ids[-1] + 1)).difference(row_ids))
        result = {
            "kind": "NEW",
            "event": "new_confirmation_block",
            "block_index": int(len(state["block_results"])),
            "nomination_step": int(state["nomination_step"]),
            "block_first_step": row_ids[0],
            "block_last_step": row_ids[-1],
            "row_ids": row_ids,
            "row_ids_sha256": hashlib.sha256(
                ",".join((str(value) for value in row_ids)).encode("ascii")
            ).hexdigest(),
            "row_count": int(len(row_ids)),
            "observed_rows": int(len(row_ids)),
            "missing_rows": missing,
            "source_context_id": source,
            "source_revision": revision,
            "mean_new_increment": sum((item["new_increment"] for item in block))
            / len(block),
            "mean_new_log_prob": sum((item["new_log_prob"] for item in block))
            / len(block),
            "source_mean_logp_joint": sum((item["source_logp_joint"] for item in block))
            / len(block),
            "source_mean_logp_dynamics": sum(
                (item["source_logp_dynamics"] for item in block)
            )
            / len(block),
            "source_mean_logp_reward": sum(
                (item["source_logp_reward"] for item in block)
            )
            / len(block),
            "source_mean_q_dynamics": sum((item["source_q_dynamics"] for item in block))
            / len(block),
            "source_mean_q_reward": sum((item["source_q_reward"] for item in block))
            / len(block),
            "source_mean_signed_reward_residual": sum(
                (item["source_reward_residual"] for item in block)
            )
            / len(block),
            "source_mean_reward_total_variance": sum(
                (item["source_reward_total_variance"] for item in block)
            )
            / len(block),
            "thresholds": {"mean_new_increment_min_exclusive": 0.0},
        }
        result["passed"] = bool(
            math.isfinite(result["mean_new_increment"])
            and result["mean_new_increment"] > 0.0
            and (not missing)
        )
        result["reject_reason"] = (
            []
            if result["passed"]
            else [
                (
                    "new_increment_not_positive"
                    if not missing
                    else "noncontiguous_future_rows"
                )
            ]
        )
        result["admit_reason"] = (
            "continuous_new_future_block_passed"
            if result["passed"]
            else "continuous_new_future_block_rejected"
        )
        residual_result = {
            "reference_ready": bool(residual_ref.get("reference_ready", False)),
            "reference_count": int(residual_ref.get("reference_count", 0)),
            "center": float(residual_ref.get("center", float("nan"))),
            "tail_rank": None,
            "direction": 0,
            "passed": False,
            "block_first_step": row_ids[0],
            "block_last_step": row_ids[-1],
            "revision_first": revision,
            "revision_last": revision,
        }
        residual_block = list(state.get("residual_current_block", []))
        if (
            residual_result["reference_ready"]
            and len(residual_block) == self.residual_block_size
        ):
            b = sum((float(x["residual"]) for x in residual_block)) / len(
                residual_block
            )
            center = float(residual_result["center"])
            refs = [
                float(x["mean_residual"])
                for x in residual_ref.get("reference_blocks", [])
            ]
            rank = (1 + sum((abs(ref - center) >= abs(b - center) for ref in refs))) / (
                len(refs) + 1
            )
            signs = [self._residual_direction(b, center)]
            residual_result.update(
                {
                    "block_mean_residual": float(b),
                    "tail_rank": float(rank),
                    "direction": int(signs[0]),
                    "revision_first": int(residual_block[0]["revision"]),
                    "revision_last": int(residual_block[-1]["revision"]),
                    "passed": bool(
                        math.isfinite(b)
                        and math.isfinite(rank)
                        and (rank <= self.residual_tail_rank_alpha)
                        and (signs[0] != 0)
                    ),
                }
            )
        state.setdefault("residual_block_results", []).append(
            copy.deepcopy(residual_result)
        )
        state["residual_current_block"] = []
        result.update(
            {
                "residual_reference_ready": bool(residual_result["reference_ready"]),
                "residual_reference_count": int(residual_result["reference_count"]),
                "residual_center": residual_result["center"],
                "residual_tail_rank": residual_result["tail_rank"],
                "residual_direction": int(residual_result["direction"]),
                "residual_passed": bool(residual_result["passed"]),
                "residual_revision_first": residual_result["revision_first"],
                "residual_revision_last": residual_result["revision_last"],
            }
        )
        state["block_results"].append(copy.deepcopy(result))
        state["current_block"] = []
        self._last_confirmation_block = copy.deepcopy(result)
        if not result["passed"] and (not bool(residual_result["passed"])):
            if self.residual_confirmation_enabled and bool(
                residual_ref.get("reference_ready", False)
            ):
                state["passed_blocks"] = int(state.get("passed_blocks", 0))
                if len(state["block_results"]) < int(state["required_blocks"]):
                    self.new_confirmation_rejected_count += 1
                    return "pending"
            self._new_confirmation = None
            self.new_confirmation_rejected_count += 1
            return "rejected"
        if result["passed"]:
            state["passed_blocks"] += 1
        residual_results = state.get("residual_block_results", [])
        residual_path = bool(
            residual_ref.get("reference_ready", False)
            and len(residual_results) >= int(state["required_blocks"])
            and all(
                (
                    bool(item.get("passed", False))
                    for item in residual_results[-int(state["required_blocks"]) :]
                )
            )
            and (
                len(
                    {
                        int(item.get("direction", 0))
                        for item in residual_results[-int(state["required_blocks"]) :]
                    }
                )
                == 1
            )
            and (int(residual_results[-1].get("direction", 0)) != 0)
        )
        old_path = bool(
            len(state["block_results"]) >= int(state["required_blocks"])
            and all(
                (
                    bool(item.get("passed", False))
                    for item in state["block_results"][-int(state["required_blocks"]) :]
                )
            )
        )
        if len(state["block_results"]) >= int(state["required_blocks"]):
            state["completed"] = True
            state["old_path_passed"] = old_path
            state["residual_path_passed"] = residual_path
            state["confirmation_method"] = (
                "legacy_joint"
                if old_path
                else "frozen_signed_residual" if residual_path else None
            )
            result["old_path_passed"] = old_path
            result["residual_path_passed"] = residual_path
            result["confirmation_method"] = state["confirmation_method"]
            if old_path or residual_path:
                self.new_confirmation_passed_count += 1
                return "passed"
            self.new_confirmation_rejected_count += 1
            self._new_confirmation = None
            return "rejected"
        return "pending"

    def decide(
        self,
        transition: Mapping[str, Any],
        *,
        behavior_context_id: int,
        allow_uncommitted_behavior: bool = False,
    ) -> S0Decision:
        if self._pending is not None:
            raise RuntimeError("selector decision must be committed or aborted first")
        self._last_confirmation_block = None
        row = self._clean_row(transition)
        transition_id = int(row["transition_id"])
        if transition_id <= self._last_transition_id:
            raise ValueError("transition_id must increase strictly")
        behavior = int(behavior_context_id)
        if behavior != self.active_context_id and (
            not (allow_uncommitted_behavior and behavior in self._slots)
        ):
            raise RuntimeError("behavior Context and selector active Context disagree")
        active_slot = self._slots[self.active_context_id]
        if not self._slot_ready(active_slot):
            return S0Decision(
                "WARMUP",
                self.active_context_id,
                None,
                transition_id,
                False,
                "active_predictor_not_ready",
                behavior_context_id=behavior_context_id,
            )
        self.ready_steps += 1
        prediction_cache: dict[int, tuple[Any, Any]] = {}
        for context_id, slot in self._slots.items():
            if not self._slot_ready(slot):
                continue
            predictor = slot.predictor
            predict = getattr(predictor, "predict_physical", None)
            disagreement_from = getattr(predictor, "disagreement_from_prediction", None)
            log_prob_from = getattr(predictor, "log_prob_from_prediction", None)
            new_from = getattr(
                predictor, "new_alternative_log_prob_from_prediction", None
            )
            needs_new = (
                context_id == self.active_context_id and self.new_log_prob is None
            )
            if (
                callable(predict)
                and callable(disagreement_from)
                and callable(log_prob_from)
                and (not needs_new or callable(new_from))
            ):
                prediction_cache[int(context_id)] = predict(row["obs"], row["action"])
        context_disagreements: dict[int, float] = {}
        for context_id, slot in self._slots.items():
            if not self._slot_ready(slot):
                continue
            prediction = prediction_cache.get(int(context_id))
            if prediction is None:
                disagreement_fn = getattr(slot.predictor, "disagreement", None)
                value = (
                    0.0
                    if not callable(disagreement_fn)
                    else float(disagreement_fn(row))
                )
            else:
                value = float(slot.predictor.disagreement_from_prediction(*prediction))
            if not math.isfinite(value):
                raise FloatingPointError("non-finite Context predictor disagreement")
            context_disagreements[int(context_id)] = value
        active_disagreement = context_disagreements.get(self.active_context_id, 0.0)
        active_uncertain = bool(active_disagreement >= self.max_disagreement)
        log_probs: dict[int, float] = {}
        for context_id, slot in self._slots.items():
            if not self._slot_ready(slot):
                continue
            prediction = prediction_cache.get(int(context_id))
            value = (
                float(slot.predictor.log_prob(row))
                if prediction is None
                else float(slot.predictor.log_prob_from_prediction(row, *prediction))
            )
            if not math.isfinite(value):
                raise FloatingPointError("non-finite predictor log probability")
            log_probs[context_id] = value
            if context_id != self.active_context_id:
                self.history_logprob_call_count += 1
        active_log_prob = log_probs[self.active_context_id]
        grouped_stats = {
            int(context_id): self._grouped_stats(
                self._slots[context_id],
                row,
                prediction_cache.get(int(context_id)),
                float(value),
            )
            for (context_id, value) in log_probs.items()
        }
        predictor_revisions = {
            int(context_id): self._predictor_revision(self._slots[context_id])
            for context_id in log_probs
        }
        applicability_shadow = {
            "policy": "read_only_q_le_4_v1",
            "source_context_id": int(self.active_context_id),
            "contexts": {
                int(context_id): {
                    "dynamics_usable": bool(
                        math.isfinite(float(stats["q_dynamics_mean"]))
                        and float(stats["q_dynamics_mean"]) <= 4.0
                    ),
                    "reward_usable": bool(
                        math.isfinite(float(stats["q_reward"]))
                        and float(stats["q_reward"]) <= 4.0
                    ),
                    "joint_usable": bool(
                        math.isfinite(float(stats["q_dynamics_mean"]))
                        and math.isfinite(float(stats["q_reward"]))
                        and (float(stats["q_dynamics_mean"]) <= 4.0)
                        and (float(stats["q_reward"]) <= 4.0)
                    ),
                }
                for (context_id, stats) in grouped_stats.items()
            },
        }
        source_usable = applicability_shadow["contexts"][int(self.active_context_id)][
            "joint_usable"
        ]
        history_usable = any(
            (
                value["joint_usable"]
                for (context_id, value) in applicability_shadow["contexts"].items()
                if int(context_id) != int(self.active_context_id)
            )
        )
        applicability_shadow["label"] = (
            "overlap"
            if source_usable and history_usable
            else (
                "current_usable"
                if source_usable
                else "history_usable" if history_usable else "undetermined"
            )
        )
        settling_reason = self._settling_reason(transition_id)
        winning_old = None
        old_score = 0.0
        new_increment = None
        if settling_reason is not None:
            self._reset_evidence()
            decision = S0Decision(
                "SETTLING",
                self.active_context_id,
                self.active_context_id,
                transition_id,
                True,
                settling_reason,
                dict(self._scores),
                0.0,
                log_probs,
                active_log_prob,
                behavior_context_id,
                raw_action="STAY",
                hold_reason=settling_reason,
                settle_until=int(self.settle_until),
            )
        else:
            for context_id, value in log_probs.items():
                if context_id == self.active_context_id:
                    self._scores[context_id] = 0.0
                    self._raw_scores[context_id] = 0.0
                elif self._candidate_held(context_id, transition_id):
                    self._scores[context_id] = 0.0
                    self._raw_scores[context_id] = max(
                        0.0,
                        self._raw_scores.get(context_id, 0.0) + value - active_log_prob,
                    )
                else:
                    self._scores[context_id] = max(
                        0.0, self._scores.get(context_id, 0.0) + value - active_log_prob
                    )
                    self._raw_scores[context_id] = self._scores[context_id]
            if self.new_log_prob is None:
                alternative = getattr(
                    active_slot.predictor, "new_alternative_log_prob", None
                )
                if not callable(alternative):
                    raise TypeError("active predictor has no NEW alternative scorer")
                active_prediction = prediction_cache.get(self.active_context_id)
                new_value = (
                    float(alternative(row))
                    if active_prediction is None
                    else float(
                        active_slot.predictor.new_alternative_log_prob_from_prediction(
                            *active_prediction
                        )
                    )
                )
            else:
                new_value = float(self.new_log_prob(active_slot.predictor, row))
            if not math.isfinite(new_value):
                raise FloatingPointError("non-finite NEW log probability")
            self.new_alt_call_count += 1
            new_increment = float(new_value - active_log_prob)
            self._new_score = max(0.0, self._new_score + new_increment)
            winning_old = max(
                (
                    cid
                    for cid in log_probs
                    if cid != self.active_context_id
                    and (not self._candidate_held(cid, transition_id))
                ),
                key=lambda cid: (self._scores.get(cid, 0.0), -cid),
                default=None,
            )
            old_score = 0.0 if winning_old is None else self._scores[winning_old]
            decision = None
            if self._new_confirmation is not None:
                if old_score >= self.threshold and old_score >= self._new_score:
                    state = self._new_confirmation
                    self._last_confirmation_block = {
                        "kind": "NEW",
                        "event": "new_confirmation_interrupted",
                        "nomination_step": int(state["nomination_step"]),
                        "block_first_step": (
                            None
                            if not state["current_block"]
                            else int(state["current_block"][0]["transition_id"])
                        ),
                        "block_last_step": transition_id,
                        "observed_rows": int(len(state["current_block"])),
                        "missing_rows": [],
                        "source_context_id": int(state["source_context_id"]),
                        "source_revision": int(state["source_revision"]),
                        "passed": False,
                        "admit_reason": "strong_history_interrupted_new_confirmation",
                        "reject_reason": ["strong_history_evidence"],
                    }
                    self._new_confirmation = None
                    self.new_confirmation_rejected_count += 1
                else:
                    confirmation = self._advance_new_confirmation(
                        row=row,
                        new_increment=float(new_increment),
                        new_log_prob=float(new_value),
                        source_stats=grouped_stats[self.active_context_id],
                    )
                    blocks = int(
                        self._new_confirmation.get("passed_blocks", 0)
                        if self._new_confirmation is not None
                        else 0
                    )
                    if confirmation == "rejected":
                        decision = S0Decision(
                            "NEW_REJECTED",
                            self.active_context_id,
                            self.active_context_id,
                            transition_id,
                            True,
                            "new_future_confirmation_failed",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="NEW",
                            confirmation_block_count=blocks,
                        )
                    elif confirmation == "passed" and (
                        self._new_score >= self.threshold
                        or bool(
                            self._new_confirmation
                            and self._new_confirmation.get(
                                "residual_path_passed", False
                            )
                        )
                    ):
                        if len(self._slots) >= self.max_contexts:
                            decision = S0Decision(
                                "CAPACITY_BLOCKED",
                                self.active_context_id,
                                None,
                                transition_id,
                                True,
                                "confirmed_new_but_capacity_full",
                                dict(self._scores),
                                self._new_score,
                                log_probs,
                                active_log_prob,
                                behavior_context_id,
                                raw_action="NEW",
                                confirmation_block_count=self.new_confirm_blocks,
                            )
                        else:
                            decision = S0Decision(
                                "NEW",
                                self.active_context_id,
                                None,
                                transition_id,
                                True,
                                "new_contiguous_future_confirmed",
                                dict(self._scores),
                                self._new_score,
                                log_probs,
                                active_log_prob,
                                behavior_context_id,
                                raw_action="NEW",
                                confirmation_block_count=self.new_confirm_blocks,
                            )
                    elif confirmation == "passed":
                        self._new_confirmation = None
                        self.new_confirmation_rejected_count += 1
                        decision = S0Decision(
                            "NEW_REJECTED",
                            self.active_context_id,
                            self.active_context_id,
                            transition_id,
                            True,
                            "new_threshold_not_sustained",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="NEW",
                            confirmation_block_count=blocks,
                        )
                    else:
                        decision = S0Decision(
                            "CONFIRMING_NEW",
                            self.active_context_id,
                            self.active_context_id,
                            transition_id,
                            True,
                            "new_future_confirmation_pending",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="NEW",
                            confirmation_block_count=blocks,
                        )
            if self._return_confirmation is not None:
                fixed_candidate = int(self._return_confirmation["candidate_context_id"])
                fixed_score = float(self._scores.get(fixed_candidate, 0.0))
                if self._new_score >= self.threshold and self._new_score > fixed_score:
                    state = self._return_confirmation
                    self._last_confirmation_block = {
                        "event": "confirmation_interrupted",
                        "nomination_step": int(state.get("nomination_step", -1)),
                        "block_first_step": None,
                        "block_last_step": transition_id,
                        "observed_rows": int(len(state.get("current_block", []))),
                        "missing_rows": [],
                        "candidate_context_id": fixed_candidate,
                        "source_context_id": int(self.active_context_id),
                        "candidate_revision": int(state["candidate_revision"]),
                        "source_revision": int(state["source_revision"]),
                        "passed": False,
                        "admit_reason": "strong_new_evidence_interrupted_confirmation",
                        "reject_reason": ["strong_new_evidence"],
                    }
                    self._return_confirmation = None
                elif fixed_candidate not in log_probs:
                    self._return_confirmation = None
                elif allow_uncommitted_behavior and fixed_candidate != behavior:
                    decision = S0Decision(
                        "RESELECT_REQUEST",
                        self.active_context_id,
                        fixed_candidate,
                        transition_id,
                        True,
                        "trial_candidate_changed",
                        dict(self._scores),
                        self._new_score,
                        log_probs,
                        active_log_prob,
                        behavior_context_id,
                        raw_action="RETURN",
                        candidate_context_id=fixed_candidate,
                    )
                else:
                    confirmation = self._advance_return_confirmation(
                        candidate_id=fixed_candidate,
                        row=row,
                        log_probs=log_probs,
                        prediction_cache=prediction_cache,
                    )
                    required_blocks = self._required_return_blocks(
                        fixed_candidate, transition_id
                    )
                    blocks = int(
                        self._return_confirmation.get("passed_blocks", 0)
                        if self._return_confirmation is not None
                        else 0
                    )
                    if confirmation == "rejected":
                        decision = S0Decision(
                            "TARGET_REJECTED",
                            self.active_context_id,
                            self.active_context_id,
                            transition_id,
                            True,
                            "historical_future_confirmation_failed",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="RETURN",
                            hold_reason="failed_candidate_hold",
                            candidate_context_id=fixed_candidate,
                            confirmation_block_count=blocks,
                        )
                    elif (
                        confirmation == "passed"
                        and fixed_score >= self.threshold
                        and (fixed_score >= self._new_score)
                    ):
                        decision = S0Decision(
                            "RETURN",
                            self.active_context_id,
                            fixed_candidate,
                            transition_id,
                            True,
                            "historical_future_confirmed",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="RETURN",
                            candidate_context_id=fixed_candidate,
                            confirmation_block_count=required_blocks,
                        )
                    elif confirmation != "invalidated":
                        decision = S0Decision(
                            "CONFIRMING",
                            self.active_context_id,
                            fixed_candidate,
                            transition_id,
                            True,
                            (
                                "historical_future_confirmed_awaiting_threshold"
                                if confirmation == "passed"
                                else "historical_future_confirmation_pending"
                            ),
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="RETURN",
                            candidate_context_id=fixed_candidate,
                            confirmation_block_count=blocks,
                        )
            if (
                decision is None
                and old_score >= self.threshold
                and (old_score >= self._new_score)
            ):
                raw_action = "RETURN"
                required_blocks = self._required_return_blocks(
                    int(winning_old), transition_id
                )
                if allow_uncommitted_behavior and int(winning_old) != behavior:
                    decision = S0Decision(
                        "RESELECT_REQUEST",
                        self.active_context_id,
                        winning_old,
                        transition_id,
                        True,
                        "trial_candidate_changed",
                        dict(self._scores),
                        self._new_score,
                        log_probs,
                        active_log_prob,
                        behavior_context_id,
                        raw_action="RETURN",
                        candidate_context_id=int(winning_old),
                    )
                else:
                    confirmation = (
                        self._advance_return_confirmation(
                            candidate_id=int(winning_old),
                            row=row,
                            log_probs=log_probs,
                            prediction_cache=prediction_cache,
                        )
                        if self.return_confirmation_enabled
                        else "passed"
                    )
                    if confirmation == "passed":
                        decision = S0Decision(
                            "RETURN",
                            self.active_context_id,
                            winning_old,
                            transition_id,
                            True,
                            "historical_future_confirmed",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="RETURN",
                            candidate_context_id=int(winning_old),
                            confirmation_block_count=int(
                                self._return_confirmation.get(
                                    "required_blocks", required_blocks
                                )
                                if self._return_confirmation is not None
                                else required_blocks
                            ),
                        )
                    elif confirmation == "rejected":
                        decision = S0Decision(
                            "TARGET_REJECTED",
                            self.active_context_id,
                            self.active_context_id,
                            transition_id,
                            True,
                            "historical_future_confirmation_failed",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="RETURN",
                            hold_reason="failed_candidate_hold",
                            candidate_context_id=int(winning_old),
                        )
                    else:
                        blocks = int(self._return_confirmation["passed_blocks"])
                        decision = S0Decision(
                            "CONFIRMING",
                            self.active_context_id,
                            winning_old,
                            transition_id,
                            True,
                            "historical_future_confirmation_pending",
                            dict(self._scores),
                            self._new_score,
                            log_probs,
                            active_log_prob,
                            behavior_context_id,
                            raw_action="RETURN",
                            candidate_context_id=int(winning_old),
                            confirmation_block_count=blocks,
                        )
            elif decision is None and self._new_score >= self.threshold:
                self._return_confirmation = None
                if len(self._slots) >= self.max_contexts:
                    decision = S0Decision(
                        "CAPACITY_BLOCKED",
                        self.active_context_id,
                        None,
                        transition_id,
                        True,
                        "new_evidence_but_capacity_full",
                        dict(self._scores),
                        self._new_score,
                        log_probs,
                        active_log_prob,
                        behavior_context_id,
                        raw_action="NEW",
                    )
                elif self.new_confirmation_enabled:
                    confirmation = self._advance_new_confirmation(
                        row=row,
                        new_increment=float(new_increment),
                        new_log_prob=float(new_value),
                        source_stats=grouped_stats[self.active_context_id],
                    )
                    if confirmation != "pending":
                        raise RuntimeError("NEW nomination consumed its trigger row")
                    decision = S0Decision(
                        "CONFIRMING_NEW",
                        self.active_context_id,
                        self.active_context_id,
                        transition_id,
                        True,
                        "new_threshold_crossed_nomination_only",
                        dict(self._scores),
                        self._new_score,
                        log_probs,
                        active_log_prob,
                        behavior_context_id,
                        raw_action="NEW",
                        confirmation_block_count=0,
                    )
                else:
                    decision = S0Decision(
                        "NEW",
                        self.active_context_id,
                        None,
                        transition_id,
                        True,
                        "new_cumulative_evidence",
                        dict(self._scores),
                        self._new_score,
                        log_probs,
                        active_log_prob,
                        behavior_context_id,
                        raw_action="NEW",
                    )
            elif decision is None and (
                self.trial_requests_enabled
                and winning_old is not None
                and (old_score >= self.trial_probe_fraction * self.threshold)
                and (
                    sum(
                        (
                            1
                            for cid in log_probs
                            if cid != self.active_context_id
                            and (not self._candidate_held(cid, transition_id))
                            and (self._scores.get(cid, 0.0) == old_score)
                        )
                    )
                    == 1
                )
            ):
                decision = S0Decision(
                    "TRIAL_REQUEST",
                    self.active_context_id,
                    winning_old,
                    transition_id,
                    True,
                    "historical_probe_threshold",
                    dict(self._scores),
                    self._new_score,
                    log_probs,
                    active_log_prob,
                    behavior_context_id,
                    raw_action="TRIAL_REQUEST",
                    candidate_context_id=int(winning_old),
                )
            elif decision is None:
                decision = S0Decision(
                    "STAY",
                    self.active_context_id,
                    self.active_context_id,
                    transition_id,
                    True,
                    "below_threshold",
                    dict(self._scores),
                    self._new_score,
                    log_probs,
                    active_log_prob,
                    behavior_context_id,
                    raw_action="STAY",
                )
        raw_scores_snapshot = dict(self._raw_scores)
        raw_new_score_snapshot = float(self._new_score)
        if decision.action in {"TARGET_REJECTED", "NEW_REJECTED"}:
            self._reset_evidence()
            winning_old = None
            old_score = 0.0
        if decision.action == "TRIAL_REQUEST":
            self.trial_request_count += 1
        old_score = 0.0 if winning_old is None else self._scores[winning_old]
        evidence_peak = max(float(self._new_score), float(old_score))
        probe_threshold = float(self.trial_probe_fraction * self.threshold)
        formal = decision.action in {"NEW", "RETURN"}
        self._pending_investigation_snapshot = {
            "investigating": bool(self.investigating),
            "source_context_id": self.investigation_source_context_id,
            "pending_rows": copy.deepcopy(self.pending_identity_rows),
            "investigation_count": int(self.investigation_count),
            "timeout_count": int(self.investigation_timeout_count),
            "scores": copy.deepcopy(self._scores),
            "raw_scores": copy.deepcopy(self._raw_scores),
            "new_score": float(self._new_score),
            "return_confirmation": copy.deepcopy(self._return_confirmation),
            "new_confirmation": copy.deepcopy(self._new_confirmation),
        }
        if not formal and evidence_peak >= probe_threshold:
            if not self.investigating:
                self.investigating = True
                self.investigation_source_context_id = self.active_context_id
                self.investigation_count += 1
            elif self.investigation_source_context_id != self.active_context_id:
                raise RuntimeError("investigation source Context changed unexpectedly")
        elif formal and self.investigating:
            if self.investigation_source_context_id != self.active_context_id:
                raise RuntimeError("formal decision has stale investigation source")
        if active_uncertain:
            self.active_uncertain_diagnostic_count += 1
            self.uncertain_but_scored_count += 1
        if self.investigating:
            self.investigation_steps += 1
        if formal:
            disposition = "FLUSH_TARGET"
        elif decision.action == "SETTLING":
            disposition = "TRAIN"
        elif decision.action in {"TARGET_REJECTED", "NEW_REJECTED"}:
            disposition = "FLUSH_SOURCE"
        elif decision.action == "CONFIRMING_NEW":
            disposition = "STAGE"
        elif decision.action == "RESELECT_REQUEST":
            disposition = "TRIAL_HELD"
        elif decision.action == "TRIAL_REQUEST":
            disposition = "TRIAL_HELD"
        elif self.investigating and evidence_peak <= 0.0:
            self.investigating = False
            self.investigation_source_context_id = None
            disposition = "FLUSH_SOURCE"
        elif self.investigating:
            disposition = "STAGE"
        elif evidence_peak <= 0.0 and self.pending_identity_rows:
            disposition = "FLUSH_SOURCE"
        else:
            disposition = "TRAIN"
        new_confirmation = self._new_confirmation
        block_view = self._last_confirmation_block or {}
        residual_authorized = bool(
            new_confirmation and new_confirmation.get("residual_path_passed", False)
        )
        confirmation_method = block_view.get(
            "confirmation_method",
            new_confirmation.get("confirmation_method") if new_confirmation else None,
        )
        if decision.action == "NEW" and residual_authorized:
            confirmation_method = "frozen_signed_residual"
        decision = replace(
            decision,
            active_disagreement=active_disagreement,
            context_disagreements=context_disagreements,
            active_uncertain_diagnostic=active_uncertain,
            evidence_peak=evidence_peak,
            investigation_active=bool(self.investigating),
            identity_row_disposition=disposition,
            comparison_performed=True,
            raw_scores=raw_scores_snapshot,
            new_increment=new_increment,
            grouped_stats=grouped_stats,
            predictor_revisions=predictor_revisions,
            applicability_shadow=applicability_shadow,
            new_confirmation_active=bool(
                self._new_confirmation is not None
                or decision.action == "CONFIRMING_NEW"
            ),
            new_transaction_id=(
                None
                if new_confirmation is None
                else str(new_confirmation["transaction_id"])
            ),
            new_nomination_step=(
                None
                if new_confirmation is None
                else int(new_confirmation["nomination_step"])
            ),
            new_source_revision=(
                None
                if new_confirmation is None
                else int(new_confirmation["source_revision"])
            ),
            confirmation_method=confirmation_method,
            natural_passed=(
                bool(new_confirmation.get("old_path_passed", False))
                if new_confirmation is not None
                else None
            ),
            effective_passed=True if decision.action == "NEW" else None,
            residual_reference_ready=bool(
                new_confirmation
                and new_confirmation.get("residual_reference", {}).get(
                    "reference_ready", False
                )
            ),
            residual_reference_count=int(
                new_confirmation.get("residual_reference", {}).get("reference_count", 0)
                if new_confirmation
                else 0
            ),
            residual_center=(
                _finite_or_none(
                    new_confirmation.get("residual_reference", {}).get("center")
                )
                if new_confirmation
                else None
            ),
            residual_tail_rank=_finite_or_none(block_view.get("residual_tail_rank")),
            residual_direction=int(block_view.get("direction", 0)),
            automatic_new_success=bool(
                decision.action == "NEW"
                and confirmation_method == "frozen_signed_residual"
            ),
            confirmation_block=(
                None
                if self._last_confirmation_block is None
                else {
                    **copy.deepcopy(self._last_confirmation_block),
                    "raw_scores": raw_scores_snapshot,
                    "admissible_scores": dict(decision.scores),
                    "new_score_before_reset": raw_new_score_snapshot,
                    "new_score_after_reset": (
                        0.0
                        if decision.action in {"TARGET_REJECTED", "NEW_REJECTED"}
                        else float(decision.new_score)
                    ),
                    "candidate_hold_until": (
                        self._candidate_holds.get(
                            (
                                int(self.active_context_id),
                                int(decision.candidate_context_id),
                            )
                        )
                        if decision.candidate_context_id is not None
                        else None
                    ),
                }
            ),
        )
        if (
            self.residual_confirmation_enabled
            and decision.identity_row_disposition == "TRAIN"
            and (
                decision.action
                not in {"NEW", "RETURN", "TRIAL_REQUEST", "CONFIRMING_NEW"}
            )
            and (self.active_context_id in grouped_stats)
        ):
            self._reference_candidates[transition_id] = {
                "context_id": int(self.active_context_id),
                "revision": int(predictor_revisions.get(self.active_context_id, 0)),
                "residual": float(
                    grouped_stats[self.active_context_id].get(
                        "reward_residual", float("nan")
                    )
                ),
            }
        self._pending = decision
        return decision

    def commit(self, decision: S0Decision, *, assigned_context_id: int) -> None:
        if decision is not self._pending:
            raise RuntimeError("decision is not the pending selector transaction")
        assigned = int(assigned_context_id)
        if decision.action == "NEW":
            if assigned in self._slots:
                raise ValueError("NEW Context id already exists in predictor registry")
            source = int(decision.source_context_id)
            pending_failed = {
                key
                for key in self._rejected_candidates_pending_new
                if int(key[0]) == source
            }
            for (hold_source, candidate), until in list(self._candidate_holds.items()):
                if hold_source != source:
                    continue
                if (
                    int(until) <= int(decision.transition_id)
                    and (hold_source, candidate) not in pending_failed
                ):
                    continue
                new_until = max(
                    int(until),
                    int(decision.transition_id)
                    + self.post_new_failed_candidate_hold_steps,
                )
                self._candidate_holds[assigned, int(candidate)] = new_until
                self._post_new_return_requirements[assigned, int(candidate)] = new_until
            for hold_source, candidate in pending_failed:
                if (assigned, int(candidate)) in self._candidate_holds:
                    continue
                new_until = (
                    int(decision.transition_id)
                    + self.post_new_failed_candidate_hold_steps
                )
                self._candidate_holds[assigned, int(candidate)] = new_until
                self._post_new_return_requirements[assigned, int(candidate)] = new_until
            self._rejected_candidates_pending_new.difference_update(pending_failed)
            self._slots[assigned] = self._new_slot(
                assigned, self.predictor_factory(assigned)
            )
            self._residual_reference_blocks.setdefault(assigned, [])
            self._residual_current_block.setdefault(assigned, [])
            self._scores[assigned] = 0.0
            self.active_context_id = assigned
        elif decision.action == "RETURN":
            if decision.target_context_id != assigned or assigned not in self._slots:
                raise ValueError("RETURN assignment does not match selector target")
            self.active_context_id = assigned
        elif decision.action in {
            "STAY",
            "CAPACITY_BLOCKED",
            "UNCERTAIN",
            "TRIAL_REQUEST",
            "SETTLING",
            "CONFIRMING",
            "TARGET_REJECTED",
            "RESELECT_REQUEST",
            "CONFIRMING_NEW",
            "NEW_REJECTED",
        }:
            if assigned != self.active_context_id:
                raise ValueError("STAY assignment changed Context")
        else:
            raise ValueError(f"cannot commit decision action {decision.action}")
        self._last_transition_id = int(decision.transition_id)
        self._decision_records.append(decision.to_record())
        if decision.action in {"NEW", "RETURN"}:
            self._reset_evidence()
            self.last_switch_transition_id = int(decision.transition_id)
            self.last_switch_action = str(decision.action)
            if decision.action == "NEW":
                self.settle_until = (
                    int(decision.transition_id) + self.settle_after_new_steps
                )
                self._newborn_context_id = assigned
                self._newborn_maintenance_baseline = int(
                    self._slots[assigned].maintenance_rounds
                )
            else:
                self.settle_until = (
                    int(decision.transition_id) + self.settle_after_return_steps
                )
                self._newborn_context_id = None
        if decision.action == "NEW":
            self.new_count += 1
        elif decision.action == "RETURN":
            self.return_count += 1
        self._pending_investigation_snapshot = None
        self._pending = None

    def observer_commit(self, decision: S0Decision) -> None:
        if decision is not self._pending:
            raise RuntimeError("decision is not the pending selector transaction")
        record = decision.to_record()
        record["observer_only"] = True
        self._decision_records.append(record)
        self._last_transition_id = int(decision.transition_id)
        if decision.action in {"NEW", "RETURN"}:
            self._scores = {context_id: 0.0 for context_id in self._slots}
            self._raw_scores = {context_id: 0.0 for context_id in self._slots}
            self._new_score = 0.0
            self._new_confirmation = None
        self._pending_investigation_snapshot = None
        self._pending = None

    def abort(self, decision: S0Decision, *, reason: str) -> None:
        if decision is not self._pending:
            raise RuntimeError("decision is not the pending selector transaction")
        record = decision.to_record()
        record["commit_error"] = str(reason)
        self._decision_records.append(record)
        snapshot = self._pending_investigation_snapshot
        if snapshot is not None:
            self.investigating = bool(snapshot["investigating"])
            self.investigation_source_context_id = snapshot["source_context_id"]
            self.pending_identity_rows = copy.deepcopy(snapshot["pending_rows"])
            self.investigation_count = int(snapshot["investigation_count"])
            self.investigation_timeout_count = int(snapshot["timeout_count"])
            self._scores = copy.deepcopy(snapshot["scores"])
            self._raw_scores = copy.deepcopy(
                snapshot.get("raw_scores", snapshot["scores"])
            )
            self._new_score = float(snapshot["new_score"])
            self._return_confirmation = copy.deepcopy(
                snapshot.get("return_confirmation")
            )
            self._new_confirmation = copy.deepcopy(snapshot.get("new_confirmation"))
        self._pending_investigation_snapshot = None
        self._reference_candidates.pop(int(decision.transition_id), None)
        self._pending = None

    def stage_identity_row(self, transition: Mapping[str, Any]) -> None:
        row = self._clean_row(transition)
        if not self.investigating:
            raise RuntimeError("cannot stage an identity row outside investigation")
        if self.investigation_source_context_id != self.active_context_id:
            raise RuntimeError("investigation source Context is not active")
        if self.pending_identity_rows and int(row["transition_id"]) <= int(
            self.pending_identity_rows[-1]["transition_id"]
        ):
            raise ValueError("pending identity transition_id must increase strictly")
        if len(self.pending_identity_rows) >= self.detection_recent_capacity:
            self.pending_identity_rows.append(row)
            self.pending_identity_rows = self.pending_identity_rows[
                -self.detection_recent_capacity :
            ]
            self.investigation_timeout_count += 1
            self.pending_identity_rows.clear()
            self.investigating = False
            self.investigation_source_context_id = None
            self._scores = {context_id: 0.0 for context_id in self._slots}
            self._new_score = 0.0
            return
        self.pending_identity_rows.append(row)
        self.pending_identity_rows_peak = max(
            self.pending_identity_rows_peak, len(self.pending_identity_rows)
        )

    def flush_pending_identity_rows(
        self,
        assigned_context_id: int,
        *,
        extra_rows: Iterable[Mapping[str, Any]] = (),
        train: bool = True,
    ) -> str:
        rows = [copy.deepcopy(row) for row in self.pending_identity_rows]
        rows.extend((self._clean_row(row) for row in extra_rows))
        unique: dict[int, dict[str, Any]] = {
            int(row["transition_id"]): row for row in rows
        }
        ordered = [unique[key] for key in sorted(unique)]
        if ordered:
            self.assign_rows_and_train(
                ordered, assigned_context_id=int(assigned_context_id), train=train
            )
        self.pending_identity_rows.clear()
        self.investigating = False
        self.investigation_source_context_id = None
        self._pending_investigation_snapshot = None
        return (
            "FLUSH_TARGET"
            if int(assigned_context_id) != self.active_context_id
            else "FLUSH_SOURCE"
        )

    def restore_trial_selector_state(self, state: Mapping[str, Any]) -> None:
        self.load_state_dict(copy.deepcopy(state))
        self.failed_trial_selector_rollback_count += 1

    def assign_and_train(
        self, transition: Mapping[str, Any], *, assigned_context_id: int
    ) -> None:
        self.assign_rows_and_train(
            (transition,), assigned_context_id=assigned_context_id
        )

    def assign_rows_and_train(
        self, transitions, *, assigned_context_id: int, train: bool = True
    ) -> None:
        rows = [self._clean_row(transition) for transition in transitions]
        if not rows:
            return
        ids = [int(row["transition_id"]) for row in rows]
        if ids != sorted(set(ids)):
            raise ValueError("assigned predictor rows must increase strictly")
        if ids[-1] != self._last_transition_id:
            raise ValueError("training rows do not end at committed transition")
        if int(assigned_context_id) != self.active_context_id:
            raise RuntimeError("predictor training owner is not active Context")
        slot = self._slots[self.active_context_id]
        for row in rows:
            self._record_reference_row(row)
            slot.training_view.add(row)
            if self.investigating:
                self.identity_updates_during_investigation += 1
            if train:
                self._train_if_due(slot)

    def warmup_assign_and_train(
        self, transition: Mapping[str, Any], *, behavior_context_id: int
    ) -> None:
        row = self._clean_row(transition)
        transition_id = int(row["transition_id"])
        if transition_id <= self._last_transition_id:
            raise ValueError("transition_id must increase strictly")
        if int(behavior_context_id) != self.active_context_id:
            raise RuntimeError("warmup owner is not active Context")
        self._last_transition_id = transition_id
        slot = self._slots[self.active_context_id]
        slot.training_view.add(row)
        self._train_if_due(slot)

    def context_predictor_state(self, context_id: int) -> dict[str, Any]:
        slot = self._slots[int(context_id)]
        state_fn = getattr(slot.predictor, "state_dict", None)
        if not callable(state_fn):
            raise TypeError("Context predictor has no state_dict")
        return {
            "context_id": int(context_id),
            "predictor": copy.deepcopy(state_fn()),
            "fitted_rows": int(slot.fitted_rows),
            "initial_fit_complete": bool(slot.initial_fit_complete),
            "optimizer_updates": int(slot.optimizer_updates),
            "maintenance_rounds": int(slot.maintenance_rounds),
            "training_view": slot.training_view.state_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("cannot checkpoint an uncommitted selector transaction")
        return {
            "schema": self.schema,
            "active_context_id": self.active_context_id,
            "threshold": self.threshold,
            "scores": copy.deepcopy(self._scores),
            "raw_scores": copy.deepcopy(self._raw_scores),
            "new_score": self._new_score,
            "last_transition_id": self._last_transition_id,
            "slots": {
                int(context_id): {
                    "training_view": slot.training_view.state_dict(),
                    "fitted_rows": int(slot.fitted_rows),
                    "initial_fit_complete": bool(slot.initial_fit_complete),
                    "optimizer_updates": int(slot.optimizer_updates),
                    "maintenance_rounds": int(slot.maintenance_rounds),
                    "predictor": copy.deepcopy(slot.predictor.state_dict()),
                }
                for (context_id, slot) in self._slots.items()
            },
            "decision_records": copy.deepcopy(self._decision_records),
            "train_wall_ms": self.train_wall_ms,
            "detection_recent_capacity": self.detection_recent_capacity,
            "detection_reservoir_capacity": self.detection_reservoir_capacity,
            "detection_seed": self.detection_seed,
            "trial_requests_enabled": self.trial_requests_enabled,
            "trial_probe_fraction": self.trial_probe_fraction,
            "settle_after_new_steps": self.settle_after_new_steps,
            "settle_after_return_steps": self.settle_after_return_steps,
            "new_min_maintenance_rounds": self.new_min_maintenance_rounds,
            "return_confirmation_enabled": self.return_confirmation_enabled,
            "return_confirm_block_size": self.return_confirm_block_size,
            "return_confirm_blocks": self.return_confirm_blocks,
            "failed_candidate_hold_steps": self.failed_candidate_hold_steps,
            "post_new_failed_candidate_hold_steps": self.post_new_failed_candidate_hold_steps,
            "post_new_return_confirm_blocks": self.post_new_return_confirm_blocks,
            "new_confirmation_enabled": self.new_confirmation_enabled,
            "new_confirm_block_size": self.new_confirm_block_size,
            "new_confirm_blocks": self.new_confirm_blocks,
            "residual_confirmation_enabled": self.residual_confirmation_enabled,
            "residual_block_size": self.residual_block_size,
            "residual_reference_min_blocks": self.residual_reference_min_blocks,
            "residual_reference_max_blocks": self.residual_reference_max_blocks,
            "residual_tail_rank_alpha": self.residual_tail_rank_alpha,
            "last_switch_transition_id": self.last_switch_transition_id,
            "last_switch_action": self.last_switch_action,
            "settle_until": self.settle_until,
            "newborn_context_id": self._newborn_context_id,
            "newborn_maintenance_baseline": self._newborn_maintenance_baseline,
            "candidate_holds": [
                {"source": key[0], "candidate": key[1], "until": value}
                for (key, value) in sorted(self._candidate_holds.items())
            ],
            "post_new_return_requirements": [
                {"source": key[0], "candidate": key[1], "until": value}
                for (key, value) in sorted(self._post_new_return_requirements.items())
            ],
            "rejected_candidates_pending_new": [
                {"source": key[0], "candidate": key[1]}
                for key in sorted(self._rejected_candidates_pending_new)
            ],
            "return_confirmation": copy.deepcopy(self._return_confirmation),
            "new_confirmation": copy.deepcopy(self._new_confirmation),
            "residual_reference_blocks": copy.deepcopy(self._residual_reference_blocks),
            "residual_current_block": copy.deepcopy(self._residual_current_block),
            "residual_frozen": copy.deepcopy(self._residual_frozen),
            "reference_candidates": copy.deepcopy(self._reference_candidates),
            "fit_records": copy.deepcopy(self._fit_records),
            "investigating": bool(self.investigating),
            "investigation_source_context_id": self.investigation_source_context_id,
            "pending_identity_rows": copy.deepcopy(self.pending_identity_rows),
            "investigation_count": int(self.investigation_count),
            "investigation_timeout_count": int(self.investigation_timeout_count),
            "ready_steps": int(self.ready_steps),
            "active_uncertain_diagnostic_count": int(
                self.active_uncertain_diagnostic_count
            ),
            "uncertain_but_scored_count": int(self.uncertain_but_scored_count),
            "history_logprob_call_count": int(self.history_logprob_call_count),
            "new_alt_call_count": int(self.new_alt_call_count),
            "investigation_steps": int(self.investigation_steps),
            "pending_identity_rows_peak": int(self.pending_identity_rows_peak),
            "identity_updates_during_investigation": int(
                self.identity_updates_during_investigation
            ),
            "trial_request_count": int(self.trial_request_count),
            "new_count": int(self.new_count),
            "return_count": int(self.return_count),
            "failed_trial_selector_rollback_count": int(
                self.failed_trial_selector_rollback_count
            ),
            "return_confirmation_started_count": int(
                self.return_confirmation_started_count
            ),
            "return_confirmation_passed_count": int(
                self.return_confirmation_passed_count
            ),
            "return_confirmation_rejected_count": int(
                self.return_confirmation_rejected_count
            ),
            "new_confirmation_started_count": int(self.new_confirmation_started_count),
            "new_confirmation_passed_count": int(self.new_confirmation_passed_count),
            "new_confirmation_rejected_count": int(
                self.new_confirmation_rejected_count
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.schema:
            raise ValueError("invalid S0 selector state schema")
        self.active_context_id = int(state["active_context_id"])
        self.threshold = float(state["threshold"])
        if int(state["detection_recent_capacity"]) != self.detection_recent_capacity:
            raise ValueError("selector recent detection capacity changed")
        if (
            int(state["detection_reservoir_capacity"])
            != self.detection_reservoir_capacity
        ):
            raise ValueError("selector reservoir detection capacity changed")
        if int(state["detection_seed"]) != self.detection_seed:
            raise ValueError("selector detection RNG seed changed")
        if bool(state["trial_requests_enabled"]) != self.trial_requests_enabled:
            raise ValueError("selector trial request setting changed")
        if float(state["trial_probe_fraction"]) != self.trial_probe_fraction:
            raise ValueError("selector trial probe fraction changed")
        for key in (
            "settle_after_new_steps",
            "settle_after_return_steps",
            "new_min_maintenance_rounds",
            "return_confirm_block_size",
            "return_confirm_blocks",
            "failed_candidate_hold_steps",
            "post_new_failed_candidate_hold_steps",
            "post_new_return_confirm_blocks",
            "new_confirm_block_size",
            "new_confirm_blocks",
        ):
            if int(state[key]) != int(getattr(self, key)):
                raise ValueError(f"selector identity-stability setting changed: {key}")
        for key in (
            "residual_confirmation_enabled",
            "residual_block_size",
            "residual_reference_min_blocks",
            "residual_reference_max_blocks",
            "residual_tail_rank_alpha",
        ):
            expected = getattr(self, key)
            observed = state.get(key, expected)
            if isinstance(expected, float):
                if float(observed) != float(expected):
                    raise ValueError(f"selector residual setting changed: {key}")
            elif isinstance(expected, bool):
                if bool(observed) != bool(expected):
                    raise ValueError(f"selector residual setting changed: {key}")
            elif int(observed) != int(expected):
                raise ValueError(f"selector residual setting changed: {key}")
        if (
            bool(state["return_confirmation_enabled"])
            != self.return_confirmation_enabled
        ):
            raise ValueError("selector return confirmation setting changed")
        if bool(state["new_confirmation_enabled"]) != self.new_confirmation_enabled:
            raise ValueError("selector NEW confirmation setting changed")
        self._scores = copy.deepcopy(state["scores"])
        self._raw_scores = copy.deepcopy(state.get("raw_scores", state["scores"]))
        self._new_score = float(state["new_score"])
        self._last_transition_id = int(state["last_transition_id"])
        self._slots = {}
        for context_id, payload in state["slots"].items():
            context_id = int(context_id)
            predictor = self.predictor_factory(context_id)
            predictor.load_state_dict(copy.deepcopy(payload["predictor"]))
            slot = self._new_slot(context_id, predictor)
            slot.training_view.load_state_dict(payload["training_view"])
            slot.fitted_rows = int(payload["fitted_rows"])
            slot.initial_fit_complete = bool(payload["initial_fit_complete"])
            slot.optimizer_updates = int(payload["optimizer_updates"])
            slot.maintenance_rounds = int(payload.get("maintenance_rounds", 0))
            self._slots[context_id] = slot
        self._decision_records = copy.deepcopy(state["decision_records"])
        self.train_wall_ms = float(state["train_wall_ms"])
        self.investigating = bool(state.get("investigating", False))
        self.investigation_source_context_id = state.get(
            "investigation_source_context_id"
        )
        self.pending_identity_rows = copy.deepcopy(
            state.get("pending_identity_rows", [])
        )
        self.investigation_count = int(state.get("investigation_count", 0))
        self.investigation_timeout_count = int(
            state.get("investigation_timeout_count", 0)
        )
        self.ready_steps = int(state.get("ready_steps", 0))
        self.active_uncertain_diagnostic_count = int(
            state.get("active_uncertain_diagnostic_count", 0)
        )
        self.uncertain_but_scored_count = int(
            state.get("uncertain_but_scored_count", 0)
        )
        self.history_logprob_call_count = int(
            state.get("history_logprob_call_count", 0)
        )
        self.new_alt_call_count = int(state.get("new_alt_call_count", 0))
        self.investigation_steps = int(state.get("investigation_steps", 0))
        self.pending_identity_rows_peak = int(
            state.get("pending_identity_rows_peak", 0)
        )
        self.identity_updates_during_investigation = int(
            state.get("identity_updates_during_investigation", 0)
        )
        self.trial_request_count = int(state.get("trial_request_count", 0))
        self.new_count = int(state.get("new_count", 0))
        self.return_count = int(state.get("return_count", 0))
        self.failed_trial_selector_rollback_count = int(
            state.get("failed_trial_selector_rollback_count", 0)
        )
        self.last_switch_transition_id = state.get("last_switch_transition_id")
        self.last_switch_action = state.get("last_switch_action")
        self.settle_until = int(state.get("settle_until", -1))
        self._newborn_context_id = state.get("newborn_context_id")
        self._newborn_maintenance_baseline = int(
            state.get("newborn_maintenance_baseline", 0)
        )
        self._candidate_holds = {
            (int(item["source"]), int(item["candidate"])): int(item["until"])
            for item in state.get("candidate_holds", [])
        }
        self._post_new_return_requirements = {
            (int(item["source"]), int(item["candidate"])): int(item["until"])
            for item in state.get("post_new_return_requirements", [])
        }
        self._rejected_candidates_pending_new = {
            (int(item["source"]), int(item["candidate"]))
            for item in state.get("rejected_candidates_pending_new", [])
        }
        self._return_confirmation = copy.deepcopy(state.get("return_confirmation"))
        self._new_confirmation = copy.deepcopy(state.get("new_confirmation"))
        self._residual_reference_blocks = {
            int(key): copy.deepcopy(value)
            for (key, value) in state.get("residual_reference_blocks", {}).items()
        }
        self._residual_current_block = {
            int(key): copy.deepcopy(value)
            for (key, value) in state.get("residual_current_block", {}).items()
        }
        self._residual_frozen = copy.deepcopy(state.get("residual_frozen", {}))
        self._reference_candidates = {
            int(key): copy.deepcopy(value)
            for (key, value) in state.get("reference_candidates", {}).items()
        }
        self._fit_records = copy.deepcopy(state.get("fit_records", []))
        self.return_confirmation_started_count = int(
            state.get("return_confirmation_started_count", 0)
        )
        self.return_confirmation_passed_count = int(
            state.get("return_confirmation_passed_count", 0)
        )
        self.return_confirmation_rejected_count = int(
            state.get("return_confirmation_rejected_count", 0)
        )
        self.new_confirmation_started_count = int(
            state.get("new_confirmation_started_count", 0)
        )
        self.new_confirmation_passed_count = int(
            state.get("new_confirmation_passed_count", 0)
        )
        self.new_confirmation_rejected_count = int(
            state.get("new_confirmation_rejected_count", 0)
        )
        if (
            self.investigating
            and self.investigation_source_context_id not in self._slots
        ):
            raise ValueError(
                "investigation source Context is missing from selector state"
            )
        self._pending = None
        self._pending_investigation_snapshot = None

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._decision_records))

    def fit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._fit_records))

    def audit_metrics(self) -> dict[str, int | float]:
        return {
            "ready_steps": int(self.ready_steps),
            "active_uncertain_diagnostic_count": int(
                self.active_uncertain_diagnostic_count
            ),
            "uncertain_but_scored_count": int(self.uncertain_but_scored_count),
            "history_logprob_call_count": int(self.history_logprob_call_count),
            "new_alt_call_count": int(self.new_alt_call_count),
            "investigation_count": int(self.investigation_count),
            "investigation_steps": int(self.investigation_steps),
            "investigation_timeout_count": int(self.investigation_timeout_count),
            "pending_identity_rows_peak": int(self.pending_identity_rows_peak),
            "identity_updates_during_investigation": int(
                self.identity_updates_during_investigation
            ),
            "trial_request_count": int(self.trial_request_count),
            "new_count": int(self.new_count),
            "return_count": int(self.return_count),
            "failed_trial_selector_rollback_count": int(
                self.failed_trial_selector_rollback_count
            ),
            "return_confirmation_started_count": int(
                self.return_confirmation_started_count
            ),
            "return_confirmation_passed_count": int(
                self.return_confirmation_passed_count
            ),
            "return_confirmation_rejected_count": int(
                self.return_confirmation_rejected_count
            ),
            "new_confirmation_started_count": int(self.new_confirmation_started_count),
            "new_confirmation_passed_count": int(self.new_confirmation_passed_count),
            "new_confirmation_rejected_count": int(
                self.new_confirmation_rejected_count
            ),
            "residual_reference_blocks_active": int(
                len(self._residual_reference_blocks.get(self.active_context_id, []))
            ),
            "residual_reference_ready": bool(
                self._residual_reference_ready(self.active_context_id)
            ),
            "residual_reference_updates": int(
                sum((len(value) for value in self._residual_reference_blocks.values()))
            ),
            "pnn_fit_count": int(len(self._fit_records)),
        }
