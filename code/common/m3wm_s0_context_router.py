from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from common.m3wm_s0_context_selector import S0Decision, S0PNNEvidenceContextSelector
from common.m3wm_s0_trial_controller import TrialController, TrialResolution


@dataclass(frozen=True)
class S0RouteReceipt:
    decision: S0Decision
    behavior_context_id: int
    assigned_context_id: int | None
    bank_switch_receipt: Any = None
    trial_requested: bool = False
    trial_started: bool = False
    trial_checkpoint_due: bool = False
    trial_resolution: TrialResolution | None = None
    identity_row_disposition: str = "TRAIN"
    pending_identity_rows: int = 0


class S0ContextRouter:

    def __init__(
        self,
        *,
        selector: S0PNNEvidenceContextSelector,
        bank: Any,
        trial_controller: TrialController | None = None,
        before_switch: Callable[[S0Decision], None] | None = None,
        observer_only: bool = False,
    ) -> None:
        self.selector = selector
        self.bank = bank
        self.trial_controller = trial_controller
        self.before_switch = before_switch
        self.observer_only = bool(observer_only)

    def route(
        self, transition: Mapping[str, Any], *, control_payload: Any = None
    ) -> S0RouteReceipt:
        trial = self.trial_controller
        trial_active = bool(trial is not None and trial.active)
        behavior = int(
            trial.behavior_context_id if trial_active else self.bank.active_context_id
        )
        decision = self.selector.decide(
            transition,
            behavior_context_id=behavior,
            allow_uncommitted_behavior=trial_active,
        )
        if self.observer_only and decision.action != "WARMUP":
            self.selector.observer_commit(decision)
            disposition = str(decision.identity_row_disposition)
            if disposition in {"STAGE", "TRIAL_HELD"}:
                self.selector.stage_identity_row(transition)
            elif disposition in {"FLUSH_SOURCE", "FLUSH_TARGET"}:
                self.selector.flush_pending_identity_rows(
                    self.bank.active_context_id, extra_rows=(transition,)
                )
            elif decision.action != "WARMUP":
                self.selector.assign_and_train(
                    transition, assigned_context_id=self.bank.active_context_id
                )
            return S0RouteReceipt(
                decision,
                behavior,
                self.bank.active_context_id,
                identity_row_disposition=disposition,
                pending_identity_rows=len(self.selector.pending_identity_rows),
            )
        if decision.action == "WARMUP":
            if trial_active:
                raise RuntimeError("an unready selector cannot own an active trial")
            self.selector.warmup_assign_and_train(
                transition, behavior_context_id=behavior
            )
            return S0RouteReceipt(
                decision,
                behavior,
                behavior,
                identity_row_disposition="WARMUP",
                pending_identity_rows=len(self.selector.pending_identity_rows),
            )
        checkpoint_due = False
        trial_limit_reason = None
        if trial_active:
            source_id = int(trial.committed_context_id)
            candidate_id = int(trial.behavior_context_id)
            checkpoint_due, trial_limit_reason = trial.record_transition(
                transition,
                control_payload=control_payload,
                identity_payload={
                    "transition_id": int(transition["transition_id"]),
                    "source_context_id": source_id,
                    "candidate_context_id": candidate_id,
                    "source_log_prob": decision.log_probs.get(source_id),
                    "candidate_log_prob": decision.log_probs.get(candidate_id),
                    "scores": dict(decision.scores),
                    "new_score": float(decision.new_score),
                },
            )
        bank_receipt = None
        try:
            if (
                decision.action == "CONFIRMING_NEW"
                and decision.new_transaction_id is not None
                and hasattr(self.bank, "capture_pending_init_capsule")
            ):
                self.bank.capture_pending_init_capsule(
                    source_context_id=int(decision.source_context_id),
                    nomination_transition_id=int(decision.new_nomination_step),
                    transaction_id=str(decision.new_transaction_id),
                    source_revision=int(decision.new_source_revision),
                )
            if decision.action == "NEW":
                self.sync_active_predictor_state()
                if self.before_switch is not None:
                    self.before_switch(decision)
                if decision.new_transaction_id is not None and hasattr(
                    self.bank, "capture_pending_init_capsule"
                ):
                    active, bank_receipt = self.bank.create_and_activate(
                        reason="pnn_evidence_s0_new",
                        init_transaction_id=str(decision.new_transaction_id),
                        source_revision=int(decision.new_source_revision),
                    )
                else:
                    active, bank_receipt = self.bank.create_and_activate(
                        reason="pnn_evidence_s0_new"
                    )
                assigned = int(active.context_id)
            elif decision.action == "RETURN":
                self.sync_active_predictor_state()
                if self.before_switch is not None:
                    self.before_switch(decision)
                assigned = int(decision.target_context_id)
                bank_receipt = self.bank.activate(
                    assigned, reason="pnn_evidence_s0_return"
                )
            elif decision.action in {
                "STAY",
                "UNCERTAIN",
                "CAPACITY_BLOCKED",
                "TRIAL_REQUEST",
                "SETTLING",
                "CONFIRMING",
                "TARGET_REJECTED",
                "RESELECT_REQUEST",
                "CONFIRMING_NEW",
                "NEW_REJECTED",
            }:
                assigned = int(self.bank.active_context_id)
            else:
                raise RuntimeError(f"unsupported S0 decision {decision.action}")
            self.selector.commit(decision, assigned_context_id=assigned)
            confirmation_event = str(
                (decision.confirmation_block or {}).get("event", "")
            )
            if (
                decision.action == "CONFIRMING"
                and confirmation_event == "new_confirmation_interrupted"
                and hasattr(self.bank, "discard_pending_init_capsule")
            ):
                self.bank.discard_pending_init_capsule(
                    reason="new_confirmation_interrupted_by_history"
                )
            if decision.action in {"NEW_REJECTED", "RETURN"} and hasattr(
                self.bank, "discard_pending_init_capsule"
            ):
                self.bank.discard_pending_init_capsule(
                    reason=f"selector_{decision.action.lower()}"
                )
        except Exception as exc:
            if hasattr(self.bank, "discard_pending_init_capsule"):
                self.bank.discard_pending_init_capsule(
                    reason=f"route_abort_{type(exc).__name__}"
                )
            self.selector.abort(decision, reason=f"{type(exc).__name__}: {exc}")
            raise
        resolution = None
        if trial_active:
            if decision.action == "CONFIRMING_NEW":
                resolution = trial.resolve(
                    end_step=int(transition["transition_id"]),
                    outcome="ABORTED",
                    reason="new_confirmation_nominated",
                )
                self.selector.stage_identity_row(transition)
                return S0RouteReceipt(
                    decision,
                    behavior,
                    int(self.bank.active_context_id),
                    None,
                    trial_resolution=resolution,
                    identity_row_disposition="STAGE",
                    pending_identity_rows=len(self.selector.pending_identity_rows),
                )
            if decision.action == "RESELECT_REQUEST":
                resolution = trial.resolve(
                    end_step=int(transition["transition_id"]),
                    outcome="ABORTED",
                    reason="trial_candidate_changed",
                )
                self.selector.restore_trial_selector_state(resolution.selector_state)
                self.selector.register_failed_candidate(
                    source_context_id=resolution.source_context_id,
                    candidate_context_id=resolution.candidate_context_id,
                    end_transition_id=resolution.end_step,
                )
                return S0RouteReceipt(
                    decision,
                    behavior,
                    None,
                    bank_receipt,
                    trial_resolution=resolution,
                    identity_row_disposition="DISCARD",
                    pending_identity_rows=len(self.selector.pending_identity_rows),
                )
            if decision.action == "TARGET_REJECTED":
                expected = int((resolution_candidate := trial.behavior_context_id))
                observed = decision.candidate_context_id
                reason = "historical_future_confirmation_failed"
                if observed is None or int(observed) != expected:
                    reason = "target_rejected_trial_candidate_mismatch"
                resolution = trial.resolve(
                    end_step=int(transition["transition_id"]),
                    outcome="ABORTED",
                    reason=reason,
                )
                self.selector.restore_trial_selector_state(resolution.selector_state)
                self.selector.register_failed_candidate(
                    source_context_id=resolution.source_context_id,
                    candidate_context_id=resolution.candidate_context_id,
                    end_transition_id=resolution.end_step,
                )
                return S0RouteReceipt(
                    decision,
                    behavior,
                    int(self.bank.active_context_id),
                    None,
                    trial_resolution=resolution,
                    identity_row_disposition="DISCARD",
                    pending_identity_rows=len(self.selector.pending_identity_rows),
                )
            if decision.action in {"NEW", "RETURN"}:
                resolution = trial.resolve(
                    end_step=int(transition["transition_id"]),
                    outcome=decision.action,
                    assigned_context_id=assigned,
                    reason=f"selector_{decision.action.lower()}",
                )
                self.selector.flush_pending_identity_rows(
                    assigned, extra_rows=resolution.rows
                )
                return S0RouteReceipt(
                    decision,
                    behavior,
                    assigned,
                    bank_receipt,
                    trial_resolution=resolution,
                    identity_row_disposition=(
                        "DISCARD"
                        if resolution.outcome in {"INCONCLUSIVE", "ABORTED"}
                        else "FLUSH_TARGET"
                    ),
                    pending_identity_rows=len(self.selector.pending_identity_rows),
                )
            if (
                checkpoint_due
                and trial_limit_reason is None
                and (not bool(transition.get("terminated", False)))
                and trial.evidence_dissipated_at_checkpoint()
            ):
                resolution = trial.resolve(
                    end_step=int(transition["transition_id"]),
                    outcome="ABORTED",
                    reason="trial_evidence_dissipated",
                )
                self.selector.restore_trial_selector_state(resolution.selector_state)
                self.selector.register_failed_candidate(
                    source_context_id=resolution.source_context_id,
                    candidate_context_id=resolution.candidate_context_id,
                    end_transition_id=resolution.end_step,
                )
                return S0RouteReceipt(
                    decision,
                    behavior,
                    int(self.bank.active_context_id),
                    None,
                    trial_checkpoint_due=True,
                    trial_resolution=resolution,
                    identity_row_disposition="DISCARD",
                    pending_identity_rows=len(self.selector.pending_identity_rows),
                )
            if trial_limit_reason is not None or bool(
                transition.get("terminated", False)
            ):
                outcome = (
                    "ABORTED"
                    if bool(transition.get("terminated", False))
                    else "INCONCLUSIVE"
                )
                reason = (
                    "environment_terminated"
                    if outcome == "ABORTED"
                    else str(trial_limit_reason)
                )
                resolution = trial.resolve(
                    end_step=int(transition["transition_id"]),
                    outcome=outcome,
                    reason=reason,
                )
            if resolution is not None:
                if resolution.outcome in {"INCONCLUSIVE", "ABORTED"}:
                    self.selector.restore_trial_selector_state(
                        resolution.selector_state
                    )
                    self.selector.register_failed_candidate(
                        source_context_id=resolution.source_context_id,
                        candidate_context_id=resolution.candidate_context_id,
                        end_transition_id=resolution.end_step,
                    )
                elif resolution.outcome in {"NEW", "RETURN"}:
                    self.selector.flush_pending_identity_rows(
                        assigned, extra_rows=resolution.rows
                    )
            return S0RouteReceipt(
                decision,
                behavior,
                None,
                bank_receipt,
                trial_checkpoint_due=checkpoint_due,
                trial_resolution=resolution,
                identity_row_disposition="TRIAL_HELD",
                pending_identity_rows=len(self.selector.pending_identity_rows),
            )
        disposition = str(decision.identity_row_disposition)
        if disposition == "STAGE" or disposition == "TRIAL_HELD":
            self.selector.stage_identity_row(transition)
        elif disposition in {"FLUSH_SOURCE", "FLUSH_TARGET"}:
            self.selector.flush_pending_identity_rows(
                assigned, extra_rows=(transition,)
            )
        else:
            self.selector.assign_and_train(transition, assigned_context_id=assigned)
        return S0RouteReceipt(
            decision,
            behavior,
            assigned,
            bank_receipt,
            trial_requested=decision.action == "TRIAL_REQUEST" and trial is not None,
            identity_row_disposition=disposition,
            pending_identity_rows=len(self.selector.pending_identity_rows),
        )

    def start_trial(self, decision: S0Decision, *, env_steps_seen: int) -> bool:
        trial = self.trial_controller
        if trial is None or decision.action != "TRIAL_REQUEST":
            return False
        target = int(decision.target_context_id)
        selector_ok, selector_reason = self.selector.can_start_trial(
            target, int(env_steps_seen)
        )
        trial_ok, trial_reason, _ = trial.can_start(env_steps_seen=int(env_steps_seen))
        if not selector_ok or not trial_ok:
            trial.record_rejection(
                env_steps_seen=int(env_steps_seen),
                source_context_id=int(self.bank.active_context_id),
                candidate_context_id=target,
                reason=selector_reason if not selector_ok else trial_reason,
            )
            return False
        self.sync_active_predictor_state()
        try:
            working, source_hash, candidate_hash = self.bank.trial_working_copy(target)
        except Exception:
            working, source_hash, candidate_hash = (None, "", "")
        predictor_versions = {
            context_id: self.selector.context_predictor_state(context_id)
            for context_id in self.selector._slots
        }
        lease = trial.start(
            env_steps_seen=int(env_steps_seen),
            source_context_id=int(self.bank.active_context_id),
            candidate_context_id=target,
            source_bundle_hash=source_hash,
            candidate_bundle_hash=candidate_hash,
            selector_state=self.selector.state_dict(),
            predictor_versions=predictor_versions,
            working_bundle=working,
        )
        return lease is not None

    def _sync_predictor_state(self, context_id: int) -> None:
        self.bank.bundle(context_id).predictor_state = (
            self.selector.context_predictor_state(context_id)
        )

    def sync_active_predictor_state(self) -> None:
        self._sync_predictor_state(int(self.bank.active_context_id))
