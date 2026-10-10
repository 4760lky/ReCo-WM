from __future__ import annotations
import copy
from common.m3wm_s0_trial_controller import TrialController


def _start(controller: TrialController, step: int = 6400):
    return controller.start(
        env_steps_seen=step,
        source_context_id=1,
        candidate_context_id=0,
        source_bundle_hash="source-hash",
        candidate_bundle_hash="candidate-hash",
        selector_state={"scores": {0: 5.0}},
        predictor_versions={0: 3, 1: 4},
        working_bundle=object(),
    )


def test_trial_budget_reserves_only_current_two_percent_allowance():
    controller = TrialController()
    lease = _start(controller, step=1600)
    assert lease is not None
    assert lease.reserved_steps == 32
    due = False
    reason = None
    for index in range(32):
        due, reason = controller.record_transition({"transition_id": index})
    assert due is True
    assert reason == "reserved_budget_exhausted"
    resolution = controller.resolve(
        end_step=1632, outcome="INCONCLUSIVE", reason=reason
    )
    assert resolution.used_steps == 32
    assert controller.trial_steps_total == 32
    assert controller.discarded_training_rows == 32
    assert controller.can_start(env_steps_seen=2655)[1] == "trial_cooldown"
    assert controller.can_start(env_steps_seen=2656)[1] != "trial_cooldown"


def test_trial_success_keeps_rows_for_one_committed_assignment():
    controller = TrialController()
    lease = _start(controller)
    assert lease is not None and lease.reserved_steps == 128
    for index in range(17):
        due, reason = controller.record_transition(
            {"transition_id": index, "reward": -float(index)},
            control_payload={"td": index},
        )
        assert reason is None
    resolution = controller.resolve(
        end_step=6417, outcome="RETURN", assigned_context_id=0, reason="selector_return"
    )
    assert resolution.assigned_context_id == 0
    assert len(resolution.rows) == len(resolution.control_payloads) == 17
    assert controller.discarded_training_rows == 0
    assert controller.active is False


def test_trial_state_persists_budget_and_unassigned_rows_without_working_bundle():
    controller = TrialController()
    _start(controller)
    controller.record_transition({"transition_id": 7}, control_payload="payload")
    state = copy.deepcopy(controller.state_dict())
    restored = TrialController()
    restored.load_state_dict(state)
    assert restored.active is True
    assert restored.working_bundle is None
    assert restored.trial_steps_total == 1
    assert restored.state_dict()["lease"]["rows"][0]["transition_id"] == 7


def test_disabled_or_missing_bundle_never_starts_trial():
    disabled = TrialController(enabled=False)
    assert _start(disabled) is None
    enabled = TrialController()
    assert (
        enabled.start(
            env_steps_seen=6400,
            source_context_id=1,
            candidate_context_id=0,
            source_bundle_hash="source",
            candidate_bundle_hash="candidate",
            selector_state={},
            predictor_versions={},
            working_bundle=None,
        )
        is None
    )


def _identity_payload(step: int, *, candidate_logp: float, score: float = 0.0):
    return {
        "transition_id": step,
        "source_context_id": 1,
        "candidate_context_id": 0,
        "source_log_prob": 0.0,
        "candidate_log_prob": candidate_logp,
        "scores": {0: score, 1: 0.0},
        "new_score": 0.0,
    }


def test_trial_checkpoint_marks_nonpositive_llr_with_cleared_evidence():
    controller = TrialController()
    _start(controller)
    for step in range(32):
        due, reason = controller.record_transition(
            {"transition_id": step, "reward": -0.25},
            identity_payload=_identity_payload(step, candidate_logp=-2.0),
        )
    assert due is True and reason is None
    assert controller.evidence_dissipated_at_checkpoint() is True
    checkpoint = controller.records()[-1]
    assert checkpoint["event"] == "TRIAL_CHECKPOINT"
    assert checkpoint["trial_mean_llr"] == -2.0
    assert checkpoint["evidence_cleared"] is True
    resolution = controller.resolve(
        end_step=6432, outcome="ABORTED", reason="trial_evidence_dissipated"
    )
    assert resolution.checkpoint_count == 1
    assert resolution.trial_mean_llr == -2.0
    assert resolution.evidence_cleared is True
    assert resolution.trial_mean_reward == -0.25


def test_trial_checkpoint_keeps_supported_candidate_active_and_roundtrips():
    controller = TrialController()
    _start(controller)
    for step in range(32):
        due, reason = controller.record_transition(
            {"transition_id": step},
            identity_payload=_identity_payload(
                step, candidate_logp=1.0, score=float(step + 1)
            ),
        )
    assert due is True and reason is None
    assert controller.evidence_dissipated_at_checkpoint() is False
    state = copy.deepcopy(controller.state_dict())
    restored = TrialController()
    restored.load_state_dict(state)
    assert restored.active is True
    assert restored.evidence_dissipated_at_checkpoint() is False
    assert (
        restored.state_dict()["lease"]["checkpoint_records"][0]["trial_mean_llr"] == 1.0
    )
