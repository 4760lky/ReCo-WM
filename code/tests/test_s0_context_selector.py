from __future__ import annotations
import copy
import pytest
import torch
from common.m3wm_s0_context_router import S0ContextRouter
from common.m3wm_s0_context_selector import S0PNNEvidenceContextSelector
from common.m3wm_s0_pnn import PersistentPNNEnsemble
from common.m3wm_s0_trial_controller import TrialController


def _row(step: int, reward: float = 0.0):
    return {
        "transition_id": step,
        "obs": torch.tensor([0.1, -0.2]),
        "action": torch.tensor([0.3]),
        "reward": reward,
        "next_obs": torch.tensor([0.2, -0.1]),
        "terminated": False,
        "truncated": False,
    }


class FakePredictor:

    def __init__(self, context_id: int, log_prob_fn=None):
        self.context_id = int(context_id)
        self.log_prob_fn = log_prob_fn or (lambda row, cid: -abs(row["reward"] - cid))
        self.fitted = False
        self.initial_calls = []
        self.incremental_calls = []

    def fit_initial(self, rows, *, updates, batch_size):
        self.initial_calls.append((len(rows), updates, batch_size))
        self.fitted = True

    def fit_incremental(self, rows, *, updates, batch_size):
        self.incremental_calls.append((len(rows), updates, batch_size))

    def log_prob(self, row):
        return float(self.log_prob_fn(row, self.context_id))

    def disagreement(self, row):
        return 0.0

    def state_dict(self):
        return {
            "context_id": self.context_id,
            "fitted": self.fitted,
            "initial_calls": copy.deepcopy(self.initial_calls),
            "incremental_calls": copy.deepcopy(self.incremental_calls),
        }

    def load_state_dict(self, state):
        self.context_id = int(state["context_id"])
        self.fitted = bool(state["fitted"])
        self.initial_calls = copy.deepcopy(state["initial_calls"])
        self.incremental_calls = copy.deepcopy(state["incremental_calls"])


class FakeBundle:

    def __init__(self, context_id):
        self.context_id = int(context_id)
        self.predictor_state = None


class FakeBank:

    def __init__(self):
        self.active_context_id = 0
        self.bundles = {0: FakeBundle(0)}
        self.fail_return = False

    def bundle(self, context_id):
        return self.bundles[int(context_id)]

    def create_and_activate(self, *, reason):
        context_id = max(self.bundles) + 1
        self.bundles[context_id] = FakeBundle(context_id)
        self.active_context_id = context_id
        receipt = {"source": context_id - 1, "target": context_id}
        return (self.bundles[context_id], receipt)

    def activate(self, target_context_id, *, reason):
        if self.fail_return:
            raise RuntimeError("synthetic load failure")
        source = self.active_context_id
        self.active_context_id = int(target_context_id)
        return {"source": source, "target": self.active_context_id}

    def trial_working_copy(self, context_id):
        context_id = int(context_id)
        if context_id not in self.bundles or context_id == self.active_context_id:
            raise RuntimeError("invalid trial target")
        return (FakeBundle(context_id), "source-hash", f"candidate-{context_id}-hash")


def _selector(*, threshold=5.0, new_log_prob=None, min_ready_rows=1):
    factory = lambda context_id: FakePredictor(context_id)
    return S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=threshold,
        new_log_prob=new_log_prob or (lambda predictor, row: -100.0),
        min_ready_rows=min_ready_rows,
    )


def _make_ready(selector):
    decision = selector.decide(_row(0), behavior_context_id=0)
    assert decision.action == "WARMUP"
    selector.warmup_assign_and_train(_row(0), behavior_context_id=0)
    assert selector.ready


def test_transition_contract_rejects_environment_labels():
    selector = _selector()
    leaked = dict(_row(0), actual_task="joint-malfunction")
    with pytest.raises(ValueError, match="forbidden selector fields"):
        selector.decide(leaked, behavior_context_id=0)


def test_stay_preserves_mcusum_until_threshold_then_new():
    selector = _selector(
        threshold=10.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row) + 3.0,
    )
    _make_ready(selector)
    for step in range(1, 4):
        decision = selector.decide(_row(step), behavior_context_id=0)
        assert decision.action == "STAY"
        assert decision.new_score == pytest.approx(3.0 * step)
        selector.commit(decision, assigned_context_id=0)
        selector.assign_and_train(_row(step), assigned_context_id=0)
    decision = selector.decide(_row(4), behavior_context_id=0)
    assert decision.action == "NEW"
    assert decision.new_score == pytest.approx(12.0)


def test_router_new_binds_bank_id_then_assigns_trigger_row_to_new_context():
    selector = _selector(
        threshold=2.0, new_log_prob=lambda predictor, row: predictor.log_prob(row) + 3.0
    )
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0))
    receipt = router.route(_row(1))
    assert receipt.decision.action == "NEW"
    assert receipt.behavior_context_id == 0
    assert receipt.assigned_context_id == 1
    assert selector.active_context_id == bank.active_context_id == 1
    assert selector._slots[0].rows[-1]["transition_id"] == 0
    assert selector._slots[1].rows[0]["transition_id"] == 1
    assert bank.bundle(0).predictor_state["fitted_rows"] == 1
    assert bank.bundle(1).predictor_state is None


def test_ordinary_routes_do_not_snapshot_full_predictor_state():
    selector = _selector(min_ready_rows=1)
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0))
    assert bank.bundle(0).predictor_state is None
    router.route(_row(1))
    assert bank.bundle(0).predictor_state is None
    router.sync_active_predictor_state()
    assert bank.bundle(0).predictor_state["fitted_rows"] == 1


def test_failed_return_keeps_selector_and_bank_on_source_context():

    def log_prob(row, context_id):
        return 0.0 if float(row["reward"]) == context_id else -10.0

    factory = lambda context_id: FakePredictor(context_id, log_prob)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=2.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row)
        + (3.0 if predictor.context_id == 0 and float(row["reward"]) == 1.0 else -3.0),
        min_ready_rows=1,
    )
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0, reward=0.0))
    router.route(_row(1, reward=1.0))
    router.route(_row(2, reward=1.0))
    bank.fail_return = True
    with pytest.raises(RuntimeError, match="synthetic load failure"):
        router.route(_row(3, reward=0.0))
    assert selector.active_context_id == 1
    assert bank.active_context_id == 1
    assert selector.records()[-1]["commit_error"].startswith("RuntimeError")


def test_training_budget_is_200_then_20_after_250_new_rows():
    selector = _selector(min_ready_rows=1)
    selector.warmup_assign_and_train(_row(0), behavior_context_id=0)
    predictor = selector._slots[0].predictor
    assert predictor.initial_calls == [(1, 200, 256)]
    for step in range(1, 251):
        decision = selector.decide(_row(step), behavior_context_id=0)
        selector.commit(decision, assigned_context_id=0)
        selector.assign_and_train(_row(step), assigned_context_id=0)
    assert predictor.incremental_calls == [(251, 20, 256)]


def test_h1_selector_view_is_bounded_and_state_roundtrips():
    selector = _selector(min_ready_rows=1)
    selector.warmup_assign_and_train(_row(0), behavior_context_id=0)
    for step in range(1, 5001):
        decision = selector.decide(_row(step), behavior_context_id=0)
        selector.commit(decision, assigned_context_id=0)
        selector.assign_and_train(_row(step), assigned_context_id=0)
    slot = selector._slots[0]
    assert len(slot.rows) == 1024
    assert len(slot.training_view.reservoir_rows()) == 4096
    assert len(slot.training_view) == 5001
    slot.training_view.training_rows(max_rows=256)
    assert slot.training_view.last_sample_stats["unique"] == 256
    saved = selector.state_dict()
    restored = _selector(min_ready_rows=1)
    restored.load_state_dict(saved)
    assert restored.state_dict()["slots"][0]["training_view"]["seen"] == 5001
    assert [row["transition_id"] for row in restored._slots[0].rows] == [
        row["transition_id"] for row in slot.rows
    ]


def test_trial_request_is_below_commit_and_never_delays_formal_return():

    def log_prob(row, context_id):
        return 0.0 if float(row["reward"]) == context_id else -6.0

    factory = lambda context_id: FakePredictor(context_id, log_prob)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=10.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row)
        + (11.0 if predictor.context_id == 0 and row["reward"] == 1.0 else -100.0),
        min_ready_rows=1,
        trial_requests_enabled=True,
    )
    selector.warmup_assign_and_train(_row(0, reward=0.0), behavior_context_id=0)
    created = selector.decide(_row(1, reward=1.0), behavior_context_id=0)
    assert created.action == "NEW"
    selector.commit(created, assigned_context_id=1)
    selector.assign_and_train(_row(1, reward=1.0), assigned_context_id=1)
    probe = selector.decide(_row(2, reward=0.0), behavior_context_id=1)
    assert probe.action == "TRIAL_REQUEST"
    assert probe.target_context_id == 0
    assert probe.scores[0] == pytest.approx(6.0)
    selector.commit(probe, assigned_context_id=1)
    selector.assign_and_train(_row(2, reward=0.0), assigned_context_id=1)
    returned = selector.decide(_row(3, reward=0.0), behavior_context_id=1)
    assert returned.action == "RETURN"
    assert returned.target_context_id == 0


def test_router_trial_keeps_committed_owner_then_flushes_on_real_return():

    def log_prob(row, context_id):
        return 0.0 if float(row["reward"]) == context_id else -6.0

    factory = lambda context_id: FakePredictor(context_id, log_prob)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=10.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row)
        + (11.0 if predictor.context_id == 0 and row["reward"] == 1.0 else -100.0),
        min_ready_rows=1,
        trial_requests_enabled=True,
    )
    bank = FakeBank()
    trial = TrialController(total_ratio=1.0, absolute_limit=4096, cooldown_steps=0)
    router = S0ContextRouter(selector=selector, bank=bank, trial_controller=trial)
    router.route(_row(0, reward=0.0))
    router.route(_row(1, reward=1.0))
    request = router.route(_row(3200, reward=0.0))
    assert request.decision.action == "TRIAL_REQUEST"
    assert request.trial_requested is True
    assert router.start_trial(request.decision, env_steps_seen=3200) is True
    assert bank.active_context_id == 1
    assert trial.behavior_context_id == 0
    returned = router.route(_row(3201, reward=0.0), control_payload={"td": 3201})
    assert returned.decision.action == "RETURN"
    assert returned.behavior_context_id == 0
    assert returned.assigned_context_id == 0
    assert returned.trial_resolution.outcome == "RETURN"
    assert returned.trial_resolution.control_payloads == ({"td": 3201},)
    assert bank.active_context_id == selector.active_context_id == 0
    assert selector._slots[0].rows[-1]["transition_id"] == 3201


def test_target_rejected_aborts_trial_and_never_becomes_trial_new(monkeypatch):

    def log_prob(row, context_id):
        return 0.0 if float(row["reward"]) == context_id else -6.0

    factory = lambda context_id: FakePredictor(context_id, log_prob)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=10.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row)
        + (11.0 if predictor.context_id == 0 and row["reward"] == 1.0 else -100.0),
        min_ready_rows=1,
        trial_requests_enabled=True,
        return_confirmation_enabled=True,
    )
    bank = FakeBank()
    trial = TrialController(total_ratio=1.0, absolute_limit=4096, cooldown_steps=0)
    router = S0ContextRouter(selector=selector, bank=bank, trial_controller=trial)
    router.route(_row(0, reward=0.0))
    router.route(_row(1, reward=1.0))
    request = router.route(_row(3200, reward=0.0))
    assert request.decision.action == "TRIAL_REQUEST"
    assert router.start_trial(request.decision, env_steps_seen=3200)
    monkeypatch.setattr(
        selector, "_advance_return_confirmation", lambda **kwargs: "rejected"
    )
    rejected = router.route(_row(3201, reward=0.0), control_payload={"step": 3201})
    assert rejected.decision.action == "TARGET_REJECTED"
    assert rejected.trial_resolution.outcome == "ABORTED"
    assert rejected.trial_resolution.reason == "historical_future_confirmation_failed"
    assert trial.active is False
    assert bank.active_context_id == selector.active_context_id == 1
    events = [record["event"] for record in trial.records()]
    assert "TRIAL_ABORTED" in events
    assert "TRIAL_NEW" not in events
    assert selector._candidate_holds[1, 0] >= 3201


def test_pnn_log_prob_and_optimizer_state_roundtrip():
    rows = []
    for step in range(40):
        obs = torch.tensor([step / 100.0, -step / 200.0])
        rows.append(
            {
                **_row(step, reward=0.5 + step / 1000.0),
                "obs": obs,
                "next_obs": obs + torch.tensor([0.02, -0.01]),
            }
        )
    model = PersistentPNNEnsemble(2, 1, members=1, hidden=8, layers=1, seed=7)
    model.fit_initial(rows, updates=2, batch_size=16)
    model.fit_incremental(rows, updates=1, batch_size=16)
    value = model.log_prob(rows[-1])
    state = model.state_dict()
    assert model.total_updates == 3
    assert next(iter(state["optimizers"][0]["state"].values()))["step"].item() == 3
    restored = PersistentPNNEnsemble(2, 1, members=1, hidden=8, layers=1, seed=7)
    restored.load_state_dict(state)
    assert restored.log_prob(rows[-1]) == pytest.approx(value, abs=1e-12)
    restored.fit_incremental(rows, updates=1, batch_size=16)
    restored_state = restored.state_dict()
    assert (
        next(iter(restored_state["optimizers"][0]["state"].values()))["step"].item()
        == 4
    )


def test_pnn_moment_matching_matches_direct_gaussian_formula():
    target = torch.tensor([1.2, -0.3, 0.4], dtype=torch.float64)
    means = torch.tensor([[[1.0, -0.2, 0.3]], [[1.4, -0.4, 0.5]]], dtype=torch.float64)
    variances = torch.tensor(
        [[[0.1, 0.2, 0.3]], [[0.2, 0.1, 0.2]]], dtype=torch.float64
    )
    mean = means.mean(dim=0)
    variance = (means.square() + variances).mean(dim=0) - mean.square() + 1e-06
    expected = -0.5 * (
        3 * torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64))
        + torch.log(variance).sum()
        + ((target.reshape(1, -1) - mean).square() / variance).sum()
    )
    assert PersistentPNNEnsemble.moment_matched_log_prob(
        target, means, variances
    ) == pytest.approx(float(expected.item()), abs=1e-12)


def test_new_alternative_uses_identical_moment_matched_covariance():
    means = torch.tensor([[[-3.0, 2.0, 1.0]], [[3.0, -2.0, 1.0]]], dtype=torch.float64)
    variances = torch.full_like(means, 0.01)
    mean, _ = PersistentPNNEnsemble.moment_matched_moments(means, variances)
    active = PersistentPNNEnsemble.moment_matched_log_prob(mean, means, variances)
    new = PersistentPNNEnsemble.new_alternative_log_prob_from_prediction(
        means, variances, num_stds=2.0
    )
    assert new - active == pytest.approx(-0.5 * 3 * 4.0, abs=1e-12)


def test_new_context_settles_until_time_and_two_real_maintenance_rounds():
    factory = lambda context_id: FakePredictor(context_id)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=2.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row)
        + (3.0 if predictor.context_id == 0 else -100.0),
        min_ready_rows=1,
        maintenance_every=1,
        settle_after_new_steps=4,
        new_min_maintenance_rounds=2,
    )
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0))
    assert router.route(_row(1)).decision.action == "NEW"
    assert router.route(_row(2)).decision.action == "SETTLING"
    assert router.route(_row(3)).decision.action == "SETTLING"
    assert selector._slots[1].maintenance_rounds >= 2
    assert router.route(_row(4)).decision.action == "SETTLING"
    released = router.route(_row(5))
    assert released.decision.action == "STAY"
    assert released.decision.new_score == 0.0


def test_new_carries_rejected_history_hold_and_marks_asymmetric_return():
    selector = _selector(
        threshold=2.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row) + 3.0,
        min_ready_rows=1,
    )
    selector.settle_after_new_steps = 0
    selector.new_min_maintenance_rounds = 0
    selector.post_new_failed_candidate_hold_steps = 2048
    selector.post_new_return_confirm_blocks = 3
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0))
    first = router.route(_row(1))
    assert first.decision.action == "NEW"
    assert bank.active_context_id == selector.active_context_id == 1
    selector.register_failed_candidate(
        source_context_id=1, candidate_context_id=0, end_transition_id=1
    )
    second = router.route(_row(2))
    assert second.decision.action == "NEW"
    assert bank.active_context_id == selector.active_context_id == 2
    assert selector._candidate_holds[2, 0] >= 2050
    assert selector._required_return_blocks(0, 2) == 3


def test_return_requires_two_disjoint_future_blocks():

    def log_prob(row, context_id):
        return 0.0 if float(row["reward"]) == context_id else -6.0

    factory = lambda context_id: FakePredictor(context_id, log_prob)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=2.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row)
        + (3.0 if predictor.context_id == 0 and row["reward"] == 1.0 else -100.0),
        min_ready_rows=1,
        return_confirmation_enabled=True,
        return_confirm_block_size=2,
        return_confirm_blocks=2,
    )
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0, reward=0.0))
    assert router.route(_row(1, reward=1.0)).decision.action == "NEW"
    assert router.route(_row(2, reward=0.0)).decision.action == "CONFIRMING"
    assert router.route(_row(3, reward=0.0)).decision.action == "CONFIRMING"
    assert router.route(_row(4, reward=0.0)).decision.action == "CONFIRMING"
    assert router.route(_row(5, reward=0.0)).decision.action == "CONFIRMING"
    returned = router.route(_row(6, reward=0.0))
    assert returned.decision.action == "RETURN"
    assert returned.decision.target_context_id == 0


def _ready_two_context_selector(*, log_prob_fn, threshold=3.0, block_size=2):
    factory = lambda context_id: FakePredictor(context_id, log_prob_fn)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=threshold,
        new_log_prob=lambda predictor, row: predictor.log_prob(row) - 100.0,
        min_ready_rows=1,
        return_confirmation_enabled=True,
        return_confirm_block_size=block_size,
        return_confirm_blocks=1,
    )
    for context_id in (0, 1):
        if context_id not in selector._slots:
            selector._slots[context_id] = selector._new_slot(
                context_id, factory(context_id)
            )
        slot = selector._slots[context_id]
        slot.predictor.fitted = True
        slot.initial_fit_complete = True
        selector._scores[context_id] = 0.0
        selector._raw_scores[context_id] = 0.0
    selector.active_context_id = 1
    return selector


def _commit_identity_row(selector, decision, row):
    selector.commit(decision, assigned_context_id=selector.active_context_id)
    if decision.identity_row_disposition == "STAGE":
        selector.stage_identity_row(row)
    elif decision.identity_row_disposition == "TRAIN":
        selector.assign_and_train(row, assigned_context_id=selector.active_context_id)


def test_return_confirmation_keeps_contiguous_negative_row_below_threshold():
    selector = _ready_two_context_selector(
        log_prob_fn=lambda row, context_id: (
            float(row["reward"]) if context_id == 0 else 0.0
        )
    )
    nomination_row = _row(1, reward=4.0)
    nomination = selector.decide(nomination_row, behavior_context_id=1)
    assert nomination.action == "CONFIRMING"
    _commit_identity_row(selector, nomination, nomination_row)
    negative_row = _row(2, reward=-2.0)
    below_threshold = selector.decide(negative_row, behavior_context_id=1)
    assert below_threshold.scores[0] == pytest.approx(2.0)
    assert below_threshold.action == "CONFIRMING"
    assert [
        item["transition_id"] for item in selector._return_confirmation["current_block"]
    ] == [2]
    _commit_identity_row(selector, below_threshold, negative_row)
    final_row = _row(3, reward=1.0)
    rejected = selector.decide(final_row, behavior_context_id=1)
    assert rejected.action == "TARGET_REJECTED"
    block = rejected.confirmation_block
    assert block["row_ids"] == [2, 3]
    assert block["missing_rows"] == []
    assert block["mean_llr"] == pytest.approx(-0.5)
    assert block["admit_reason"] == "continuous_future_block_rejected"


def test_confirmation_revision_change_invalidates_whole_transaction():
    selector = _ready_two_context_selector(
        log_prob_fn=lambda row, context_id: 4.0 if context_id == 0 else 0.0
    )
    first_row = _row(1)
    first = selector.decide(first_row, behavior_context_id=1)
    _commit_identity_row(selector, first, first_row)
    selector._slots[0].optimizer_updates += 1
    second = selector.decide(_row(2), behavior_context_id=1)
    assert second.action == "CONFIRMING"
    assert second.confirmation_block["event"] == "confirmation_invalidated"
    assert selector._return_confirmation["nomination_step"] == 2
    assert selector._return_confirmation["current_block"] == []


def test_contiguous_confirmation_state_roundtrip_preserves_partial_block():
    selector = _ready_two_context_selector(
        log_prob_fn=lambda row, context_id: 4.0 if context_id == 0 else 0.0,
        block_size=2,
    )
    for step in (1, 2):
        row = _row(step)
        decision = selector.decide(row, behavior_context_id=1)
        _commit_identity_row(selector, decision, row)
    saved = selector.state_dict()
    restored = _ready_two_context_selector(
        log_prob_fn=lambda row, context_id: 4.0 if context_id == 0 else 0.0,
        block_size=2,
    )
    restored.load_state_dict(saved)
    returned = restored.decide(_row(3), behavior_context_id=1)
    assert returned.action == "RETURN"
    assert returned.confirmation_block["row_ids"] == [2, 3]
    assert returned.confirmation_block["observed_rows"] == 2


def test_router_aborts_trial_at_checkpoint_when_identity_evidence_dissipates():
    selector = _ready_two_context_selector(
        log_prob_fn=lambda row, context_id: -1.0 if context_id == 0 else 0.0,
        threshold=300.0,
        block_size=2,
    )
    selector.failed_candidate_hold_steps = 1024
    bank = FakeBank()
    bank.bundles[1] = FakeBundle(1)
    bank.active_context_id = 1
    trial = TrialController()
    lease = trial.start(
        env_steps_seen=6400,
        source_context_id=1,
        candidate_context_id=0,
        source_bundle_hash="source-hash",
        candidate_bundle_hash="candidate-hash",
        selector_state=selector.state_dict(),
        predictor_versions={},
        working_bundle=object(),
    )
    assert lease is not None
    router = S0ContextRouter(selector=selector, bank=bank, trial_controller=trial)
    receipt = None
    for step in range(1, 33):
        receipt = router.route(_row(step))
    assert receipt.trial_checkpoint_due is True
    assert receipt.trial_resolution is not None
    assert receipt.trial_resolution.outcome == "ABORTED"
    assert receipt.trial_resolution.reason == "trial_evidence_dissipated"
    assert receipt.trial_resolution.used_steps == 32
    assert selector.active_context_id == bank.active_context_id == 1
    assert selector._candidate_held(0, 33)


def test_failed_candidate_hold_blocks_direct_return_and_roundtrips():
    selector = _selector(threshold=2.0, min_ready_rows=1)
    _make_ready(selector)
    selector._slots[1] = selector._new_slot(1, FakePredictor(1))
    selector._slots[1].predictor.fitted = True
    selector._slots[1].initial_fit_complete = True
    selector._scores[1] = 0.0
    selector.failed_candidate_hold_steps = 10
    selector.register_failed_candidate(
        source_context_id=0, candidate_context_id=1, end_transition_id=2
    )
    saved = selector.state_dict()
    restored = _selector(threshold=2.0, min_ready_rows=1)
    restored.failed_candidate_hold_steps = 10
    restored.load_state_dict(saved)
    assert restored._candidate_held(1, 3)
    assert not restored._candidate_held(1, 12)


def _new_confirming_selector(*, block_size=2, blocks=2):
    factory = lambda context_id: FakePredictor(context_id)
    selector = S0PNNEvidenceContextSelector(
        initial_predictor=factory(0),
        predictor_factory=factory,
        threshold=3.0,
        new_log_prob=lambda predictor, row: predictor.log_prob(row)
        + float(row["reward"]),
        min_ready_rows=1,
        new_confirmation_enabled=True,
        new_confirm_block_size=block_size,
        new_confirm_blocks=blocks,
    )
    return selector


def test_new_spike_only_nominates_and_negative_future_block_rejects():
    selector = _new_confirming_selector()
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0, reward=0.0))
    nominated = router.route(_row(1, reward=4.0))
    assert nominated.decision.action == "CONFIRMING_NEW"
    assert selector._new_confirmation["current_block"] == []
    assert router.route(_row(2, reward=-1.0)).decision.action == "CONFIRMING_NEW"
    rejected = router.route(_row(3, reward=-1.0))
    assert rejected.decision.action == "NEW_REJECTED"
    assert rejected.decision.confirmation_block["row_ids"] == [2, 3]
    assert 1 not in rejected.decision.confirmation_block["row_ids"]
    assert bank.active_context_id == selector.active_context_id == 0
    assert selector.new_count == 0
    assert selector._slots[0].rows[-1]["transition_id"] == 3


def test_new_requires_two_disjoint_positive_future_blocks():
    selector = _new_confirming_selector()
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0, reward=0.0))
    assert router.route(_row(1, reward=4.0)).decision.action == "CONFIRMING_NEW"
    for step in (2, 3, 4):
        receipt = router.route(_row(step, reward=1.0))
        assert receipt.decision.action == "CONFIRMING_NEW"
    created = router.route(_row(5, reward=1.0))
    assert created.decision.action == "NEW"
    assert created.decision.confirmation_block["row_ids"] == [4, 5]
    assert bank.active_context_id == selector.active_context_id == 1
    assert selector.new_count == 1
    assert selector.new_confirmation_passed_count == 1


def test_new_confirmation_rejects_noncontiguous_future_transition_ids():
    selector = _new_confirming_selector()
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0, reward=0.0))
    router.route(_row(1, reward=4.0))
    rejected = router.route(_row(3, reward=1.0))
    assert rejected.decision.action == "NEW_REJECTED"
    assert rejected.decision.confirmation_block["missing_rows"] == [2]
    assert selector.new_count == 0


def test_new_confirmation_state_roundtrip_preserves_partial_block():
    selector = _new_confirming_selector()
    bank = FakeBank()
    router = S0ContextRouter(selector=selector, bank=bank)
    router.route(_row(0, reward=0.0))
    router.route(_row(1, reward=4.0))
    router.route(_row(2, reward=1.0))
    saved = selector.state_dict()
    restored = _new_confirming_selector()
    restored.load_state_dict(saved)
    assert restored._new_confirmation["nomination_step"] == 1
    assert [
        item["transition_id"] for item in restored._new_confirmation["current_block"]
    ] == [2]
    decision = restored.decide(_row(3, reward=1.0), behavior_context_id=0)
    assert decision.action == "CONFIRMING_NEW"
    assert decision.confirmation_block["row_ids"] == [2, 3]


def test_pnn_fit_audit_records_causal_rows_and_update_budget():
    selector = _new_confirming_selector()
    selector.warmup_assign_and_train(_row(0), behavior_context_id=0)
    records = selector.fit_records()
    assert len(records) == 1
    assert records[0]["fit_kind"] == "initial"
    assert records[0]["context_id"] == 0
    assert records[0]["recent_first_transition_id"] == 0
    assert records[0]["recent_last_transition_id"] == 0
    assert records[0]["optimizer_updates_before"] == 0
    assert records[0]["optimizer_updates_after"] == 200
