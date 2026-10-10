import pytest
import torch
from run import _parser, _resolve
from test_s0_context_selector import _make_ready, _row, _selector


@pytest.mark.parametrize("family", ["HC", "Hopper", "Walker2d", "DMC"])
def test_environment_configuration(family, tmp_path):
    args = _parser().parse_args(["--env", family, "--output-dir", str(tmp_path)])
    cfg, protocol, _ = _resolve(args)
    assert cfg.steps == 320000
    assert cfg.m3wm_context_selector == "pnn_evidence"
    assert cfg.m3wm_pnn_evidence_cusum_threshold == 300.0
    assert cfg.wandb_project == "reco-wm"
    assert protocol["change_freq"] == 40000
    assert sum(protocol["phase_lengths"]) == cfg.steps
    assert protocol["expected_action_dim"] in {3, 6}


def test_pnn_evidence_selector_class_name():
    from common.m3wm_s0_context_selector import S0PNNEvidenceContextSelector

    assert isinstance(_selector(), S0PNNEvidenceContextSelector)


def test_public_project_name():
    assert "ReCo-WM" in _parser().description


def test_pnn_evidence_selector_current_schema_roundtrip():
    selector = _selector()
    _make_ready(selector)
    saved = selector.state_dict()
    assert saved["schema"] == "reco-wm-s0-pnn-evidence-context-selector-v1"
    restored = _selector()
    restored.load_state_dict(saved)
    assert restored.ready
    assert restored.active_context_id == selector.active_context_id
    assert restored.state_dict()["schema"] == saved["schema"]


@pytest.mark.parametrize(
    "schema",
    [
        None,
        "unsupported-selector-schema",
        "reco-wm-s0-pnn-evidence-context-selector-v0",
    ],
)
def test_pnn_evidence_selector_rejects_noncurrent_schema(schema):
    selector = _selector()
    saved = selector.state_dict()
    saved["schema"] = schema
    with pytest.raises(ValueError, match="invalid S0 selector state schema"):
        selector.load_state_dict(saved)


@pytest.mark.parametrize(
    "field",
    ["task", "actual_task", "phase", "stage", "context_label", "target_context_id"],
)
def test_detector_rejects_extra_labels(field):
    selector = _selector()
    row = dict(_row(0), **{field: 1})
    with pytest.raises(ValueError, match="forbidden selector fields"):
        selector.decide(row, behavior_context_id=0)


def test_elapsed_step_alone_does_not_change_context():
    selector = _selector()
    _make_ready(selector)
    steps = [
        1,
        39999,
        40000,
        40001,
        79999,
        80000,
        120000,
        160000,
        200000,
        240000,
        280000,
        319999,
    ]
    for step in steps:
        row = _row(step)
        decision = selector.decide(row, behavior_context_id=0)
        assert decision.action == "STAY"
        selector.commit(decision, assigned_context_id=0)
        selector.assign_and_train(row, assigned_context_id=0)
    assert selector.active_context_id == 0


def test_logger_finish_fails_when_checkpoint_is_not_written(tmp_path):
    from common.logger import Logger

    logger = Logger.__new__(Logger)
    logger._save_agent = True
    logger._model_dir = tmp_path
    logger._wandb = None

    with pytest.raises(RuntimeError, match="final model"):
        logger.finish(object())


def test_proposal_records_a_restored_context_return():
    from common.m3wm_control_identity_shadow import PlainControlIdentityShadow

    shadow = PlainControlIdentityShadow.__new__(PlainControlIdentityShadow)
    shadow._pending_context_return = False
    assert shadow._consume_context_return() is False
    shadow.mark_context_return()
    assert shadow._consume_context_return() is True
    assert shadow._consume_context_return() is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA learner test")
def test_real_learner_bundle_roundtrip(tmp_path):
    from common.m3wm_context_agent_bundle import ContextAgentBundle
    from common.m3wm_control_identity_shadow import PlainControlIdentityShadow
    from common.buffer import Buffer
    from tdmpc2 import TDMPC2

    cfg, protocol, _ = _resolve(
        _parser().parse_args(["--env", "HC", "--output-dir", str(tmp_path)])
    )
    cfg.obs_shape = {"state": (protocol["expected_observation_dim"],)}
    cfg.action_dim = protocol["expected_action_dim"]
    cfg.episode_length = 1000

    def initial_bundle(directory):
        return ContextAgentBundle(
            context_id=0,
            control_agent=TDMPC2(cfg),
            control_replay=Buffer(cfg),
            theta_agent=PlainControlIdentityShadow(cfg, work_dir=directory),
            active=True,
        )

    source = initial_bundle(tmp_path / "source")
    for agent in (source.control_agent, source.theta_agent.side_agent):
        agent.optim.zero_grad(set_to_none=True)
        loss = sum(
            (
                p.square().mean()
                for group in agent.optim.param_groups
                for p in group["params"]
            )
        )
        loss.backward()
        agent.optim.step()
    source.theta_agent.current_theta = torch.tensor([0.1, 0.2, 0.3, 0.4])
    path = tmp_path / "context.pt"
    source.save_atomic(path)
    target = initial_bundle(tmp_path / "target")
    target.import_train_state(ContextAgentBundle.load_payload(path))
    for left, right in (
        (source.control_agent, target.control_agent),
        (source.theta_agent.side_agent, target.theta_agent.side_agent),
    ):
        for key, value in left.model.state_dict().items():
            torch.testing.assert_close(
                value, right.model.state_dict()[key], rtol=0, atol=0
            )
        assert len(right.optim.state) == len(left.optim.state) > 0
    torch.testing.assert_close(
        source.theta_agent.current_theta.cpu(),
        target.theta_agent.current_theta.cpu(),
        rtol=0,
        atol=0,
    )
