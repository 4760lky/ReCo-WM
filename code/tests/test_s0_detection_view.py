from __future__ import annotations
import copy
import torch
from common.m3wm_s0_detection_view import DetectionTrainingView
from common.m3wm_s0_transition_io import decode_replay_transitions, selector_rows


def _row(index: int):
    return {
        "transition_id": index,
        "obs": torch.tensor([float(index)]),
        "action": torch.tensor([0.1]),
        "reward": float(index),
        "next_obs": torch.tensor([float(index + 1)]),
        "terminated": False,
        "truncated": False,
    }


def test_detection_view_is_bounded_uniform_and_roundtrips_private_rng():
    parent_before = torch.random.get_rng_state().clone()
    view = DetectionTrainingView(recent_capacity=4, reservoir_capacity=8, seed=17)
    for index in range(100):
        view.add(_row(index))
    assert torch.equal(parent_before, torch.random.get_rng_state())
    assert [row["transition_id"] for row in view.recent_rows()] == [96, 97, 98, 99]
    assert len(view.reservoir_rows()) == 8
    assert any((row["transition_id"] < 90 for row in view.reservoir_rows()))
    saved = copy.deepcopy(view.state_dict())
    expected = view.training_rows(max_rows=8)
    restored = DetectionTrainingView(recent_capacity=4, reservoir_capacity=8, seed=17)
    restored.load_state_dict(saved)
    actual = restored.training_rows(max_rows=8)
    assert [row["transition_id"] for row in actual] == [
        row["transition_id"] for row in expected
    ]
    assert len({row["transition_id"] for row in actual}) == len(actual)


def test_replay_decoder_uses_previous_observation_and_skips_seed_rows():
    nan = float("nan")
    replay = {
        "history": {
            "_storage": {
                "_len": 5,
                "_storage": {
                    "obs": torch.tensor([[0.0], [1.0], [2.0], [10.0], [11.0]]),
                    "action": torch.tensor([[nan], [0.1], [0.2], [nan], [0.3]]),
                    "reward": torch.tensor([nan, 1.0, 2.0, nan, 3.0]),
                    "terminated": torch.tensor([nan, 0.0, 1.0, nan, 0.0]),
                    "truncated": torch.tensor([nan, 0.0, 0.0, nan, 1.0]),
                    "episode": torch.tensor([0, 0, 0, 1, 1]),
                },
            }
        },
        "total_transitions": 3,
    }
    rows = decode_replay_transitions(replay)
    assert len(rows) == 3
    assert torch.equal(rows[0]["obs"], torch.tensor([0.0]))
    assert torch.equal(rows[0]["next_obs"], torch.tensor([1.0]))
    assert torch.equal(rows[2]["obs"], torch.tensor([10.0]))
    assert rows[2]["episode_id"] == 1
    assert rows[2]["truncated"] is True
    clean = selector_rows(rows)
    assert "agent_action" not in clean[0]
    assert "episode_id" not in clean[0]
    assert set(clean[0]) == {
        "transition_id",
        "obs",
        "action",
        "reward",
        "next_obs",
        "terminated",
        "truncated",
    }
