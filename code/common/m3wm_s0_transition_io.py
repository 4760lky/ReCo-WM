from __future__ import annotations
from typing import Any, Iterable, Mapping
import torch

NEUTRAL_FIELDS = (
    "transition_id",
    "obs",
    "agent_action",
    "reward",
    "next_obs",
    "terminated",
    "truncated",
    "episode_id",
)


def _storage_tensors(replay_state: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
    history = replay_state["history"]
    storage = history["_storage"]
    tensors = storage["_storage"]
    length = int(storage["_len"])
    required = {"obs", "action", "reward", "terminated", "truncated", "episode"}
    missing = required - set(tensors)
    if missing:
        raise ValueError(f"replay storage is missing fields: {sorted(missing)}")
    if any((len(tensors[key]) < length for key in required)):
        raise ValueError("replay storage tensor is shorter than its logical length")
    return (tensors, length)


def decode_replay_transitions(replay_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    tensors, length = _storage_tensors(replay_state)
    rows: list[dict[str, Any]] = []
    for index in range(1, length):
        action = torch.as_tensor(tensors["action"][index]).detach().cpu()
        reward = torch.as_tensor(tensors["reward"][index]).reshape(())
        episode = int(torch.as_tensor(tensors["episode"][index]).item())
        previous_episode = int(torch.as_tensor(tensors["episode"][index - 1]).item())
        if episode != previous_episode or not torch.isfinite(action).all():
            continue
        if not torch.isfinite(reward):
            continue
        terminated = bool(
            float(torch.as_tensor(tensors["terminated"][index]).reshape(()).item())
            > 0.5
        )
        truncated = bool(
            float(torch.as_tensor(tensors["truncated"][index]).reshape(()).item()) > 0.5
        )
        rows.append(
            {
                "transition_id": len(rows),
                "obs": torch.as_tensor(tensors["obs"][index - 1])
                .detach()
                .cpu()
                .clone(),
                "agent_action": action.clone(),
                "reward": float(reward.item()),
                "next_obs": torch.as_tensor(tensors["obs"][index])
                .detach()
                .cpu()
                .clone(),
                "terminated": terminated,
                "truncated": truncated,
                "episode_id": episode,
            }
        )
    expected = replay_state.get("total_transitions")
    if expected is not None and len(rows) != int(expected):
        raise ValueError(
            f"decoded {len(rows)} transitions but replay reports {int(expected)}"
        )
    return rows


def selector_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "transition_id": int(row["transition_id"]),
            "obs": torch.as_tensor(row["obs"]).detach().cpu().clone(),
            "action": torch.as_tensor(row["agent_action"]).detach().cpu().clone(),
            "reward": float(row["reward"]),
            "next_obs": torch.as_tensor(row["next_obs"]).detach().cpu().clone(),
            "terminated": bool(row["terminated"]),
            "truncated": bool(row["truncated"]),
        }
        for row in rows
    ]
