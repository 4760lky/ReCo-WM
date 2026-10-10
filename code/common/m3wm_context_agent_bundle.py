from __future__ import annotations
import copy
import hashlib
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

SCHEMA = "m3wm-context-agent-bundle-v1"


def _clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone(v) for (k, v) in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple((_clone(v) for v in value))
    return copy.deepcopy(value)


def _digest(value: Any) -> str:
    chunks: list[bytes] = []

    def visit(item: Any, path: str = "") -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            chunks.append(path.encode() + b"|tensor|" + str(tensor.dtype).encode())
            chunks.append(str(tuple(tensor.shape)).encode())
            chunks.append(tensor.numpy().tobytes())
        elif item.__class__.__module__.startswith("tensordict") and callable(
            getattr(item, "to_dict", None)
        ):
            visit(item.to_dict(), path)
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
        else:
            chunks.append(path.encode() + b"|" + repr(item).encode())

    visit(value)
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def _capture_rng(
    *, generators: Mapping[str, torch.Generator] | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state().clone(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = None
    if generators:
        state["generators"] = {
            str(name): generator.get_state().clone()
            for (name, generator) in generators.items()
            if isinstance(generator, torch.Generator)
        }
    return state


def _restore_rng(
    state: Mapping[str, Any], *, generators: Mapping[str, torch.Generator] | None = None
) -> None:
    torch.random.set_rng_state(state["torch_cpu"].clone())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.clone() for item in cuda_state])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    for name, value in state.get("generators", {}).items():
        generator = None if generators is None else generators.get(name)
        if isinstance(generator, torch.Generator):
            generator.set_state(value.clone())


_RUNTIME_ATTRS = (
    "_m3wm_train_updates",
    "_m3wm_schedule_update_offset",
    "_m3wm_inherited_specialization_eligible",
    "_m3wm_control_update_count",
    "_m3wm_specialization_active",
    "_m3wm_current_temperature",
    "_m3wm_current_base_lr_scale",
    "_m3wm_current_residual_scale",
    "_m3wm_current_reward_residual_scale",
    "_m3wm_current_value_residual_scale",
    "_plain_incumbent_last_accept_step",
    "_plain_incumbent_proposal_metrics",
    "_prev_mean",
    "_m3wm_raw_theta",
    "_m3wm_smoothed_theta",
    "_m3wm_last_theta",
    "_m3wm_context_version",
    "_m3wm_active_context_id",
    "_m3wm_active_context_origin",
)


def _capture_agent(
    agent: Any, *, generators: Mapping[str, torch.Generator] | None = None
) -> dict[str, Any]:
    if agent is None:
        return {"present": False}
    context_export = getattr(agent, "export_context_train_state", None)
    if callable(context_export):
        return {
            "present": True,
            "custom_context_codec": True,
            "payload": _clone(context_export()),
        }
    payload: dict[str, Any] = {
        "present": True,
        "model": _clone(agent.model.state_dict()) if hasattr(agent, "model") else None,
        "optim": _clone(agent.optim.state_dict()) if hasattr(agent, "optim") else None,
        "pi_optim": (
            _clone(agent.pi_optim.state_dict()) if hasattr(agent, "pi_optim") else None
        ),
        "scale": _clone(agent.scale.state_dict()) if hasattr(agent, "scale") else None,
        "runtime": {},
        "rng": _capture_rng(generators=generators),
    }
    for name in _RUNTIME_ATTRS:
        if hasattr(agent, name):
            payload["runtime"][name] = _clone(getattr(agent, name))
    cfg = getattr(agent, "cfg", None)
    return payload


def _restore_agent(
    agent: Any,
    payload: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator] | None = None,
    restore_rng: bool = True,
) -> None:
    if agent is None or not payload.get("present", False):
        return
    if payload.get("custom_context_codec", False):
        context_import = getattr(agent, "import_context_train_state", None)
        if not callable(context_import):
            raise TypeError("Target agent has no Context train-state importer")
        context_import(_clone(payload["payload"]), restore_rng=restore_rng)
        return
    if payload.get("model") is not None and hasattr(agent, "model"):
        agent.model.load_state_dict(payload["model"])
    if payload.get("optim") is not None and hasattr(agent, "optim"):
        agent.optim.load_state_dict(payload["optim"])
    if payload.get("pi_optim") is not None and hasattr(agent, "pi_optim"):
        agent.pi_optim.load_state_dict(payload["pi_optim"])
    if payload.get("scale") is not None and hasattr(agent, "scale"):
        agent.scale.load_state_dict(payload["scale"])
    for name, value in payload.get("runtime", {}).items():
        if hasattr(agent, name):
            restored = _clone(value)
            current = getattr(agent, name)
            if torch.is_tensor(restored) and torch.is_tensor(current):
                restored = restored.to(device=current.device, dtype=current.dtype)
            elif torch.is_tensor(restored) and hasattr(agent, "device"):
                restored = restored.to(device=agent.device)
            setattr(agent, name, restored)
    context_state = payload.get("context_state")
    if restore_rng and payload.get("rng") is not None:
        _restore_rng(payload["rng"], generators=generators)


def _capture_replay(buffer: Any) -> dict[str, Any]:
    if buffer is None:
        return {"present": False}
    payload: dict[str, Any] = {"present": True, "schema": "buffer-codec-v1"}
    if hasattr(buffer, "state_dict") and callable(buffer.state_dict):
        payload["state_dict"] = _clone(buffer.state_dict())
    else:
        raise TypeError("Context replay has no state_dict codec")
    metadata = {}
    for name in (
        "_num_eps",
        "_total_transitions",
        "_active_episode_id",
        "_active_episode_length",
        "_max_episode_length",
        "_next_replay_trajectory_id",
        "_replay_trajectory_ids",
        "_replay_flush_state",
        "_replay_resolution_count",
        "_replay_flushed_transition_count",
        "_replay_context_switch_count",
        "_recent_refresh_count",
        "_last_sample_composition",
        "_last_sample_valid_fraction",
    ):
        if hasattr(buffer, name):
            metadata[name] = _clone(getattr(buffer, name))
    payload["metadata"] = metadata
    payload["digest"] = _digest(
        {"state_dict": payload.get("state_dict"), "metadata": metadata}
    )
    return payload


def _restore_replay(buffer: Any, payload: Mapping[str, Any]) -> None:
    if buffer is None or not payload.get("present", False):
        return
    expected = payload.get("digest")
    actual = _digest(
        {
            "state_dict": payload.get("state_dict"),
            "metadata": payload.get("metadata", {}),
        }
    )
    if expected and expected != actual:
        raise ValueError("Replay state digest mismatch")
    state = payload.get("state_dict")
    if state is None:
        raise ValueError("Context replay payload is missing its state_dict")
    if not hasattr(buffer, "load_state_dict"):
        raise TypeError("Target Context replay has no load_state_dict codec")
    buffer.load_state_dict(state)
    for name, value in payload.get("metadata", {}).items():
        if hasattr(buffer, name):
            setattr(buffer, name, _clone(value))


@dataclass
class ContextAgentBundle:
    context_id: int
    control_agent: Any
    control_replay: Any = None
    theta_agent: Any = None
    theta_replay: Any = None
    predictor_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = False
    runtime_segment_id: int = 0
    schema = SCHEMA

    @property
    def materialized(self) -> bool:
        return self.control_agent is not None and self.control_replay is not None

    def release_runtime(self) -> None:
        if self.active:
            raise RuntimeError("Cannot release the active Context runtime")
        self.control_agent = None
        self.control_replay = None
        self.theta_agent = None
        self.theta_replay = None

    def reset_activation_transients(self) -> None:
        for owner in (self.control_agent, self.theta_agent):
            reset = getattr(owner, "reset_context_activation_transients", None)
            if callable(reset):
                reset()

    def export_train_state(self) -> dict[str, Any]:
        if not self.materialized:
            raise RuntimeError("Cannot export an offloaded Context without loading it")
        generators = {}
        for name, owner in (
            ("control", self.control_agent),
            ("theta", self.theta_agent),
        ):
            generator = getattr(owner, "_plain_incumbent_score_generator", None)
            if isinstance(generator, torch.Generator):
                generators[f"{name}_score"] = generator
            generator = getattr(owner, "proposal_generator", None)
            if isinstance(generator, torch.Generator):
                generators[f"{name}_proposal"] = generator
        payload = {
            "schema": self.schema,
            "context_id": int(self.context_id),
            "active": bool(self.active),
            "runtime_segment_id": int(self.runtime_segment_id),
            "metadata": _clone(self.metadata),
            "control": _capture_agent(self.control_agent, generators=generators),
            "theta": _capture_agent(self.theta_agent, generators=generators),
            "control_replay": _capture_replay(self.control_replay),
            "theta_replay": _capture_replay(self.theta_replay),
            "predictor_state": _clone(self.predictor_state),
            "global_rng": _capture_rng(generators=generators),
        }
        payload["bundle_hash"] = _digest(
            {k: v for (k, v) in payload.items() if k != "bundle_hash"}
        )
        return payload

    def import_train_state(
        self, payload: Mapping[str, Any], *, restore_rng: bool = True
    ) -> None:
        if payload.get("schema") != self.schema:
            raise ValueError(
                f"Unsupported Context bundle schema: {payload.get('schema')!r}"
            )
        expected = payload.get("bundle_hash")
        actual = _digest({k: v for (k, v) in payload.items() if k != "bundle_hash"})
        if expected and expected != actual:
            raise ValueError("Context bundle hash mismatch")
        if int(payload.get("context_id", -1)) != int(self.context_id):
            raise ValueError("Context bundle id does not match target bundle")
        generators = {}
        for name, owner in (
            ("control", self.control_agent),
            ("theta", self.theta_agent),
        ):
            for suffix, attr in (
                ("score", "_plain_incumbent_score_generator"),
                ("proposal", "proposal_generator"),
            ):
                generator = getattr(owner, attr, None)
                if isinstance(generator, torch.Generator):
                    generators[f"{name}_{suffix}"] = generator
        _restore_agent(
            self.control_agent,
            payload["control"],
            generators=generators,
            restore_rng=False,
        )
        _restore_agent(
            self.theta_agent, payload["theta"], generators=generators, restore_rng=False
        )
        _restore_replay(self.control_replay, payload["control_replay"])
        _restore_replay(self.theta_replay, payload["theta_replay"])
        self.predictor_state = _clone(payload.get("predictor_state"))
        self.metadata = _clone(payload.get("metadata", {}))
        self.runtime_segment_id = int(payload.get("runtime_segment_id", 0))
        self.active = bool(payload.get("active", False))
        if restore_rng:
            _restore_rng(payload["global_rng"], generators=generators)

    def save_atomic(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.export_train_state()
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            torch.save(payload, tmp)
            with tmp.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()
        return str(payload["bundle_hash"])

    @staticmethod
    def load_payload(
        path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("Invalid Context bundle payload")
        expected = payload.get("bundle_hash")
        actual = _digest({k: v for (k, v) in payload.items() if k != "bundle_hash"})
        if expected != actual:
            raise ValueError("Context bundle hash mismatch")
        return payload
