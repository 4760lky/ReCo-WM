from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Mapping
import torch
from common.buffer import Buffer
from common.m3wm_context_agent_bundle import ContextAgentBundle, _clone, _digest
from common.m3wm_control_identity_shadow import PlainControlIdentityShadow
from tdmpc2 import TDMPC2


class TDMPC2ContextBundleFactory:

    def __init__(self, cfg, *, work_dir: str | Path):
        self.cfg = cfg
        self.work_dir = Path(work_dir)

    @staticmethod
    def _capture_runtime_source(source) -> dict[str, Any]:
        runtime = {}
        for name in (
            "_m3wm_specialization_active",
            "_m3wm_current_temperature",
            "_m3wm_current_base_lr_scale",
            "_m3wm_current_residual_scale",
            "_m3wm_current_reward_residual_scale",
            "_m3wm_current_value_residual_scale",
            "_plain_incumbent_last_accept_step",
        ):
            if hasattr(source, name):
                runtime[name] = _clone(getattr(source, name))
        inherited_progress = int(getattr(source, "_m3wm_train_updates", 0)) + int(
            getattr(source, "_m3wm_schedule_update_offset", 0)
        )
        return {
            "model": _clone(source.model.state_dict()),
            "scale": _clone(source.scale.state_dict()),
            "runtime": runtime,
            "inherited_schedule_progress": inherited_progress,
            "inherited_specialization_eligible": bool(
                getattr(source, "_m3wm_specialization_active", False)
            ),
        }

    @staticmethod
    def _apply_runtime_source(target, payload: Mapping[str, Any]) -> None:
        target.model.load_state_dict(_clone(payload["model"]))
        target.scale.load_state_dict(_clone(payload["scale"]))
        for name, value in payload.get("runtime", {}).items():
            if hasattr(target, name):
                setattr(target, name, _clone(value))
        target._m3wm_schedule_update_offset = int(
            payload["inherited_schedule_progress"]
        )
        target._m3wm_inherited_specialization_eligible = bool(
            payload["inherited_specialization_eligible"]
        )
        target._prev_mean.zero_()

    def capture_init_capsule(
        self,
        source: ContextAgentBundle,
        *,
        nomination_transition_id: int,
        transaction_id: str,
        source_revision: int,
    ) -> dict[str, Any]:
        control = self._capture_runtime_source(source.control_agent)
        theta_side = self._capture_runtime_source(source.theta_agent.side_agent)
        theta = {
            "side_agent": theta_side,
            "current_theta": _clone(source.theta_agent.current_theta),
            "proposal_theta_version": int(source.theta_agent.proposal_theta_version),
            "proposal_generator_state": _clone(
                source.theta_agent.proposal_generator.get_state()
            ),
        }
        component_hashes = {"control": _digest(control), "theta": _digest(theta)}
        capsule = {
            "schema": "m3wm-context-init-capsule-v1",
            "source_context_id": int(source.context_id),
            "nomination_transition_id": int(nomination_transition_id),
            "transaction_id": str(transaction_id),
            "source_revision": int(source_revision),
            "captured_control_updates": int(
                getattr(source.control_agent, "_m3wm_train_updates", 0)
            ),
            "captured_theta_updates": int(
                getattr(source.theta_agent.side_agent, "_m3wm_train_updates", 0)
            ),
            "control": control,
            "theta": theta,
            "component_hashes": component_hashes,
        }
        capsule["capsule_hash"] = _digest(capsule)
        return capsule

    @staticmethod
    def validate_init_capsule(payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != "m3wm-context-init-capsule-v1":
            raise ValueError("invalid Context initialization capsule schema")
        expected = str(payload.get("capsule_hash", ""))
        body = {key: value for (key, value) in payload.items() if key != "capsule_hash"}
        if not expected or expected != _digest(body):
            raise ValueError("Context initialization capsule digest mismatch")
        expected_components = payload.get("component_hashes", {})
        if expected_components.get("control") != _digest(payload["control"]):
            raise ValueError("control initialization component digest mismatch")
        if expected_components.get("theta") != _digest(payload["theta"]):
            raise ValueError("theta initialization component digest mismatch")

    @staticmethod
    def _copy_runtime_weights(target, source) -> None:
        target.model.load_state_dict(copy.deepcopy(source.model.state_dict()))
        target.scale.load_state_dict(copy.deepcopy(source.scale.state_dict()))
        for name in (
            "_m3wm_specialization_active",
            "_m3wm_current_temperature",
            "_m3wm_current_base_lr_scale",
            "_m3wm_current_residual_scale",
            "_m3wm_current_reward_residual_scale",
            "_m3wm_current_value_residual_scale",
        ):
            if hasattr(source, name) and hasattr(target, name):
                setattr(target, name, copy.deepcopy(getattr(source, name)))
        inherited_progress = int(getattr(source, "_m3wm_train_updates", 0)) + int(
            getattr(source, "_m3wm_schedule_update_offset", 0)
        )
        target._m3wm_schedule_update_offset = inherited_progress
        target._m3wm_inherited_specialization_eligible = bool(
            getattr(source, "_m3wm_specialization_active", False)
        )
        if hasattr(source, "_plain_incumbent_last_accept_step"):
            target._plain_incumbent_last_accept_step = int(
                source._plain_incumbent_last_accept_step
            )
        target._prev_mean.zero_()

    def __call__(
        self,
        context_id: int,
        source: ContextAgentBundle,
        fresh_optimizer: bool,
        *,
        init_payload: Mapping[str, Any] | None = None,
    ) -> ContextAgentBundle:
        context_dir = self.work_dir / f"context_{int(context_id):04d}"
        context_dir.mkdir(parents=True, exist_ok=True)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            seed = int(self.cfg.seed) + int(context_id) * 1000003
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            control = TDMPC2(self.cfg)
            theta = PlainControlIdentityShadow(
                self.cfg,
                device=getattr(control, "device", "cuda:0"),
                seed=seed + 7919,
                work_dir=context_dir,
            )
        if not theta.full_tdmpc2_enabled:
            raise ValueError("Context-agent mode requires a complete theta sidecar")
        if fresh_optimizer:
            if init_payload is None:
                self._copy_runtime_weights(control, source.control_agent)
                self._copy_runtime_weights(
                    theta.side_agent, source.theta_agent.side_agent
                )
                theta.current_theta = source.theta_agent.current_theta.detach().clone()
                theta.proposal_theta_version = int(
                    source.theta_agent.proposal_theta_version
                )
                theta.proposal_generator.set_state(
                    source.theta_agent.proposal_generator.get_state().clone()
                )
            else:
                self.validate_init_capsule(init_payload)
                if int(init_payload["source_context_id"]) != int(source.context_id):
                    raise ValueError("initialization capsule source Context mismatch")
                self._apply_runtime_source(control, init_payload["control"])
                self._apply_runtime_source(
                    theta.side_agent, init_payload["theta"]["side_agent"]
                )
                theta.current_theta = _clone(init_payload["theta"]["current_theta"])
                theta.proposal_theta_version = int(
                    init_payload["theta"]["proposal_theta_version"]
                )
                theta.proposal_generator.set_state(
                    _clone(init_payload["theta"]["proposal_generator_state"])
                )
            theta.initial_data_warmup_required = False
        live_parent_init_hash = None
        if fresh_optimizer and init_payload is not None:
            live_parent_init_hash = _digest(
                {
                    "control": self._capture_runtime_source(source.control_agent),
                    "theta": {
                        "side_agent": self._capture_runtime_source(
                            source.theta_agent.side_agent
                        ),
                        "current_theta": _clone(source.theta_agent.current_theta),
                        "proposal_theta_version": int(
                            source.theta_agent.proposal_theta_version
                        ),
                        "proposal_generator_state": _clone(
                            source.theta_agent.proposal_generator.get_state()
                        ),
                    },
                }
            )
        return ContextAgentBundle(
            context_id=int(context_id),
            control_agent=control,
            control_replay=Buffer(self.cfg),
            theta_agent=theta,
            theta_replay=None,
            predictor_state=None,
            metadata={
                "creation": (
                    "copy_control_weights"
                    if fresh_optimizer
                    else "reconstruct_for_durable_restore"
                ),
                "parent_context_id": int(source.context_id),
                "new_optimizer_policy": "fresh" if fresh_optimizer else "restore",
                "replay_policy": "empty_independent" if fresh_optimizer else "restore",
                "control_init_source_step": (
                    None
                    if init_payload is None
                    else int(init_payload["nomination_transition_id"])
                ),
                "init_capsule_hash": (
                    None if init_payload is None else str(init_payload["capsule_hash"])
                ),
                "init_transaction_id": (
                    None
                    if init_payload is None
                    else str(init_payload["transaction_id"])
                ),
                "live_parent_init_hash": live_parent_init_hash,
                "init_component_hashes": (
                    None
                    if init_payload is None
                    else copy.deepcopy(init_payload["component_hashes"])
                ),
            },
            active=False,
            runtime_segment_id=int(source.runtime_segment_id) + 1,
        )
