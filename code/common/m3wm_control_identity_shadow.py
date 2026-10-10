from __future__ import annotations
import copy
import json
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import torch
from common import math as m3wm_math


class PlainControlIdentityShadow:
    schema = "m3wm_plain_control_identity_shadow"
    version = 1

    def __init__(
        self,
        cfg,
        *,
        device: torch.device | str = "cuda:0",
        seed: int = 0,
        work_dir: str | Path | None = None,
    ) -> None:
        self.control_cfg = cfg
        self.device = torch.device(device)
        self.seed = int(seed)
        self.work_dir = None if work_dir is None else Path(work_dir)
        self.identity_events_enabled = False
        self.proposal_mode = str(
            getattr(cfg, "m3wm_plain_incumbent_proposal_mode", "D")
        ).upper()
        self.theta_learning_enabled = True
        self.full_tdmpc2_enabled = True
        side_cfg = copy.deepcopy(cfg)
        setattr(side_cfg, "enable_m3wm", True)
        setattr(side_cfg, "m3wm_mode", "interface_direct_theta")
        setattr(side_cfg, "m3wm_composition_mode", "hybrid_modular")
        setattr(side_cfg, "m3wm_num_primitives", 4)
        setattr(side_cfg, "m3wm_separate_composition_heads", False)
        setattr(side_cfg, "m3wm_modular_value", True)
        self.side_cfg = side_cfg
        self.side_agent = None
        self.side_buffer = None
        from common.buffer import Buffer
        from tdmpc2 import TDMPC2

        self.side_agent = TDMPC2(side_cfg)
        self.side_buffer = Buffer(side_cfg)
        self.model = self.side_agent.model
        self.optimizer = self.side_agent.optim
        obs_shape = cfg.obs_shape.get("state", cfg.obs_shape)
        self.observer = None
        self.replay = deque(maxlen=4096)
        self.feature_history = deque(maxlen=32)
        self.update_interval = int(getattr(cfg, "m3wm_sidecar_update_interval", 8))
        self.min_update_records = int(
            getattr(cfg, "m3wm_sidecar_min_update_records", 32)
        )
        self.step = 0
        self.sidecar_updates = 0
        self.last_prediction_loss = float("nan")
        self.last_identity_status = "ready"
        self.last_error = ""
        self.failed = False
        self.last_theta_entropy = float("nan")
        self.last_feature_norm = float("nan")
        self._wall_ms = 0.0
        self.current_theta = torch.full((4,), 0.25, dtype=torch.float32)
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.proposal_generator = torch.Generator(device=generator_device)
        self.proposal_generator.manual_seed(self.seed + 15485863)
        self.proposal_count = int(getattr(cfg, "m3wm_plain_proposal_count", 16))
        self.profile_cuda_sections = False
        if self.proposal_count != 16:
            raise ValueError("the initial proposal budget is fixed at 16")
        self.proposal_calls = 0
        self.proposal_valid_count = 0
        self.proposal_theta_version = 0
        self.last_proposal_status = "not_evaluated"
        self.last_proposal_context_id = -1
        self.last_proposal_wall_ms = 0.0
        self.last_proposal_input_copy_sync_ms = 0.0
        self.last_proposal_sampling_ms = 0.0
        self.last_proposal_ranking_ms = 0.0
        self.last_proposal_other_ms = 0.0
        self.last_proposal_mode = "OFF"
        self.last_proposal_ranking_source = "disabled"
        self.last_proposal_selected_index = -1
        self.last_proposal_used_recall = 0
        self._pending_context_return = False
        self.last_proposal_theta_current_linf_delta = float("nan")
        private_cpu = torch.Generator(device="cpu")
        private_cpu.manual_seed(self.seed + 32452843)
        self._private_cpu_rng_state = private_cpu.get_state().clone()
        self._private_cuda_rng_state = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            devices = [self.device.index or torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.cuda.manual_seed_all(self.seed + 49979687)
                self._private_cuda_rng_state = torch.cuda.get_rng_state(
                    self.device
                ).clone()
        self.initial_data_warmup_required = True

    @contextmanager
    def private_rng_scope(self):
        parent_cpu = torch.random.get_rng_state().clone()
        parent_cuda = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        )
        torch.random.set_rng_state(self._private_cpu_rng_state.clone())
        if self._private_cuda_rng_state is not None:
            torch.cuda.set_rng_state(
                self._private_cuda_rng_state.clone(), device=self.device
            )
        try:
            yield
        finally:
            self._private_cpu_rng_state = torch.random.get_rng_state().clone()
            if self._private_cuda_rng_state is not None:
                self._private_cuda_rng_state = torch.cuda.get_rng_state(
                    self.device
                ).clone()
            torch.random.set_rng_state(parent_cpu)
            if parent_cuda is not None:
                torch.cuda.set_rng_state_all(parent_cuda)

    @staticmethod
    def _clone_cpu(value: Any) -> torch.Tensor:
        return (
            torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone().flatten()
        )

    def _identity_observe(self, record: dict[str, Any]) -> None:
        return

    def mark_context_return(self) -> None:
        """Mark the next proposal as using a restored Context state."""
        self._pending_context_return = True

    def _consume_context_return(self) -> bool:
        value = bool(self._pending_context_return)
        self._pending_context_return = False
        return value

    @torch.no_grad()
    def _build_live_feature(self, record: dict[str, Any]) -> torch.Tensor:
        obs = record["obs"].to(self.device).unsqueeze(0)
        action = record["action"].to(self.device).unsqueeze(0)
        reward = torch.tensor(
            [record["reward"]], dtype=torch.float32, device=self.device
        )
        next_obs = record["next_obs"].to(self.device).unsqueeze(0)
        feature = self.model.build_context_features_from_observations(
            obs, action, reward, next_obs, task=None
        )
        return feature.reshape(-1).detach().cpu().clone()

    def observe_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        *,
        terminated: bool = False,
        truncated: bool = False,
        episode_id: int | None = None,
    ) -> None:
        with self.private_rng_scope():
            return self._observe_transition_private(
                obs,
                action,
                reward,
                next_obs,
                terminated=terminated,
                truncated=truncated,
                episode_id=episode_id,
            )

    def _observe_transition_private(
        self,
        obs,
        action,
        reward,
        next_obs,
        *,
        terminated: bool = False,
        truncated: bool = False,
        episode_id: int | None = None,
    ) -> None:
        if self.failed:
            return
        started = time.perf_counter()
        record = {
            "step": int(self.step + 1),
            "episode_id": None if episode_id is None else int(episode_id),
            "obs": self._clone_cpu(obs),
            "action": self._clone_cpu(action),
            "reward": float(torch.as_tensor(reward).detach().cpu().item()),
            "next_obs": self._clone_cpu(next_obs),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        try:
            self.step += 1
            self.replay.append(record)
            self.side_agent.observe_transition(
                record["obs"],
                record["action"],
                record["reward"],
                record["next_obs"],
                terminal=bool(record["terminated"]),
            )
            self._identity_observe(record)
            feature = self._build_live_feature(record)
            self.feature_history.append(feature)
            self.last_feature_norm = float(feature.norm().item())
            if len(self.feature_history) >= 1:
                features = torch.stack(tuple(self.feature_history)).to(self.device)
                theta = self.model.context_theta_from_features(features)
                theta_last = theta[-1].detach()
                self.current_theta = theta_last.cpu().clone()
                entropy = -(theta_last * theta_last.clamp_min(1e-08).log()).sum()
                self.last_theta_entropy = float(entropy.cpu().item())
                active_id = None
            self.last_identity_status = "ok"
        except Exception as exc:
            self.last_identity_status = "failed"
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failed = True
        self._wall_ms += (time.perf_counter() - started) * 1000.0

    def update(self) -> dict[str, float | int | str]:
        with self.private_rng_scope():
            return self._update_private()

    def _update_private(self) -> dict[str, float | int | str]:
        if (
            self.failed
            or self.side_buffer is None
            or (not bool(getattr(self.side_buffer, "ready", False)))
            or (
                self.initial_data_warmup_required
                and self.step <= int(getattr(self.control_cfg, "seed_steps", 0))
            )
            or (self.step % self.update_interval != 0)
        ):
            return self.metrics()
        started = time.perf_counter()
        try:
            info = self.side_agent.update(self.side_buffer)
            self.sidecar_updates += 1
            losses = []
            for key, value in dict(info).items():
                if "loss" not in str(key).lower():
                    continue
                try:
                    losses.append(float(torch.as_tensor(value).detach().cpu().item()))
                except (TypeError, ValueError, RuntimeError):
                    continue
            if losses:
                self.last_prediction_loss = float(sum(losses) / len(losses))
            self.last_identity_status = "ok"
        except Exception as exc:
            self.last_identity_status = "failed"
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failed = True
        self._wall_ms += (time.perf_counter() - started) * 1000.0
        return self.metrics()

    def begin_episode_td(self, td) -> None:
        if not self.failed:
            self.side_buffer.begin_episode(td.detach().clone().cpu())

    def append_transition_td(self, td) -> None:
        if not self.failed:
            self.side_buffer.append_transition(td.detach().clone().cpu())

    def end_episode(self) -> None:
        if not self.failed:
            self.side_buffer.end_episode()

    def reset_episode(self) -> None:
        if self.failed:
            return
        self.feature_history.clear()
        self.side_agent.reset_context(clear_memory=False)

    def _proposal_theta(self, mode: str) -> torch.Tensor:
        mode = str(mode).upper()
        theta = self.side_agent._activate_runtime_theta()
        if theta is None:
            reference = torch.zeros(
                1, int(self.side_cfg.latent_dim), device=self.device
            )
            theta = self.model.uniform_theta(reference)[0]
        self.current_theta = theta.detach().cpu().clone()
        theta_cpu = self.current_theta.clamp_min(1e-08)
        self.last_theta_entropy = float(
            (-(theta_cpu * theta_cpu.log()).sum(dim=-1)).mean().item()
        )
        self.proposal_theta_version = int(self.sidecar_updates)
        return theta.detach().to(self.device).clone()

    def _synchronize_proposal_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _profile_sync(self) -> None:
        if self.profile_cuda_sections:
            self._synchronize_proposal_device()

    @torch.inference_mode()
    def _rank_proposals(self, obs, candidates, theta):
        batch_size = int(candidates.shape[1])
        z = self.model.encode(obs, None).repeat(batch_size, 1)
        returns = torch.zeros(batch_size, 1, device=self.device, dtype=z.dtype)
        survival = torch.ones_like(returns)
        discount = 1.0
        episode_length = float(getattr(self.control_cfg, "episode_length", 1000))
        denominator = float(getattr(self.control_cfg, "discount_denom", 5.0))
        fraction = episode_length / denominator
        gamma = min(
            max(
                (fraction - 1.0) / fraction,
                float(getattr(self.control_cfg, "discount_min", 0.95)),
            ),
            float(getattr(self.control_cfg, "discount_max", 0.995)),
        )
        for index in range(int(candidates.shape[0])):
            reward = m3wm_math.two_hot_inv(
                self.model.reward(z, candidates[index], None, theta=theta),
                self.side_cfg,
            )
            returns = returns + discount * survival * reward
            z = self.model.next(z, candidates[index], None, theta=theta)
            discount *= gamma
            if bool(getattr(self.control_cfg, "episodic", False)):
                continuation = torch.sigmoid(
                    -self.model.termination(z, None, unnormalized=True)
                )
                survival = survival * continuation
        actor_mean = torch.tanh(self.model._pi(z).chunk(2, dim=-1)[0])
        q_logits = self.model.Q(z, actor_mean, None, return_type="all", theta=theta)
        q_values = m3wm_math.two_hot_inv(q_logits, self.side_cfg).mean(dim=0)
        return returns + discount * survival * q_values

    @torch.inference_mode()
    def propose(
        self, obs, proposal_mean, proposal_std, *, mode=None, plain_ranker=None
    ):
        self._profile_sync()
        started = time.perf_counter()
        self.proposal_calls += 1
        mode = self.proposal_mode if mode is None else str(mode).upper()
        self.last_proposal_input_copy_sync_ms = 0.0
        self.last_proposal_sampling_ms = 0.0
        self.last_proposal_ranking_ms = 0.0
        self.last_proposal_other_ms = 0.0
        self.last_proposal_mode = mode
        self.last_proposal_ranking_source = "invalid"
        self.last_proposal_selected_index = -1
        self.last_proposal_used_recall = 0
        self.last_proposal_theta_current_linf_delta = float("nan")
        receipt = {
            "action_sequence": None,
            "reason": "invalid_input",
            "proposal_count": self.proposal_count,
            "context_id": -1,
            "theta_version": int(self.proposal_theta_version),
            "model_version": int(self.sidecar_updates),
            "mode": mode,
            "ranking_source": "invalid",
            "selected_index": -1,
            "used_recall": 0,
            "theta_current_linf_delta": float("nan"),
        }
        try:
            if mode not in {"B", "C", "D"}:
                raise ValueError("proposal mode must be B, C, or D")
            stage_started = time.perf_counter()
            mean = (
                torch.as_tensor(proposal_mean, device=self.device, dtype=torch.float32)
                .detach()
                .clone()
            )
            std = (
                torch.as_tensor(proposal_std, device=self.device, dtype=torch.float32)
                .detach()
                .clone()
            )
            expected = (int(self.control_cfg.horizon), int(self.control_cfg.action_dim))
            if tuple(mean.shape) != expected or tuple(std.shape) != expected:
                raise ValueError("proposal distribution shape mismatch")
            input_checks = (
                torch.stack(
                    (
                        torch.isfinite(mean).all(),
                        torch.isfinite(std).all(),
                        (std >= 0).all(),
                    )
                )
                .detach()
                .cpu()
            )
            if not bool(input_checks[0] and input_checks[1]):
                raise ValueError("proposal distribution is non-finite")
            if not bool(input_checks[2]):
                raise ValueError("proposal standard deviation is negative")
            self._profile_sync()
            self.last_proposal_input_copy_sync_ms = (
                time.perf_counter() - stage_started
            ) * 1000.0
            stage_started = time.perf_counter()
            epsilon = torch.randn(
                int(self.control_cfg.horizon),
                self.proposal_count,
                int(self.control_cfg.action_dim),
                device=self.device,
                generator=self.proposal_generator,
            )
            candidates = (mean.unsqueeze(1) + std.unsqueeze(1) * epsilon).clamp(
                -1.0, 1.0
            )
            self._profile_sync()
            self.last_proposal_sampling_ms = (
                time.perf_counter() - stage_started
            ) * 1000.0
            stage_started = time.perf_counter()
            if mode == "B":
                if not callable(plain_ranker):
                    raise ValueError("mode B requires the read-only Plain ranker")
                selected_index = int(plain_ranker(obs, candidates))
                ranking_source = "plain_all_head_maximin"
                theta_current_delta = float("nan")
                used_recall = 0
            else:
                obs_tensor = torch.as_tensor(
                    obs, device=self.device, dtype=torch.float32
                )
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                theta = self._proposal_theta(mode)
                theta_current_delta_tensor = (
                    (theta - self.current_theta.detach().to(self.device)).abs().max()
                )
                used_recall = int(mode == "D" and self._consume_context_return())
                was_training = bool(self.model.training)
                self.model.eval()
                try:
                    values = self._rank_proposals(obs_tensor, candidates, theta)
                finally:
                    self.model.train(was_training)
                ranking_cpu = (
                    torch.cat(
                        (values.squeeze(-1), theta_current_delta_tensor.reshape(1))
                    )
                    .detach()
                    .cpu()
                )
                values_cpu = ranking_cpu[:-1]
                theta_current_delta = float(ranking_cpu[-1].item())
                if not bool(torch.isfinite(values_cpu).all()):
                    raise FloatingPointError("sidecar proposal score is non-finite")
                selected_index = int(values_cpu.argmax().item())
                ranking_source = (
                    "sidecar_recalled_theta" if used_recall else "sidecar_current_theta"
                )
            if not 0 <= selected_index < self.proposal_count:
                raise ValueError("proposal ranker returned an invalid index")
            self._profile_sync()
            self.last_proposal_ranking_ms = (
                time.perf_counter() - stage_started
            ) * 1000.0
            other_started = time.perf_counter()
            selected = candidates[:, selected_index].detach().clone()
            if not bool(torch.isfinite(selected).all()):
                raise FloatingPointError("selected proposal is non-finite")
            receipt["action_sequence"] = selected
            receipt["reason"] = "ok"
            receipt["ranking_source"] = ranking_source
            receipt["selected_index"] = int(selected_index)
            receipt["used_recall"] = int(used_recall)
            receipt["theta_current_linf_delta"] = theta_current_delta
            self.proposal_valid_count += 1
            self.last_proposal_status = "ok"
            self.last_proposal_context_id = int(receipt["context_id"])
            self.last_proposal_ranking_source = ranking_source
            self.last_proposal_selected_index = int(selected_index)
            self.last_proposal_used_recall = int(used_recall)
            self.last_proposal_theta_current_linf_delta = theta_current_delta
            self._profile_sync()
            self.last_proposal_other_ms = (time.perf_counter() - other_started) * 1000.0
        except Exception as exc:
            receipt["reason"] = f"{type(exc).__name__}:{exc}"
            self.last_proposal_status = receipt["reason"]
        self.last_proposal_wall_ms = (time.perf_counter() - started) * 1000.0
        return receipt

    def metrics(self) -> dict[str, float | int | str]:
        observer_metrics = {}
        active_id = int(observer_metrics.get("identity_active_context_id", -1))
        return {
            "identity_status": self.last_identity_status,
            "identity_error": self.last_error,
            "identity_active_id": active_id,
            "identity_active_context_id": active_id,
            "identity_creation_count": int(
                observer_metrics.get("identity_creation_count", 0)
            ),
            "identity_recall_count": int(
                observer_metrics.get("identity_recall_count", 0)
            ),
            "identity_change_proposal_count": int(
                observer_metrics.get("identity_change_proposal_count", 0)
            ),
            "identity_last_action": str(
                observer_metrics.get("identity_last_action", "UNKNOWN")
            ),
            "identity_prediction_loss": float(self.last_prediction_loss),
            "identity_theta_norm": float(self.current_theta.norm().item()),
            "identity_theta_entropy": float(self.last_theta_entropy),
            "identity_feature_norm": float(self.last_feature_norm),
            "identity_sidecar_updates": int(self.sidecar_updates),
            "identity_replay_transitions": int(len(self.replay)),
            "identity_sidecar_wall_ms": float(self._wall_ms),
            "control_reuse_applied": 0,
            "proposal_sidecar_status": self.last_proposal_status,
            "proposal_sidecar_calls": int(self.proposal_calls),
            "proposal_sidecar_valid_count": int(self.proposal_valid_count),
            "proposal_compute_ms": float(self.last_proposal_wall_ms),
            "proposal_input_copy_sync_ms": float(self.last_proposal_input_copy_sync_ms),
            "proposal_sampling_ms": float(self.last_proposal_sampling_ms),
            "proposal_ranking_ms": float(self.last_proposal_ranking_ms),
            "proposal_other_ms": float(self.last_proposal_other_ms),
            "proposal_mode": self.last_proposal_mode,
            "proposal_ranking_source": self.last_proposal_ranking_source,
            "proposal_selected_index": int(self.last_proposal_selected_index),
            "proposal_used_recall": int(self.last_proposal_used_recall),
            "proposal_theta_current_linf_delta": float(
                self.last_proposal_theta_current_linf_delta
            ),
        }

    def export_context_train_state(self):
        if self.side_agent is None:
            raise RuntimeError("Context-agent state requires a full TD-MPC2 sidecar")
        return {
            "schema": "m3wm-context-theta-controller-v1",
            "side_agent": self.side_agent.export_train_state(),
            "side_buffer": self.side_buffer.state_dict(),
            "replay": list(self.replay),
            "feature_history": list(self.feature_history),
            "step": int(self.step),
            "sidecar_updates": int(self.sidecar_updates),
            "last_prediction_loss": float(self.last_prediction_loss),
            "last_identity_status": str(self.last_identity_status),
            "last_error": str(self.last_error),
            "failed": bool(self.failed),
            "current_theta": self.current_theta.detach().clone(),
            "proposal_generator": self.proposal_generator.get_state().clone(),
            "proposal_calls": int(self.proposal_calls),
            "proposal_valid_count": int(self.proposal_valid_count),
            "proposal_theta_version": int(self.proposal_theta_version),
            "private_cpu_rng_state": self._private_cpu_rng_state.clone(),
            "private_cuda_rng_state": (
                None
                if self._private_cuda_rng_state is None
                else self._private_cuda_rng_state.clone()
            ),
            "initial_data_warmup_required": bool(self.initial_data_warmup_required),
        }

    def import_context_train_state(self, state, *, restore_rng=True):
        if state.get("schema") != "m3wm-context-theta-controller-v1":
            raise ValueError("Unsupported theta-controller Context state")
        if self.side_agent is None:
            raise RuntimeError("Target theta controller is not full TD-MPC2")
        self.side_agent.import_train_state(state["side_agent"], restore_rng=restore_rng)
        self.side_buffer.load_state_dict(state["side_buffer"])
        self.replay.clear()
        self.replay.extend(copy.deepcopy(state["replay"]))
        self.feature_history.clear()
        self.feature_history.extend(copy.deepcopy(state["feature_history"]))
        self.step = int(state["step"])
        self.sidecar_updates = int(state["sidecar_updates"])
        self.last_prediction_loss = float(state["last_prediction_loss"])
        self.last_identity_status = str(state["last_identity_status"])
        self.last_error = str(state["last_error"])
        self.failed = bool(state["failed"])
        self.current_theta = state["current_theta"].detach().clone().to(self.device)
        self.proposal_generator.set_state(state["proposal_generator"].clone())
        self.proposal_calls = int(state["proposal_calls"])
        self.proposal_valid_count = int(state["proposal_valid_count"])
        self.proposal_theta_version = int(state["proposal_theta_version"])
        self._private_cpu_rng_state = state["private_cpu_rng_state"].clone()
        private_cuda = state.get("private_cuda_rng_state")
        self._private_cuda_rng_state = (
            None if private_cuda is None else private_cuda.clone()
        )
        self.initial_data_warmup_required = bool(
            state.get("initial_data_warmup_required", True)
        )

    def reset_activation_transients(self) -> None:
        self.feature_history.clear()
        self.last_feature_norm = float("nan")
        reset = getattr(self.side_agent, "reset_context_activation_transients", None)
        if callable(reset):
            reset()

    def finalize(self, work_dir: str | Path | None = None) -> Path | None:
        destination_dir = self.work_dir if work_dir is None else Path(work_dir)
        if destination_dir is None:
            return None
        destination_dir.mkdir(parents=True, exist_ok=True)
        path = destination_dir / "identity_shadow_summary.json"
        payload = {
            "schema": self.schema,
            "version": self.version,
            "sidecar_model": "WorldModel(enable_m3wm=true, hybrid_modular)",
            "control_reuse_applied": False,
            "step": int(self.step),
            "metrics": self.metrics(),
            "replay_capacity": int(self.replay.maxlen or 0),
            "sidecar_parameter_count": int(
                0
                if self.model is None
                else sum((p.numel() for p in self.model.parameters()))
            ),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return path
