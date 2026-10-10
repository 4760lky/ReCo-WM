from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional
import torch


def linear_action_response_log_norm(
    actions: torch.Tensor, deltas: torch.Tensor, ridge: float = 0.01
) -> torch.Tensor:
    if actions.ndim != 2 or deltas.ndim != 2:
        raise ValueError("actions and deltas must be matrices")
    if actions.shape[0] != deltas.shape[0] or actions.shape[0] < 2:
        raise ValueError("actions and deltas need matching sample counts")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    actions = actions.float()
    deltas = deltas.float()
    actions = actions - actions.mean(dim=0, keepdim=True)
    deltas = deltas - deltas.mean(dim=0, keepdim=True)
    gram = actions.T @ actions
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    response = torch.linalg.solve(gram + ridge * identity, actions.T @ deltas)
    return response.norm().clamp_min(1e-08).log()


def conditioned_action_response_log_norm(
    states: torch.Tensor,
    actions: torch.Tensor,
    deltas: torch.Tensor,
    ridge: float = 0.01,
    min_std: float = 0.001,
) -> torch.Tensor:
    if states.ndim != 2 or actions.ndim != 2 or deltas.ndim != 2:
        raise ValueError("states, actions, and deltas must be matrices")
    if not (
        states.shape[0] == actions.shape[0] == deltas.shape[0] and states.shape[0] >= 2
    ):
        raise ValueError("conditioned system-ID samples must align")
    if ridge <= 0 or min_std <= 0:
        raise ValueError("ridge and min_std must be positive")
    action_response = conditioned_action_response_matrix(
        states, actions, deltas, ridge=ridge, min_std=min_std
    )
    return action_response.norm().clamp_min(1e-08).log()


def conditioned_action_response_matrix(
    states: torch.Tensor,
    actions: torch.Tensor,
    deltas: torch.Tensor,
    ridge: float = 0.01,
    min_std: float = 0.001,
) -> torch.Tensor:
    if states.ndim != 2 or actions.ndim != 2 or deltas.ndim != 2:
        raise ValueError("states, actions, and deltas must be matrices")
    if not (
        states.shape[0] == actions.shape[0] == deltas.shape[0] and states.shape[0] >= 2
    ):
        raise ValueError("conditioned system-ID samples must align")
    if ridge <= 0 or min_std <= 0:
        raise ValueError("ridge and min_std must be positive")
    states = states.float()
    actions = actions.float()
    deltas = deltas.float()
    if not (
        bool(torch.isfinite(states).all())
        and bool(torch.isfinite(actions).all())
        and bool(torch.isfinite(deltas).all())
    ):
        raise ValueError("conditioned system-ID samples must be finite")

    def standardize(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        scale = centered.std(dim=0, unbiased=False).clamp_min(min_std)
        return centered / scale

    state_features = standardize(states)
    action_features = standardize(actions)
    targets = deltas - deltas.mean(dim=0, keepdim=True)
    design = torch.cat([state_features, action_features], dim=-1)
    gram = design.T @ design
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(gram + ridge * identity, design.T @ targets)
    action_response = coefficients[-actions.shape[1] :]
    if not bool(torch.isfinite(action_response).all()):
        raise ValueError("conditioned action response is non-finite")
    return action_response


def conditioned_action_response_distance(
    reference: torch.Tensor, candidate: torch.Tensor, min_scale: float = 1e-08
) -> torch.Tensor:
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("conditioned response signatures must be matrices")
    if reference.shape != candidate.shape or reference.numel() < 1:
        raise ValueError("conditioned response signatures must align")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")
    reference = reference.float()
    candidate = candidate.float()
    if not (
        bool(torch.isfinite(reference).all()) and bool(torch.isfinite(candidate).all())
    ):
        raise ValueError("conditioned response signatures must be finite")
    difference = (candidate - reference).square().mean().sqrt()
    pooled = (
        (0.5 * (reference.square().mean() + candidate.square().mean()))
        .sqrt()
        .clamp_min(float(min_scale))
    )
    result = difference / pooled
    if not bool(torch.isfinite(result)):
        raise ValueError("conditioned response distance is non-finite")
    return result


def batched_conditioned_action_response_log_norm(
    states: torch.Tensor,
    actions: torch.Tensor,
    deltas: torch.Tensor,
    ridge: float = 0.01,
    min_std: float = 0.001,
) -> torch.Tensor:
    if states.ndim != 3 or actions.ndim != 3 or deltas.ndim != 3:
        raise ValueError("states, actions, and deltas must have shape [T,B,D]")
    if not (
        states.shape[:2] == actions.shape[:2] == deltas.shape[:2]
        and states.shape[0] >= 2
    ):
        raise ValueError("conditioned system-ID T,B dimensions must align")
    if ridge <= 0 or min_std <= 0:
        raise ValueError("ridge and min_std must be positive")

    def standardize(values: torch.Tensor) -> torch.Tensor:
        centered = values.float() - values.float().mean(dim=0, keepdim=True)
        scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(min_std)
        return centered / scale

    state_features = standardize(states)
    action_features = standardize(actions)
    targets = deltas.float() - deltas.float().mean(dim=0, keepdim=True)
    design = torch.cat([state_features, action_features], dim=-1)
    gram = torch.einsum("tbd,tbe->bde", design, design)
    identity = torch.eye(
        gram.shape[-1], device=gram.device, dtype=gram.dtype
    ).expand_as(gram)
    rhs = torch.einsum("tbd,tbo->bdo", design, targets)
    coefficients = torch.linalg.solve(gram + ridge * identity, rhs)
    action_response = coefficients[:, -actions.shape[-1] :]
    return action_response.square().sum(dim=(-2, -1)).clamp_min(1e-08).sqrt().log()


def system_id_routing_targets(
    signatures: torch.Tensor,
    num_primitives: int,
    temperature: float = 0.5,
    min_std: float = 0.05,
) -> Optional[torch.Tensor]:
    if signatures.ndim != 1:
        raise ValueError("signatures must be a vector")
    if num_primitives < 2 or temperature <= 0 or min_std <= 0:
        raise ValueError("Invalid routing target parameter")
    std = signatures.std(unbiased=False)
    if not torch.isfinite(std) or float(std) < min_std:
        return None
    normalized = ((signatures - signatures.mean()) / std.clamp_min(min_std)).clamp(
        -2.0, 2.0
    )
    centers = torch.linspace(
        -1.5, 1.5, num_primitives, device=signatures.device, dtype=signatures.dtype
    )
    logits = -(normalized.unsqueeze(-1) - centers).square() / temperature
    return torch.softmax(logits, dim=-1)


def supervised_context_separation_loss(
    theta: torch.Tensor,
    severity: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    label_threshold: float = 0.1,
    margin: float = 0.75,
    min_samples_per_context: int = 2,
    multiclass: bool = False,
    label_tolerance: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if theta.ndim < 2:
        raise ValueError("theta must have shape [batch, ...]")
    if severity.ndim != 1 or severity.shape[0] != theta.shape[0]:
        raise ValueError("severity must be a vector matching theta batch")
    if (
        label_threshold < 0
        or margin <= 0
        or min_samples_per_context < 1
        or (label_tolerance <= 0)
    ):
        raise ValueError("Invalid supervised context loss parameter")
    if valid is None:
        valid = torch.ones_like(severity, dtype=torch.bool)
    elif valid.ndim != 1 or valid.shape != severity.shape:
        raise ValueError("valid must match severity")
    valid = valid.bool() & torch.isfinite(severity)
    features = theta.reshape(theta.shape[0], -1)
    zero = features.new_zeros(())
    if multiclass:
        labels = torch.round(severity / float(label_tolerance)).to(torch.int64)
        groups = []
        for label in torch.unique(labels[valid]):
            mask = valid & (labels == label)
            if int(mask.sum()) >= min_samples_per_context:
                groups.append(features[mask])
        if len(groups) < 2:
            return (zero, zero, zero)
        centers = torch.stack([group.mean(dim=0) for group in groups], dim=0)
        within = torch.stack(
            [
                (group - center).square().sum(dim=-1).mean()
                for (group, center) in zip(groups, centers)
            ]
        ).mean()
        distances = torch.pdist(centers, p=2)
        centroid_distance = distances.min()
        separation = (
            torch.relu(distances.new_tensor(margin) - distances).square().mean()
        )
    else:
        labels = severity > label_threshold
        normal_mask = valid & ~labels
        shifted_mask = valid & labels
        if (
            int(normal_mask.sum()) < min_samples_per_context
            or int(shifted_mask.sum()) < min_samples_per_context
        ):
            return (zero, zero, zero)
        normal = features[normal_mask]
        shifted = features[shifted_mask]
        normal_center = normal.mean(dim=0)
        shifted_center = shifted.mean(dim=0)
        within = 0.5 * (
            (normal - normal_center).square().sum(dim=-1).mean()
            + (shifted - shifted_center).square().sum(dim=-1).mean()
        )
        centroid_distance = torch.linalg.vector_norm(normal_center - shifted_center)
        separation = torch.relu(
            centroid_distance.new_tensor(margin) - centroid_distance
        ).square()
    return (within + separation, within, centroid_distance)


class MultiscaleShiftDetector:

    def __init__(
        self,
        threshold: float = 2.5,
        cooldown: int = 32,
        min_consecutive: int = 1,
        rearm_consecutive: int = 1,
        rearm_after_cooldown: bool = False,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if cooldown < 0:
            raise ValueError("cooldown must be non-negative")
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be positive")
        if rearm_consecutive < 1:
            raise ValueError("rearm_consecutive must be positive")
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self.min_consecutive = int(min_consecutive)
        self.rearm_consecutive = int(rearm_consecutive)
        self.rearm_after_cooldown = bool(rearm_after_cooldown)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.cooldown_remaining = 0
        self.above_threshold_count = 0
        self.below_threshold_count = 0
        self.armed = True
        self.trigger_count = 0
        self.reset_deferred_warmup_pulse()

    def reset_deferred_warmup_pulse(self) -> None:
        self.deferred_warmup_streak_count = 0
        self.deferred_warmup_latched = False
        self.deferred_warmup_consumed = False
        self.deferred_warmup_first = None
        self.deferred_warmup_last = None
        self.deferred_warmup_peak = None
        self.deferred_warmup_promotion = None
        self.deferred_warmup_discard = None

    @staticmethod
    def _deferred_warmup_point(
        *, runtime_step: int, episode_offset: int, score: float
    ) -> Dict[str, Any]:
        if isinstance(runtime_step, bool) or int(runtime_step) < 1:
            raise ValueError("runtime_step must be a positive integer")
        if isinstance(episode_offset, bool) or int(episode_offset) < 0:
            raise ValueError("episode_offset must be non-negative")
        value = float(score)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("deferred warmup score must be finite")
        return {
            "runtime_step": int(runtime_step),
            "env_step": int(runtime_step) - 1,
            "episode_offset": int(episode_offset),
            "score": value,
        }

    def observe_deferred_warmup_pulse(
        self, score: float, *, runtime_step: int, episode_offset: int
    ) -> bool:
        value = float(score)
        if not torch.isfinite(torch.tensor(value)):
            return False
        point = self._deferred_warmup_point(
            runtime_step=runtime_step, episode_offset=episode_offset, score=value
        )
        self.count += 1
        if self.deferred_warmup_consumed or self.deferred_warmup_latched:
            return self.deferred_warmup_latched
        if value <= self.threshold:
            self.deferred_warmup_streak_count = 0
            self.deferred_warmup_first = None
            self.deferred_warmup_last = None
            self.deferred_warmup_peak = None
            return False
        if self.deferred_warmup_streak_count == 0:
            self.deferred_warmup_first = point
            self.deferred_warmup_peak = point
        self.deferred_warmup_streak_count += 1
        self.deferred_warmup_last = point
        if self.deferred_warmup_peak is None or value > float(
            self.deferred_warmup_peak["score"]
        ):
            self.deferred_warmup_peak = point
        if self.deferred_warmup_streak_count >= self.min_consecutive:
            self.deferred_warmup_latched = True
        return self.deferred_warmup_latched

    def deferred_warmup_pulse_ready(self) -> bool:
        return bool(
            self.deferred_warmup_latched
            and (not self.deferred_warmup_consumed)
            and self.armed
            and (self.cooldown_remaining == 0)
        )

    def promote_deferred_warmup_pulse(
        self, score: float, *, runtime_step: int, episode_offset: int
    ) -> bool:
        value = float(score)
        if (
            not torch.isfinite(torch.tensor(value))
            or not self.deferred_warmup_pulse_ready()
        ):
            return False
        current = self._deferred_warmup_point(
            runtime_step=runtime_step, episode_offset=episode_offset, score=value
        )
        first = deepcopy(self.deferred_warmup_first)
        last = deepcopy(self.deferred_warmup_last)
        peak = deepcopy(self.deferred_warmup_peak)
        if first is None or last is None or peak is None:
            raise RuntimeError("latched deferred warmup evidence is incomplete")
        streak_count = int(self.deferred_warmup_streak_count)
        self.count += 1
        self.cooldown_remaining = self.cooldown
        self.above_threshold_count = 0
        self.below_threshold_count = 0
        self.armed = False
        self.trigger_count += 1
        self.deferred_warmup_latched = False
        self.deferred_warmup_consumed = True
        self.deferred_warmup_promotion = {
            "kind": "deferred_warmup_pulse",
            "streak_count": streak_count,
            "min_consecutive": int(self.min_consecutive),
            "threshold": float(self.threshold),
            "first": first,
            "last": last,
            "peak": peak,
            "promotion": current,
            "proposal_score": float(peak["score"]),
            "current_score": value,
        }
        return True

    def discard_deferred_warmup_pulse(
        self, *, reason: str, runtime_step: int, episode_offset: int, score: float
    ) -> None:
        if not self.deferred_warmup_latched:
            return
        current = self._deferred_warmup_point(
            runtime_step=runtime_step, episode_offset=episode_offset, score=score
        )
        self.deferred_warmup_discard = {
            "kind": "deferred_warmup_pulse",
            "reason": str(reason),
            "streak_count": int(self.deferred_warmup_streak_count),
            "min_consecutive": int(self.min_consecutive),
            "threshold": float(self.threshold),
            "first": deepcopy(self.deferred_warmup_first),
            "last": deepcopy(self.deferred_warmup_last),
            "peak": deepcopy(self.deferred_warmup_peak),
            "discard": current,
        }
        self.deferred_warmup_latched = False
        self.deferred_warmup_consumed = True

    def deferred_warmup_promotion_metadata(self) -> Optional[Dict[str, Any]]:
        return deepcopy(self.deferred_warmup_promotion)

    def update(self, score: float, *, allow_trigger: bool = True) -> bool:
        if not isinstance(allow_trigger, bool):
            raise ValueError("allow_trigger must be boolean")
        value = float(score)
        if not torch.isfinite(torch.tensor(value)):
            return False
        self.count += 1
        if not self.armed:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            if self.rearm_after_cooldown and self.cooldown_remaining == 0:
                self.armed = True
                self.below_threshold_count = 0
                self.above_threshold_count = 0
                return False
            if value <= self.threshold:
                self.below_threshold_count += 1
            else:
                self.below_threshold_count = 0
            if (
                self.cooldown_remaining == 0
                and self.below_threshold_count >= self.rearm_consecutive
            ):
                self.armed = True
                self.below_threshold_count = 0
            self.above_threshold_count = 0
            return False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.above_threshold_count = 0
            return False
        if not allow_trigger:
            self.above_threshold_count = 0
            return False
        if value > self.threshold:
            self.above_threshold_count += 1
        else:
            self.above_threshold_count = 0
        triggered = self.above_threshold_count >= self.min_consecutive
        if triggered:
            self.cooldown_remaining = self.cooldown
            self.above_threshold_count = 0
            self.below_threshold_count = 0
            self.armed = False
            self.trigger_count += 1
        return triggered

    def state_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "cooldown_remaining": self.cooldown_remaining,
            "above_threshold_count": self.above_threshold_count,
            "below_threshold_count": self.below_threshold_count,
            "armed": self.armed,
            "rearm_after_cooldown": self.rearm_after_cooldown,
            "trigger_count": self.trigger_count,
            "deferred_warmup_pulse": {
                "streak_count": self.deferred_warmup_streak_count,
                "latched": self.deferred_warmup_latched,
                "consumed": self.deferred_warmup_consumed,
                "first": deepcopy(self.deferred_warmup_first),
                "last": deepcopy(self.deferred_warmup_last),
                "peak": deepcopy(self.deferred_warmup_peak),
                "promotion": deepcopy(self.deferred_warmup_promotion),
                "discard": deepcopy(self.deferred_warmup_discard),
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.count = int(state["count"])
        self.cooldown_remaining = int(state["cooldown_remaining"])
        self.above_threshold_count = int(state.get("above_threshold_count", 0))
        self.below_threshold_count = int(state.get("below_threshold_count", 0))
        self.armed = bool(state.get("armed", True))
        self.rearm_after_cooldown = bool(
            state.get("rearm_after_cooldown", self.rearm_after_cooldown)
        )
        self.trigger_count = int(state["trigger_count"])
        deferred = state.get("deferred_warmup_pulse")
        if deferred is None:
            self.reset_deferred_warmup_pulse()
            return
        if not isinstance(deferred, dict):
            raise ValueError("invalid deferred warmup pulse state")
        required = {
            "streak_count",
            "latched",
            "consumed",
            "first",
            "last",
            "peak",
            "promotion",
            "discard",
        }
        if set(deferred) != required:
            raise ValueError("invalid deferred warmup pulse keys")
        streak_count = deferred["streak_count"]
        latched = deferred["latched"]
        consumed = deferred["consumed"]
        if (
            isinstance(streak_count, bool)
            or not isinstance(streak_count, int)
            or streak_count < 0
            or (not isinstance(latched, bool))
            or (not isinstance(consumed, bool))
            or (latched and consumed)
        ):
            raise ValueError("invalid deferred warmup pulse counters")

        def validate_point(point: Any) -> Optional[Dict[str, Any]]:
            if point is None:
                return None
            if not isinstance(point, dict) or set(point) != {
                "runtime_step",
                "env_step",
                "episode_offset",
                "score",
            }:
                raise ValueError("invalid deferred warmup evidence point")
            canonical = self._deferred_warmup_point(
                runtime_step=point["runtime_step"],
                episode_offset=point["episode_offset"],
                score=point["score"],
            )
            if point["env_step"] != canonical["env_step"]:
                raise ValueError("invalid deferred warmup env coordinate")
            return canonical

        first = validate_point(deferred["first"])
        last = validate_point(deferred["last"])
        peak = validate_point(deferred["peak"])
        if (streak_count == 0) != (first is None and last is None and (peak is None)):
            raise ValueError("incoherent deferred warmup evidence")
        if latched and streak_count < self.min_consecutive:
            raise ValueError("deferred warmup latch lacks consecutive evidence")
        promotion = deferred["promotion"]
        discard = deferred["discard"]
        if promotion is not None and (not isinstance(promotion, dict)):
            raise ValueError("invalid deferred warmup promotion")
        if discard is not None and (not isinstance(discard, dict)):
            raise ValueError("invalid deferred warmup discard")
        self.deferred_warmup_streak_count = streak_count
        self.deferred_warmup_latched = latched
        self.deferred_warmup_consumed = consumed
        self.deferred_warmup_first = first
        self.deferred_warmup_last = last
        self.deferred_warmup_peak = peak
        self.deferred_warmup_promotion = deepcopy(promotion)
        self.deferred_warmup_discard = deepcopy(discard)
