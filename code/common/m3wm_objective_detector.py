from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass, fields
import math
import secrets
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple
import torch

CONFIRMATION_SAMPLES = 160
BLOCK_COUNT = 4
BLOCK_SIZE = CONFIRMATION_SAMPLES // BLOCK_COUNT
STATE_VERSION = 2


@dataclass(frozen=True)
class ObjectiveDetectorConfig:
    min_stable_samples: int = 240
    stable_learning_rate: float = 0.12
    alternative_learning_rate: float = 0.22
    update_ridge: float = 0.001
    minimum_noise_scale: float = 0.05
    update_clip_sigma: float = 8.0
    score_clip: float = 8.0
    ood_leverage_limit: float = 16.0
    ood_topk_fraction: float = 0.01
    overlap_evaluation_ood_fraction: float = 0.1
    minimum_block_overlap_fraction: float = 0.2
    covariate_shift_score_multiplier: float = 6.0
    maximum_ood_fraction: float = 0.75
    variance_floor: float = 0.0001
    block_mean_score_threshold: float = 2.0
    block_relative_error_reduction: float = 0.08
    block_positive_fraction: float = 0.55

    def __post_init__(self) -> None:
        integer = self.min_stable_samples
        if isinstance(integer, bool) or not isinstance(integer, int) or integer < 80:
            raise ValueError("min_stable_samples must be an integer >= 80")
        positive = (
            "stable_learning_rate",
            "alternative_learning_rate",
            "update_ridge",
            "minimum_noise_scale",
            "update_clip_sigma",
            "score_clip",
            "ood_leverage_limit",
            "variance_floor",
        )
        for name in positive:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.stable_learning_rate > 1.0 or self.alternative_learning_rate > 1.0:
            raise ValueError("normalized-LMS learning rates must be <= 1")
        if (
            isinstance(self.ood_topk_fraction, bool)
            or not math.isfinite(float(self.ood_topk_fraction))
            or (not 0.0 < self.ood_topk_fraction <= 1.0)
        ):
            raise ValueError("ood_topk_fraction must be in (0, 1]")
        if (
            isinstance(self.maximum_ood_fraction, bool)
            or not math.isfinite(float(self.maximum_ood_fraction))
            or (not 0.0 <= self.maximum_ood_fraction < 1.0)
        ):
            raise ValueError("maximum_ood_fraction must be in [0, 1)")
        if (
            isinstance(self.overlap_evaluation_ood_fraction, bool)
            or not math.isfinite(float(self.overlap_evaluation_ood_fraction))
            or (not 0.0 <= self.overlap_evaluation_ood_fraction < 1.0)
            or (self.overlap_evaluation_ood_fraction >= self.maximum_ood_fraction)
        ):
            raise ValueError(
                "overlap_evaluation_ood_fraction must be below maximum_ood_fraction"
            )
        if (
            isinstance(self.minimum_block_overlap_fraction, bool)
            or not math.isfinite(float(self.minimum_block_overlap_fraction))
            or (not 0.0 < self.minimum_block_overlap_fraction <= 1.0)
        ):
            raise ValueError("minimum_block_overlap_fraction must be in (0, 1]")
        if (
            isinstance(self.covariate_shift_score_multiplier, bool)
            or not math.isfinite(float(self.covariate_shift_score_multiplier))
            or self.covariate_shift_score_multiplier < 1.0
        ):
            raise ValueError("covariate_shift_score_multiplier must be >= 1")
        if (
            isinstance(self.block_mean_score_threshold, bool)
            or not math.isfinite(float(self.block_mean_score_threshold))
            or self.block_mean_score_threshold < 0.0
        ):
            raise ValueError("block_mean_score_threshold must be non-negative")
        if (
            isinstance(self.block_relative_error_reduction, bool)
            or not math.isfinite(float(self.block_relative_error_reduction))
            or (not 0.0 <= self.block_relative_error_reduction < 1.0)
        ):
            raise ValueError("block_relative_error_reduction must be in [0, 1)")
        if (
            isinstance(self.block_positive_fraction, bool)
            or not math.isfinite(float(self.block_positive_fraction))
            or (not 0.0 <= self.block_positive_fraction <= 1.0)
        ):
            raise ValueError("block_positive_fraction must be in [0, 1]")


@dataclass(frozen=True)
class ObjectiveChangeDecision:
    changed: bool
    reason: str
    samples: int
    blocks_passed: int
    tail_supported: bool
    block_pass: Tuple[bool, bool, bool, bool]
    block_mean_scores: Tuple[float, float, float, float]
    block_relative_error_reductions: Tuple[float, float, float, float]
    block_positive_fractions: Tuple[float, float, float, float]
    total_score: float
    ood_fraction: float


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return result


def _detached_vector(value: Any, name: str) -> torch.Tensor:
    try:
        if isinstance(value, torch.Tensor):
            vector = value.detach().to(device="cpu", dtype=torch.float64)
        else:
            vector = torch.as_tensor(value, dtype=torch.float64, device="cpu")
        vector = vector.flatten().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if vector.numel() == 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must be a non-empty finite numeric array")
    return vector


def _strict_state_tensor(value: Any, shape: Sequence[int], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"state field {name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError(f"state field {name} must be CPU float64")
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"state field {name} has invalid shape or values")
    return value.detach().clone()


class ProposalLocalObjectiveChangeEvaluator:
    _STATE_KEYS = {
        "version",
        "owner_token",
        "config",
        "phase",
        "input_schema",
        "feature_dim",
        "stable_count",
        "weights",
        "feature_mean",
        "feature_m2",
        "running_scale",
        "proposal_noise_scale",
        "recent_residuals",
        "frozen_weights",
        "alternative_weights",
        "frozen_feature_mean",
        "frozen_feature_variance",
        "proposal_scores",
        "proposal_null_abs_errors",
        "proposal_alt_abs_errors",
        "proposal_ood",
        "decision",
    }

    def __init__(self, config: Optional[ObjectiveDetectorConfig] = None) -> None:
        self.config = config if config is not None else ObjectiveDetectorConfig()
        if not isinstance(self.config, ObjectiveDetectorConfig):
            raise TypeError("config must be ObjectiveDetectorConfig")
        self._owner_token = secrets.token_hex(24)
        self._phase = "building_null"
        self._input_schema: Optional[Tuple[Any, ...]] = None
        self._feature_dim: Optional[int] = None
        self._stable_count = 0
        self._weights: Optional[torch.Tensor] = None
        self._feature_mean: Optional[torch.Tensor] = None
        self._feature_m2: Optional[torch.Tensor] = None
        self._running_scale = self.config.minimum_noise_scale
        self._proposal_noise_scale: Optional[float] = None
        self._recent_residuals: Deque[float] = deque(maxlen=256)
        self._frozen_weights: Optional[torch.Tensor] = None
        self._alternative_weights: Optional[torch.Tensor] = None
        self._frozen_feature_mean: Optional[torch.Tensor] = None
        self._frozen_feature_variance: Optional[torch.Tensor] = None
        self._proposal_scores: list[float] = []
        self._proposal_null_abs_errors: list[float] = []
        self._proposal_alt_abs_errors: list[float] = []
        self._proposal_ood: list[bool] = []
        self._decision: Optional[ObjectiveChangeDecision] = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def stable_samples(self) -> int:
        return self._stable_count

    @property
    def proposal_samples(self) -> int:
        return len(self._proposal_scores)

    @property
    def decision(self) -> Optional[ObjectiveChangeDecision]:
        return self._decision

    @staticmethod
    def _bounded_polynomial_basis(
        linear: torch.Tensor,
        *,
        action_slice: Optional[slice] = None,
        delta_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        bounded = torch.tanh(torch.asinh(linear))
        squared = bounded.square()
        if action_slice is None:
            action_energy = squared.mean().reshape(1)
        else:
            action_energy = squared[action_slice].mean().reshape(1)
        if delta_slice is None:
            transition_energy = squared.mean().reshape(1)
        else:
            transition_energy = squared[delta_slice].mean().reshape(1)
        return torch.cat((bounded, squared, action_energy, transition_energy), dim=0)

    @staticmethod
    def raw_transition_feature(obs: Any, action: Any, next_obs: Any) -> torch.Tensor:
        observation = _detached_vector(obs, "obs")
        control = _detached_vector(action, "action")
        successor = _detached_vector(next_obs, "next_obs")
        if observation.numel() != successor.numel():
            raise ValueError("obs and next_obs must have the same flattened dimension")
        transition = successor - observation
        linear = torch.cat((observation, control, successor, transition), dim=0)
        obs_dim = int(observation.numel())
        action_dim = int(control.numel())
        action_start = obs_dim
        delta_start = 2 * obs_dim + action_dim
        return ProposalLocalObjectiveChangeEvaluator._bounded_polynomial_basis(
            linear,
            action_slice=slice(action_start, action_start + action_dim),
            delta_slice=slice(delta_start, delta_start + obs_dim),
        )

    def _prepare_input(
        self,
        *,
        reward: Any,
        feature: Any = None,
        obs: Any = None,
        action: Any = None,
        next_obs: Any = None,
    ) -> Tuple[torch.Tensor, float]:
        explicit = feature is not None
        raw_any = obs is not None or action is not None or next_obs is not None
        if explicit == raw_any:
            raise ValueError(
                "supply exactly one input form: feature, or obs/action/next_obs"
            )
        if explicit:
            raw_vector = _detached_vector(feature, "feature")
            schema: Tuple[Any, ...] = ("feature_v2", int(raw_vector.numel()))
            vector = self._bounded_polynomial_basis(raw_vector)
        else:
            if obs is None or action is None or next_obs is None:
                raise ValueError("raw input requires obs, action, and next_obs")
            observation = _detached_vector(obs, "obs")
            control = _detached_vector(action, "action")
            successor = _detached_vector(next_obs, "next_obs")
            if observation.numel() != successor.numel():
                raise ValueError("obs and next_obs dimensions differ")
            linear = torch.cat(
                (observation, control, successor, successor - observation), dim=0
            )
            obs_dim = int(observation.numel())
            action_dim = int(control.numel())
            action_start = obs_dim
            delta_start = 2 * obs_dim + action_dim
            vector = self._bounded_polynomial_basis(
                linear,
                action_slice=slice(action_start, action_start + action_dim),
                delta_slice=slice(delta_start, delta_start + obs_dim),
            )
            schema = (
                "raw_v2",
                int(observation.numel()),
                int(control.numel()),
                int(successor.numel()),
            )
        if self._input_schema is None:
            self._input_schema = schema
            self._feature_dim = int(vector.numel())
            self._weights = torch.zeros(self._feature_dim + 1, dtype=torch.float64)
            self._feature_mean = torch.zeros(self._feature_dim, dtype=torch.float64)
            self._feature_m2 = torch.zeros(self._feature_dim, dtype=torch.float64)
        elif schema != self._input_schema:
            raise ValueError("input schema changed within one online evaluator")
        return (vector, _finite_scalar(reward, "reward"))

    @staticmethod
    def _phi(feature: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.ones(1, dtype=torch.float64), feature), dim=0)

    def _normalized_lms_update(
        self,
        weights: torch.Tensor,
        feature: torch.Tensor,
        reward: float,
        learning_rate: float,
        scale: float,
        leverage_weight: float = 1.0,
    ) -> float:
        phi = self._phi(feature)
        prediction = float(torch.dot(weights, phi))
        residual = reward - prediction
        clip = self.config.update_clip_sigma * max(
            scale, self.config.minimum_noise_scale
        )
        clipped_residual = max(-clip, min(clip, residual))
        denominator = self.config.update_ridge + float(torch.dot(phi, phi))
        step = learning_rate * leverage_weight * clipped_residual / denominator
        weights.add_(phi, alpha=step)
        if not bool(torch.isfinite(weights).all()):
            raise FloatingPointError("non-finite objective predictor update")
        return residual

    def observe_stable(
        self,
        *,
        reward: Any,
        feature: Any = None,
        obs: Any = None,
        action: Any = None,
        next_obs: Any = None,
    ) -> None:
        if self._phase != "building_null":
            raise RuntimeError("stable observations are only valid while building null")
        vector, numeric_reward = self._prepare_input(
            reward=reward, feature=feature, obs=obs, action=action, next_obs=next_obs
        )
        assert self._weights is not None
        assert self._feature_mean is not None
        assert self._feature_m2 is not None
        self._stable_count += 1
        delta = vector - self._feature_mean
        self._feature_mean.add_(delta / self._stable_count)
        self._feature_m2.add_(delta * (vector - self._feature_mean))
        residual = self._normalized_lms_update(
            self._weights,
            vector,
            numeric_reward,
            self.config.stable_learning_rate,
            self._running_scale,
        )
        self._recent_residuals.append(float(residual))
        absolute = abs(residual)
        capped = min(
            absolute, 8.0 * max(self._running_scale, self.config.minimum_noise_scale)
        )
        self._running_scale = max(
            self.config.minimum_noise_scale, 0.98 * self._running_scale + 0.02 * capped
        )

    def _snapshot_noise_scale(self) -> float:
        residuals = torch.tensor(tuple(self._recent_residuals), dtype=torch.float64)
        if residuals.numel() < 16:
            return max(self._running_scale, self.config.minimum_noise_scale)
        center = torch.median(residuals)
        mad = float(torch.median(torch.abs(residuals - center)))
        robust = 1.4826 * mad
        return max(robust, self.config.minimum_noise_scale)

    def begin_proposal(self) -> None:
        if self._phase != "building_null":
            raise RuntimeError("a proposal is already active or awaits resolution")
        if self._stable_count < self.config.min_stable_samples:
            raise RuntimeError("insufficient online stable history for a proposal")
        assert self._weights is not None
        assert self._feature_mean is not None
        assert self._feature_m2 is not None
        assert self._feature_dim is not None
        self._frozen_weights = self._weights.detach().clone()
        self._alternative_weights = self._weights.detach().clone()
        self._frozen_feature_mean = self._feature_mean.detach().clone()
        if self._stable_count > 1:
            variance = self._feature_m2 / float(self._stable_count - 1)
        else:
            variance = torch.zeros(self._feature_dim, dtype=torch.float64)
        self._frozen_feature_variance = (
            torch.clamp(variance, min=self.config.variance_floor).detach().clone()
        )
        self._proposal_noise_scale = self._snapshot_noise_scale()
        self._proposal_scores = []
        self._proposal_null_abs_errors = []
        self._proposal_alt_abs_errors = []
        self._proposal_ood = []
        self._decision = None
        self._phase = "proposal"

    def _leverage(self, feature: torch.Tensor) -> Tuple[float, float, bool]:
        assert self._frozen_feature_mean is not None
        assert self._frozen_feature_variance is not None
        standardized = (feature - self._frozen_feature_mean) / torch.sqrt(
            self._frozen_feature_variance
        )
        squared = standardized.square()
        topk = max(1, int(math.ceil(squared.numel() * self.config.ood_topk_fraction)))
        leverage = float(torch.topk(squared, topk, largest=True).values.mean())
        if not math.isfinite(leverage):
            raise FloatingPointError("non-finite feature leverage")
        limit = self.config.ood_leverage_limit
        weight = 1.0 if leverage <= limit else limit / max(leverage, limit)
        return (leverage, weight, leverage > limit)

    def observe_proposal(
        self,
        *,
        reward: Any,
        feature: Any = None,
        obs: Any = None,
        action: Any = None,
        next_obs: Any = None,
    ) -> Optional[ObjectiveChangeDecision]:
        if self._phase != "proposal":
            raise RuntimeError("no active proposal")
        if len(self._proposal_scores) >= CONFIRMATION_SAMPLES:
            raise RuntimeError("proposal already contains 160 samples")
        vector, numeric_reward = self._prepare_input(
            reward=reward, feature=feature, obs=obs, action=action, next_obs=next_obs
        )
        assert self._frozen_weights is not None
        assert self._alternative_weights is not None
        phi = self._phi(vector)
        null_residual = numeric_reward - float(torch.dot(self._frozen_weights, phi))
        alt_residual = numeric_reward - float(torch.dot(self._alternative_weights, phi))
        _, leverage_weight, is_ood = self._leverage(vector)
        assert self._proposal_noise_scale is not None
        scale = max(self._proposal_noise_scale, self.config.minimum_noise_scale)
        clip = self.config.score_clip
        null_z = max(-clip, min(clip, null_residual / scale))
        alt_z = max(-clip, min(clip, alt_residual / scale))
        score = 0.5 * (null_z * null_z - alt_z * alt_z) * leverage_weight
        if not math.isfinite(score):
            raise FloatingPointError("non-finite prequential objective score")
        self._proposal_scores.append(float(score))
        self._proposal_null_abs_errors.append(
            abs(float(null_residual)) * leverage_weight
        )
        self._proposal_alt_abs_errors.append(abs(float(alt_residual)) * leverage_weight)
        self._proposal_ood.append(bool(is_ood))
        self._normalized_lms_update(
            self._alternative_weights,
            vector,
            numeric_reward,
            self.config.alternative_learning_rate,
            scale,
            leverage_weight,
        )
        if len(self._proposal_scores) < CONFIRMATION_SAMPLES:
            return None
        self._decision = self._make_decision()
        self._phase = "complete"
        return self._decision

    def _make_decision(self) -> ObjectiveChangeDecision:
        block_pass = []
        means = []
        reductions = []
        positive_fractions = []
        epsilon = self.config.minimum_noise_scale * 1e-06
        ood_fraction = sum(self._proposal_ood) / CONFIRMATION_SAMPLES
        excessive_ood = ood_fraction > self.config.maximum_ood_fraction
        overlap_mode = bool(
            not excessive_ood
            and ood_fraction > self.config.overlap_evaluation_ood_fraction
        )
        minimum_overlap = int(
            math.ceil(BLOCK_SIZE * self.config.minimum_block_overlap_fraction)
        )
        overlap_sufficient = True
        for block_index in range(BLOCK_COUNT):
            start = block_index * BLOCK_SIZE
            stop = start + BLOCK_SIZE
            indices = list(range(start, stop))
            if overlap_mode:
                indices = [index for index in indices if not self._proposal_ood[index]]
                if len(indices) < minimum_overlap:
                    overlap_sufficient = False
            scores = [self._proposal_scores[index] for index in indices]
            null_errors = [self._proposal_null_abs_errors[index] for index in indices]
            alt_errors = [self._proposal_alt_abs_errors[index] for index in indices]
            support = max(1, len(indices))
            mean_score = sum(scores) / support
            null_mean = sum(null_errors) / support
            alt_mean = sum(alt_errors) / support
            reduction = (null_mean - alt_mean) / max(null_mean, epsilon)
            positive = sum((score > 0.0 for score in scores)) / support
            score_threshold = self.config.block_mean_score_threshold * (
                self.config.covariate_shift_score_multiplier if overlap_mode else 1.0
            )
            passed = (
                (not overlap_mode or len(indices) >= minimum_overlap)
                and mean_score >= score_threshold
                and (reduction >= self.config.block_relative_error_reduction)
                and (positive >= self.config.block_positive_fraction)
            )
            means.append(float(mean_score))
            reductions.append(float(reduction))
            positive_fractions.append(float(positive))
            block_pass.append(bool(passed))
        passed_count = sum(block_pass)
        tail = block_pass[-1]
        changed = bool(
            passed_count >= 3 and tail and (not excessive_ood) and overlap_sufficient
        )
        if excessive_ood:
            reason = "excessive_covariate_shift"
        elif overlap_mode and (not overlap_sufficient):
            reason = "insufficient_covariate_overlap"
        elif changed:
            reason = "persistent_prequential_objective_change"
        elif not tail:
            reason = "missing_tail_support"
        else:
            reason = "insufficient_block_support"
        return ObjectiveChangeDecision(
            changed=changed,
            reason=reason,
            samples=CONFIRMATION_SAMPLES,
            blocks_passed=passed_count,
            tail_supported=tail,
            block_pass=tuple(block_pass),
            block_mean_scores=tuple(means),
            block_relative_error_reductions=tuple(reductions),
            block_positive_fractions=tuple(positive_fractions),
            total_score=float(sum(self._proposal_scores)),
            ood_fraction=ood_fraction,
        )

    def resolve_proposal(
        self, *, adopt_alternative: bool, reset_stable_statistics: bool = False
    ) -> None:
        if self._phase != "complete" or self._decision is None:
            raise RuntimeError("proposal must reach a decision before resolution")
        if not isinstance(adopt_alternative, bool):
            raise TypeError("adopt_alternative must be bool")
        if not isinstance(reset_stable_statistics, bool):
            raise TypeError("reset_stable_statistics must be bool")
        if adopt_alternative:
            assert self._alternative_weights is not None
            self._weights = self._alternative_weights.detach().clone()
        if reset_stable_statistics:
            assert self._feature_mean is not None
            assert self._feature_m2 is not None
            self._stable_count = 0
            self._feature_mean.zero_()
            self._feature_m2.zero_()
            self._recent_residuals.clear()
            self._running_scale = self.config.minimum_noise_scale
        self._frozen_weights = None
        self._alternative_weights = None
        self._frozen_feature_mean = None
        self._frozen_feature_variance = None
        self._proposal_scores = []
        self._proposal_null_abs_errors = []
        self._proposal_alt_abs_errors = []
        self._proposal_ood = []
        self._decision = None
        self._proposal_noise_scale = None
        self._phase = "building_null"

    def cancel_proposal(self) -> None:
        if self._phase not in {"proposal", "complete"}:
            raise RuntimeError("no objective proposal is active")
        self._frozen_weights = None
        self._alternative_weights = None
        self._frozen_feature_mean = None
        self._frozen_feature_variance = None
        self._proposal_scores = []
        self._proposal_null_abs_errors = []
        self._proposal_alt_abs_errors = []
        self._proposal_ood = []
        self._decision = None
        self._proposal_noise_scale = None
        self._phase = "building_null"

    def state_dict(self) -> Dict[str, Any]:

        def clone(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value.detach().clone()

        decision = None if self._decision is None else asdict(self._decision)
        return {
            "version": STATE_VERSION,
            "owner_token": self._owner_token,
            "config": asdict(self.config),
            "phase": self._phase,
            "input_schema": self._input_schema,
            "feature_dim": self._feature_dim,
            "stable_count": self._stable_count,
            "weights": clone(self._weights),
            "feature_mean": clone(self._feature_mean),
            "feature_m2": clone(self._feature_m2),
            "running_scale": float(self._running_scale),
            "proposal_noise_scale": self._proposal_noise_scale,
            "recent_residuals": tuple(self._recent_residuals),
            "frozen_weights": clone(self._frozen_weights),
            "alternative_weights": clone(self._alternative_weights),
            "frozen_feature_mean": clone(self._frozen_feature_mean),
            "frozen_feature_variance": clone(self._frozen_feature_variance),
            "proposal_scores": tuple(self._proposal_scores),
            "proposal_null_abs_errors": tuple(self._proposal_null_abs_errors),
            "proposal_alt_abs_errors": tuple(self._proposal_alt_abs_errors),
            "proposal_ood": tuple(self._proposal_ood),
            "decision": decision,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state.keys()) != self._STATE_KEYS:
            raise ValueError("objective detector state has missing or extra fields")
        if type(state["version"]) is not int or state["version"] != STATE_VERSION:
            raise ValueError("objective detector state version mismatch")
        if state["owner_token"] != self._owner_token:
            raise ValueError("cross-run objective detector state is forbidden")
        if type(state["config"]) is not dict or state["config"] != asdict(self.config):
            raise ValueError("objective detector configuration mismatch")
        phase = state["phase"]
        if type(phase) is not str or phase not in {
            "building_null",
            "proposal",
            "complete",
        }:
            raise ValueError("invalid objective detector phase")
        feature_dim = state["feature_dim"]
        stable_count = state["stable_count"]
        if (
            isinstance(stable_count, bool)
            or not isinstance(stable_count, int)
            or stable_count < 0
        ):
            raise ValueError("invalid stable_count")
        schema = state["input_schema"]
        if feature_dim is None:
            if schema is not None or stable_count != 0:
                raise ValueError("uninitialized state is inconsistent")
            tensor_fields = (
                "weights",
                "feature_mean",
                "feature_m2",
                "frozen_weights",
                "alternative_weights",
                "frozen_feature_mean",
                "frozen_feature_variance",
            )
            if any((state[name] is not None for name in tensor_fields)):
                raise ValueError("uninitialized state contains model tensors")
            weights = mean = m2 = frozen = alternative = frozen_mean = frozen_var = None
        else:
            if (
                isinstance(feature_dim, bool)
                or not isinstance(feature_dim, int)
                or feature_dim <= 0
            ):
                raise ValueError("invalid feature_dim")
            if not isinstance(schema, tuple) or len(schema) < 2:
                raise ValueError("invalid input_schema")
            if schema[0] == "feature_v2":
                if (
                    len(schema) != 2
                    or isinstance(schema[1], bool)
                    or (not isinstance(schema[1], int))
                    or (schema[1] <= 0)
                    or (2 * schema[1] + 2 != feature_dim)
                ):
                    raise ValueError("explicit input_schema does not match feature_dim")
            elif schema[0] == "raw_v2":
                if (
                    len(schema) != 4
                    or any(
                        (
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value <= 0
                            for value in schema[1:]
                        )
                    )
                    or schema[1] != schema[3]
                    or (2 * (3 * schema[1] + schema[2]) + 2 != feature_dim)
                ):
                    raise ValueError("raw input_schema does not match feature_dim")
            else:
                raise ValueError("unknown input_schema kind")
            weights = _strict_state_tensor(
                state["weights"], (feature_dim + 1,), "weights"
            )
            mean = _strict_state_tensor(
                state["feature_mean"], (feature_dim,), "feature_mean"
            )
            m2 = _strict_state_tensor(state["feature_m2"], (feature_dim,), "feature_m2")
            if bool((m2 < 0).any()):
                raise ValueError("feature_m2 must be non-negative")

            def optional_tensor(
                name: str, shape: Sequence[int]
            ) -> Optional[torch.Tensor]:
                value = state[name]
                return (
                    None if value is None else _strict_state_tensor(value, shape, name)
                )

            frozen = optional_tensor("frozen_weights", (feature_dim + 1,))
            alternative = optional_tensor("alternative_weights", (feature_dim + 1,))
            frozen_mean = optional_tensor("frozen_feature_mean", (feature_dim,))
            frozen_var = optional_tensor("frozen_feature_variance", (feature_dim,))
            if frozen_var is not None and bool((frozen_var <= 0).any()):
                raise ValueError("frozen_feature_variance must be positive")
        running_scale = _finite_scalar(state["running_scale"], "running_scale")
        if running_scale < self.config.minimum_noise_scale:
            raise ValueError("running_scale is below configured floor")
        raw_proposal_scale = state["proposal_noise_scale"]
        if raw_proposal_scale is None:
            proposal_scale = None
        else:
            proposal_scale = _finite_scalar(raw_proposal_scale, "proposal_noise_scale")
            if proposal_scale < self.config.minimum_noise_scale:
                raise ValueError("proposal_noise_scale is below configured floor")

        def finite_tuple(name: str, maximum: int) -> Tuple[float, ...]:
            value = state[name]
            if not isinstance(value, tuple) or len(value) > maximum:
                raise ValueError(f"invalid state field {name}")
            return tuple((_finite_scalar(item, name) for item in value))

        recent = finite_tuple("recent_residuals", 256)
        if len(recent) > stable_count:
            raise ValueError("recent residual history exceeds stable_count")
        scores = finite_tuple("proposal_scores", CONFIRMATION_SAMPLES)
        null_errors = finite_tuple("proposal_null_abs_errors", CONFIRMATION_SAMPLES)
        alt_errors = finite_tuple("proposal_alt_abs_errors", CONFIRMATION_SAMPLES)
        if any((value < 0 for value in null_errors + alt_errors)):
            raise ValueError("proposal absolute errors must be non-negative")
        ood = state["proposal_ood"]
        if not isinstance(ood, tuple) or any(
            (type(value) is not bool for value in ood)
        ):
            raise ValueError("proposal_ood must be a tuple of bool")
        if not len(scores) == len(null_errors) == len(alt_errors) == len(ood):
            raise ValueError("proposal evidence lengths differ")
        decision_mapping = state["decision"]
        parsed_decision: Optional[ObjectiveChangeDecision]
        if decision_mapping is None:
            parsed_decision = None
        else:
            decision_keys = {field.name for field in fields(ObjectiveChangeDecision)}
            if (
                type(decision_mapping) is not dict
                or set(decision_mapping.keys()) != decision_keys
            ):
                raise ValueError("decision must be a mapping")
            if type(decision_mapping["changed"]) is not bool:
                raise ValueError("serialized decision changed must be bool")
            if type(decision_mapping["tail_supported"]) is not bool:
                raise ValueError("serialized decision tail_supported must be bool")
            if type(decision_mapping["reason"]) is not str:
                raise ValueError("serialized decision reason must be str")
            if (
                type(decision_mapping["samples"]) is not int
                or decision_mapping["samples"] != CONFIRMATION_SAMPLES
                or type(decision_mapping["blocks_passed"]) is not int
            ):
                raise ValueError("serialized decision has invalid counts")
            raw_block_pass = decision_mapping["block_pass"]
            if (
                not isinstance(raw_block_pass, tuple)
                or len(raw_block_pass) != BLOCK_COUNT
                or any((type(item) is not bool for item in raw_block_pass))
            ):
                raise ValueError("serialized decision has invalid block_pass")
            metric_names = (
                "block_mean_scores",
                "block_relative_error_reductions",
                "block_positive_fractions",
            )
            parsed_metrics = {}
            for name in metric_names:
                raw_values = decision_mapping[name]
                if not isinstance(raw_values, tuple) or len(raw_values) != BLOCK_COUNT:
                    raise ValueError(f"serialized decision has invalid {name}")
                parsed_metrics[name] = tuple(
                    (_finite_scalar(item, f"decision.{name}") for item in raw_values)
                )
            if any(
                (
                    not 0.0 <= value <= 1.0
                    for value in parsed_metrics["block_positive_fractions"]
                )
            ):
                raise ValueError("serialized decision has invalid positive fractions")
            total_score = _finite_scalar(
                decision_mapping["total_score"], "decision.total_score"
            )
            ood_fraction = _finite_scalar(
                decision_mapping["ood_fraction"], "decision.ood_fraction"
            )
            if not 0.0 <= ood_fraction <= 1.0:
                raise ValueError("serialized decision has invalid ood_fraction")
            passed_count = sum(raw_block_pass)
            tail = raw_block_pass[-1]
            excessive_ood = ood_fraction > self.config.maximum_ood_fraction
            overlap_mode = bool(
                not excessive_ood
                and ood_fraction > self.config.overlap_evaluation_ood_fraction
            )
            minimum_overlap = int(
                math.ceil(BLOCK_SIZE * self.config.minimum_block_overlap_fraction)
            )
            overlap_sufficient = bool(
                not overlap_mode
                or all(
                    (
                        sum(
                            (
                                not value
                                for value in ood[
                                    block_index
                                    * BLOCK_SIZE : (block_index + 1)
                                    * BLOCK_SIZE
                                ]
                            )
                        )
                        >= minimum_overlap
                        for block_index in range(BLOCK_COUNT)
                    )
                )
            )
            changed = bool(
                passed_count >= 3
                and tail
                and (not excessive_ood)
                and overlap_sufficient
            )
            expected_reason = (
                "excessive_covariate_shift"
                if excessive_ood
                else (
                    "insufficient_covariate_overlap"
                    if overlap_mode and (not overlap_sufficient)
                    else (
                        "persistent_prequential_objective_change"
                        if changed
                        else (
                            "missing_tail_support"
                            if not tail
                            else "insufficient_block_support"
                        )
                    )
                )
            )
            if (
                decision_mapping["blocks_passed"] != passed_count
                or decision_mapping["tail_supported"] != tail
                or decision_mapping["changed"] != changed
                or (decision_mapping["reason"] != expected_reason)
            ):
                raise ValueError("serialized decision fields are inconsistent")
            try:
                parsed_decision = ObjectiveChangeDecision(
                    changed=changed,
                    reason=expected_reason,
                    samples=CONFIRMATION_SAMPLES,
                    blocks_passed=passed_count,
                    tail_supported=tail,
                    block_pass=raw_block_pass,
                    block_mean_scores=parsed_metrics["block_mean_scores"],
                    block_relative_error_reductions=parsed_metrics[
                        "block_relative_error_reductions"
                    ],
                    block_positive_fractions=parsed_metrics["block_positive_fractions"],
                    total_score=total_score,
                    ood_fraction=ood_fraction,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid serialized decision") from exc
        proposal_models = (frozen, alternative, frozen_mean, frozen_var)
        if phase == "building_null":
            if (
                any((value is not None for value in proposal_models))
                or proposal_scale is not None
                or scores
                or (parsed_decision is not None)
            ):
                raise ValueError("building-null state contains proposal data")
        elif phase == "proposal":
            if stable_count < self.config.min_stable_samples:
                raise ValueError("proposal state lacks sufficient stable history")
            if any((value is None for value in proposal_models)):
                raise ValueError("proposal state lacks frozen models")
            if proposal_scale is None:
                raise ValueError("proposal state lacks frozen noise scale")
            if (
                not 0 <= len(scores) < CONFIRMATION_SAMPLES
                or parsed_decision is not None
            ):
                raise ValueError("active proposal state has invalid evidence")
        else:
            if stable_count < self.config.min_stable_samples:
                raise ValueError("complete state lacks sufficient stable history")
            if any((value is None for value in proposal_models)):
                raise ValueError("complete state lacks proposal models")
            if proposal_scale is None:
                raise ValueError("complete state lacks frozen noise scale")
            if len(scores) != CONFIRMATION_SAMPLES or parsed_decision is None:
                raise ValueError("complete state lacks final evidence/decision")
        self._phase = phase
        self._input_schema = schema
        self._feature_dim = feature_dim
        self._stable_count = stable_count
        self._weights = weights
        self._feature_mean = mean
        self._feature_m2 = m2
        self._running_scale = running_scale
        self._proposal_noise_scale = proposal_scale
        self._recent_residuals = deque(recent, maxlen=256)
        self._frozen_weights = frozen
        self._alternative_weights = alternative
        self._frozen_feature_mean = frozen_mean
        self._frozen_feature_variance = frozen_var
        self._proposal_scores = list(scores)
        self._proposal_null_abs_errors = list(null_errors)
        self._proposal_alt_abs_errors = list(alt_errors)
        self._proposal_ood = list(ood)
        self._decision = parsed_decision


__all__ = [
    "BLOCK_COUNT",
    "BLOCK_SIZE",
    "CONFIRMATION_SAMPLES",
    "ObjectiveChangeDecision",
    "ObjectiveDetectorConfig",
    "ProposalLocalObjectiveChangeEvaluator",
]
