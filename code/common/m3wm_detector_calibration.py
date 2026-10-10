from __future__ import annotations
from bisect import bisect_left
from collections import defaultdict, deque
import math
from typing import Any, Iterable


class EmpiricalNullCalibrator:
    VERSION = 1

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        min_samples: int = 256,
        max_samples: int = 4096,
        fallback_threshold: float = 1.8,
        update_interval: int = 32,
        context_local_enabled: bool = True,
        robot_baseline_frozen: bool = True,
    ) -> None:
        alpha = float(alpha)
        min_samples = int(min_samples)
        max_samples = int(max_samples)
        fallback_threshold = float(fallback_threshold)
        update_interval = int(update_interval)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if min_samples < 2:
            raise ValueError("min_samples must be at least two")
        if max_samples < min_samples:
            raise ValueError("max_samples must be >= min_samples")
        if not math.isfinite(fallback_threshold) or fallback_threshold <= 0.0:
            raise ValueError("fallback_threshold must be finite and positive")
        if update_interval < 1:
            raise ValueError("update_interval must be positive")
        self.alpha = alpha
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.fallback_threshold = fallback_threshold
        self.update_interval = update_interval
        self.context_local_enabled = bool(context_local_enabled)
        self.robot_baseline_frozen = bool(robot_baseline_frozen)
        self.reset()

    def reset(self) -> None:
        self.robot_samples: deque[float] = deque(maxlen=self.max_samples)
        self.robot_dynamics_samples: deque[float] = deque(maxlen=self.max_samples)
        self.robot_reward_samples: deque[float] = deque(maxlen=self.max_samples)
        self.context_samples: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_samples)
        )
        self.context_dynamics_samples: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_samples)
        )
        self.context_reward_samples: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_samples)
        )
        self.robot_baseline_fitted = False
        self.frozen = False
        self.clean_observation_count = 0
        self.rejected_observation_count = 0
        self._sorted_cache: dict[tuple[str, int | None], list[float]] = {}
        self._dirty_counts: dict[tuple[str, int | None], int] = defaultdict(int)
        self._threshold_cache: dict[tuple[str, int | None], float] = {}
        self._threshold_revision: dict[tuple[str, int | None], int] = {}
        self._threshold_source_cache: dict[int | None, str] = {}

    @staticmethod
    def _finite(value: Any) -> float | None:
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _context_id(value: Any) -> int | None:
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if value >= 0 else None

    def _samples_for(self, name: str, context_id: int | None):
        context_id = self._context_id(context_id)
        if name == "fused":
            return (
                self.robot_samples
                if context_id is None
                else self.context_samples[context_id]
            )
        if name == "dynamics":
            return (
                self.robot_dynamics_samples
                if context_id is None
                else self.context_dynamics_samples[context_id]
            )
        if name == "reward":
            return (
                self.robot_reward_samples
                if context_id is None
                else self.context_reward_samples[context_id]
            )
        raise KeyError(name)

    def _robot_bank_frozen(self) -> bool:
        return bool(
            self.robot_baseline_fitted
            and self.robot_baseline_frozen
            and (len(self.robot_samples) >= self.min_samples)
        )

    def _invalidate(self, name: str, context_id: int | None) -> None:
        key = (name, self._context_id(context_id))
        self._dirty_counts[key] += 1
        if name == "fused":
            try:
                sample_count = len(self._samples_for(name, context_id))
            except (KeyError, TypeError, ValueError):
                sample_count = 0
            if sample_count == self.min_samples:
                self._sorted_cache.pop(key, None)
                self._threshold_cache.pop(key, None)
                self._threshold_revision.pop(key, None)
        if key not in self._sorted_cache:
            self._rebuild_sorted(name, context_id)

    def _rebuild_sorted(self, name: str, context_id: int | None) -> list[float]:
        key = (name, self._context_id(context_id))
        values = sorted((float(v) for v in self._samples_for(name, context_id)))
        self._sorted_cache[key] = values
        self._dirty_counts[key] = 0
        return values

    def _sorted(self, name: str, context_id: int | None) -> list[float]:
        key = (name, self._context_id(context_id))
        values = self._sorted_cache.get(key)
        if values is None or self._dirty_counts.get(key, 0) >= self.update_interval:
            return self._rebuild_sorted(name, context_id)
        return values

    def fit_robot(
        self,
        fused: Iterable[Any],
        *,
        dynamics: Iterable[Any] | None = None,
        reward: Iterable[Any] | None = None,
    ) -> int:
        self.robot_samples.clear()
        self.robot_dynamics_samples.clear()
        self.robot_reward_samples.clear()
        for value in fused:
            value = self._finite(value)
            if value is not None:
                self.robot_samples.append(value)
        if dynamics is not None:
            for value in dynamics:
                value = self._finite(value)
                if value is not None:
                    self.robot_dynamics_samples.append(value)
        if reward is not None:
            for value in reward:
                value = self._finite(value)
                if value is not None:
                    self.robot_reward_samples.append(value)
        self.robot_baseline_fitted = len(self.robot_samples) > 0
        self.frozen = False
        self.clean_observation_count = len(self.robot_samples)
        self.rejected_observation_count = 0
        self._sorted_cache.clear()
        self._dirty_counts.clear()
        self._threshold_cache.clear()
        self._threshold_revision.clear()
        self._threshold_source_cache.clear()
        return len(self.robot_samples)

    def observe(
        self,
        fused: Any,
        *,
        context_id: int | None = None,
        dynamics: Any | None = None,
        reward: Any | None = None,
    ) -> bool:
        if self.frozen:
            self.rejected_observation_count += 1
            return False
        fused_value = self._finite(fused)
        if fused_value is None:
            self.rejected_observation_count += 1
            return False
        if self._robot_bank_frozen():
            pass
        else:
            self.robot_samples.append(fused_value)
            self._invalidate("fused", None)
        context_id = self._context_id(context_id)
        if self.context_local_enabled and context_id is not None:
            self.context_samples[context_id].append(fused_value)
            self._invalidate("fused", context_id)
        for name, value, robot_buffer, context_buffers in (
            (
                "dynamics",
                dynamics,
                self.robot_dynamics_samples,
                self.context_dynamics_samples,
            ),
            ("reward", reward, self.robot_reward_samples, self.context_reward_samples),
        ):
            value = self._finite(value)
            if value is None:
                continue
            if not self._robot_bank_frozen():
                robot_buffer.append(value)
                self._invalidate(name, None)
            if self.context_local_enabled and context_id is not None:
                context_buffers[context_id].append(value)
                self._invalidate(name, context_id)
        self.clean_observation_count += 1
        return True

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def _source_for(self, context_id: int | None) -> tuple[str, int | None]:
        context_id = self._context_id(context_id)
        if (
            self.context_local_enabled
            and context_id is not None
            and (len(self.context_samples[context_id]) >= self.min_samples)
        ):
            return ("context", context_id)
        if len(self.robot_samples) >= self.min_samples:
            return ("robot", None)
        return ("fallback", None)

    def ready(self, context_id: int | None = None) -> bool:
        source, _ = self._source_for(context_id)
        return source != "fallback"

    def sample_count(self, context_id: int | None = None) -> int:
        source, selected = self._source_for(context_id)
        if source == "context":
            return len(self.context_samples[selected])
        return len(self.robot_samples)

    def effective_threshold(self, context_id: int | None = None) -> float:
        source, selected = self._source_for(context_id)
        if source == "fallback":
            self._threshold_source_cache[self._context_id(context_id)] = source
            return self.fallback_threshold
        key = ("fused", selected)
        threshold = self._threshold_cache.get(key)
        revision = self._dirty_counts.get(key, 0)
        cached_revision = self._threshold_revision.get(key, -1)
        if threshold is None or revision - cached_revision >= self.update_interval:
            values = self._sorted("fused", selected)
            revision = self._dirty_counts.get(key, 0)
            n = len(values)
            rank = int(math.ceil((n + 1) * (1.0 - self.alpha)) - 1)
            rank = max(0, min(rank, n - 1))
            threshold = max(self.fallback_threshold, float(values[rank]))
            self._threshold_cache[key] = threshold
            self._threshold_revision[key] = revision
        self._threshold_source_cache[self._context_id(context_id)] = source
        return float(threshold)

    def _metadata_for_channel(
        self, name: str, value: Any, context_id: int | None
    ) -> tuple[float, float, int, str]:
        score = self._finite(value)
        source, selected = self._source_for(context_id)
        if score is None or source == "fallback":
            return (float("nan"), float("nan"), 0, source)
        values = self._sorted(name, selected)
        n = len(values)
        if n == 0:
            return (float("nan"), float("nan"), 0, source)
        tail_count = n - bisect_left(values, score)
        tail = (1.0 + float(tail_count)) / float(n + 1)
        percentile = 1.0 - tail
        return (percentile, tail, n, source)

    def metadata(
        self,
        score: Any,
        *,
        context_id: int | None = None,
        dynamics: Any | None = None,
        reward: Any | None = None,
    ) -> dict[str, Any]:
        threshold = self.effective_threshold(context_id)
        percentile, tail, count, source = self._metadata_for_channel(
            "fused", score, context_id
        )
        dyn_pct, dyn_tail, dyn_count, _ = self._metadata_for_channel(
            "dynamics", dynamics, context_id
        )
        rew_pct, rew_tail, rew_count, _ = self._metadata_for_channel(
            "reward", reward, context_id
        )
        return {
            "mode": "empirical_quantile",
            "alpha": float(self.alpha),
            "fallback_threshold": float(self.fallback_threshold),
            "effective_threshold": float(threshold),
            "source": source,
            "ready": bool(source != "fallback"),
            "sample_count": int(count),
            "context_sample_count": int(
                len(self.context_samples[self._context_id(context_id)])
                if self._context_id(context_id) is not None
                else 0
            ),
            "robot_sample_count": int(len(self.robot_samples)),
            "percentile": float(percentile),
            "tail_probability": float(tail),
            "dynamics_percentile": float(dyn_pct),
            "dynamics_tail_probability": float(dyn_tail),
            "dynamics_sample_count": int(dyn_count),
            "reward_percentile": float(rew_pct),
            "reward_tail_probability": float(rew_tail),
            "reward_sample_count": int(rew_count),
            "frozen": bool(self.frozen),
            "clean_observation_count": int(self.clean_observation_count),
            "rejected_observation_count": int(self.rejected_observation_count),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "m3wm_empirical_null_calibrator",
            "version": self.VERSION,
            "alpha": float(self.alpha),
            "min_samples": int(self.min_samples),
            "max_samples": int(self.max_samples),
            "fallback_threshold": float(self.fallback_threshold),
            "update_interval": int(self.update_interval),
            "context_local_enabled": bool(self.context_local_enabled),
            "robot_baseline_frozen": bool(self.robot_baseline_frozen),
            "robot_samples": list(self.robot_samples),
            "robot_dynamics_samples": list(self.robot_dynamics_samples),
            "robot_reward_samples": list(self.robot_reward_samples),
            "context_samples": {
                str(context_id): list(values)
                for (context_id, values) in self.context_samples.items()
            },
            "context_dynamics_samples": {
                str(context_id): list(values)
                for (context_id, values) in self.context_dynamics_samples.items()
            },
            "context_reward_samples": {
                str(context_id): list(values)
                for (context_id, values) in self.context_reward_samples.items()
            },
            "robot_baseline_fitted": bool(self.robot_baseline_fitted),
            "frozen": bool(self.frozen),
            "clean_observation_count": int(self.clean_observation_count),
            "rejected_observation_count": int(self.rejected_observation_count),
        }

    @staticmethod
    def _restore_values(values: Any) -> list[float]:
        if not isinstance(values, (list, tuple)):
            raise ValueError("invalid empirical-null sample list")
        restored = []
        for value in values:
            value = EmpiricalNullCalibrator._finite(value)
            if value is None:
                raise ValueError("empirical-null samples must be finite")
            restored.append(value)
        return restored

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("invalid empirical-null calibrator state")
        if state.get("schema") != "m3wm_empirical_null_calibrator":
            raise ValueError("invalid empirical-null calibrator schema")
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported empirical-null calibrator version")
        for name in (
            "alpha",
            "min_samples",
            "max_samples",
            "fallback_threshold",
            "update_interval",
            "context_local_enabled",
            "robot_baseline_frozen",
        ):
            expected = getattr(self, name)
            observed = state.get(name)
            if isinstance(expected, float):
                if not math.isclose(
                    float(observed), expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"empirical-null configuration mismatch: {name}")
            elif observed != expected:
                raise ValueError(f"empirical-null configuration mismatch: {name}")
        self.reset()
        self.robot_samples.extend(self._restore_values(state.get("robot_samples", [])))
        self.robot_dynamics_samples.extend(
            self._restore_values(state.get("robot_dynamics_samples", []))
        )
        self.robot_reward_samples.extend(
            self._restore_values(state.get("robot_reward_samples", []))
        )
        for field, target in (
            ("context_samples", self.context_samples),
            ("context_dynamics_samples", self.context_dynamics_samples),
            ("context_reward_samples", self.context_reward_samples),
        ):
            values = state.get(field, {})
            if not isinstance(values, dict):
                raise ValueError(f"invalid empirical-null {field}")
            for raw_context_id, raw_values in values.items():
                context_id = self._context_id(raw_context_id)
                if context_id is None:
                    raise ValueError(
                        f"invalid empirical-null context id: {raw_context_id}"
                    )
                target[context_id].extend(self._restore_values(raw_values))
        self.robot_baseline_fitted = bool(state.get("robot_baseline_fitted", False))
        self.frozen = bool(state.get("frozen", False))
        self.clean_observation_count = int(state.get("clean_observation_count", 0))
        self.rejected_observation_count = int(
            state.get("rejected_observation_count", 0)
        )
        if self.clean_observation_count < 0 or self.rejected_observation_count < 0:
            raise ValueError("invalid empirical-null observation counters")


__all__ = ["EmpiricalNullCalibrator"]
