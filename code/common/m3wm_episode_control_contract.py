from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any


def _scalar_bool(value: Any, *, name: str) -> bool:
    if value is None:
        return False
    numel = getattr(value, "numel", None)
    if callable(numel) and int(numel()) != 1:
        raise ValueError(f"{name} must be scalar, received {value!r}")
    size = getattr(value, "size", None)
    if callable(size) and (not isinstance(value, (str, bytes))):
        try:
            shape = tuple(size())
        except TypeError:
            shape = None
        if shape not in (None, (), (1,)):
            raise ValueError(f"{name} must be scalar, received shape {shape}")
    return bool(value)


@dataclass(frozen=True)
class EpisodeControlContract:
    done: bool
    terminated: bool
    truncated: bool
    terminal_observation_is_post_step: bool = True

    def __post_init__(self) -> None:
        done = bool(self.done)
        terminated = bool(self.terminated)
        truncated = bool(self.truncated)
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        if (terminated or truncated) != done:
            raise ValueError(
                f"done must equal terminated OR truncated: done={done}, terminated={terminated}, truncated={truncated}"
            )
        if not isinstance(self.terminal_observation_is_post_step, bool):
            raise TypeError("terminal_observation_is_post_step must be boolean")

    @property
    def bootstrap_allowed(self) -> bool:
        return not self.terminated

    @classmethod
    def from_step(
        cls,
        done: Any,
        info: Mapping[str, Any] | None = None,
        *,
        terminal_observation_is_post_step: bool = True,
    ) -> "EpisodeControlContract":
        info = {} if info is None else info
        done_bool = _scalar_bool(done, name="done")
        explicit_truncated = info.get("truncated", None)
        if explicit_truncated is None:
            explicit_truncated = info.get("TimeLimit.truncated", False)
        truncated = _scalar_bool(explicit_truncated, name="truncated")
        explicit_terminated = info.get("terminated", None)
        if explicit_terminated is None:
            terminated = done_bool and (not truncated)
        else:
            terminated = _scalar_bool(explicit_terminated, name="terminated")
        return cls(
            done=done_bool,
            terminated=terminated,
            truncated=truncated,
            terminal_observation_is_post_step=bool(terminal_observation_is_post_step),
        )

    def as_info(self) -> dict[str, Any]:
        return {
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "bootstrap_allowed": bool(self.bootstrap_allowed),
            "terminal_observation_is_post_step": bool(
                self.terminal_observation_is_post_step
            ),
        }


def control_contract(
    done: Any, info: Mapping[str, Any] | None = None
) -> EpisodeControlContract:
    return EpisodeControlContract.from_step(done, info)
