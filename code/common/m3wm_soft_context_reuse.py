from __future__ import annotations
from copy import deepcopy
import math
from typing import Any, Mapping, Optional
import torch

SCHEMA = "m3wm_soft_context_reuse"
VERSION = 1


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


class SoftContextReuseController:
    INITIAL_RHO = 0.15
    RHO_STEP = 0.15
    MAX_RHO = 0.85
    NEGATIVE_DECAY = 0.35
    MIN_CONFIRMATIONS = 2

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state = {
            "schema": SCHEMA,
            "version": VERSION,
            "active": False,
            "target_context_id": None,
            "proposal_token": None,
            "rho": 0.0,
            "positive_observations": 0,
            "negative_observations": 0,
            "commit_ready": False,
            "confirmed_active": False,
            "last_benefit": None,
            "last_reason": "inactive",
        }

    @property
    def active(self) -> bool:
        return bool(self._state["active"])

    @property
    def target_context_id(self) -> Optional[int]:
        value = self._state.get("target_context_id")
        return None if value is None else int(value)

    @property
    def rho(self) -> float:
        return float(self._state.get("rho", 0.0))

    @property
    def confirmed_active(self) -> bool:
        return bool(self._state.get("confirmed_active", False))

    @property
    def commit_ready(self) -> bool:
        return bool(self._state.get("commit_ready", False))

    def start(self, target_context_id: int, proposal_token: Any) -> bool:
        if isinstance(target_context_id, bool) or int(target_context_id) < 0:
            return False
        token = repr(proposal_token)
        if token in {"None", "'None'", "'<none>'"}:
            return False
        if (
            self.active
            and self.target_context_id == int(target_context_id)
            and (str(self._state.get("proposal_token")) == token)
        ):
            return False
        self._state.update(
            {
                "active": True,
                "target_context_id": int(target_context_id),
                "proposal_token": token,
                "rho": float(self.INITIAL_RHO),
                "positive_observations": int(self.MIN_CONFIRMATIONS),
                "negative_observations": 0,
                "commit_ready": True,
                "confirmed_active": False,
                "last_benefit": None,
                "last_reason": "soft_reuse_confirmed_ramp",
            }
        )
        return True

    def advance_blend(self, *, reason: str = "bounded_control_ramp") -> dict[str, Any]:
        if not self.active:
            return self.state_dict()
        if self._state.get("last_benefit") is not False:
            self._state["rho"] = min(
                self.MAX_RHO, float(self._state.get("rho", 0.0)) + self.RHO_STEP
            )
        self._state["last_reason"] = str(reason)
        self._state["commit_ready"] = bool(
            int(self._state.get("positive_observations", 0)) >= self.MIN_CONFIRMATIONS
        )
        self._state["confirmed_active"] = bool(
            self._state["commit_ready"]
            and float(self._state.get("rho", 0.0)) >= self.MAX_RHO
        )
        return self.state_dict()

    def cancel(self, reason: str = "soft_reuse_cancelled") -> None:
        target = self.target_context_id
        self.reset()
        self._state["last_reason"] = str(reason)
        self._state["last_target_context_id"] = target

    def commit(self, reason: str = "soft_reuse_confirmed") -> Optional[int]:
        if not self.active or not self.confirmed_active:
            return None
        target = self.target_context_id
        self._state["active"] = False
        self._state["last_reason"] = str(reason)
        self._state["last_target_context_id"] = target
        self._state["rho"] = float(self.MAX_RHO)
        return target

    def advance(
        self, benefit: Optional[bool], *, reason: str = "selector"
    ) -> dict[str, Any]:
        if not self.active:
            return self.state_dict()
        if benefit is not None and (not isinstance(benefit, (bool, torch.Tensor))):
            raise ValueError("soft reuse benefit must be boolean or None")
        if isinstance(benefit, torch.Tensor):
            if benefit.numel() != 1:
                raise ValueError("soft reuse benefit tensor must be scalar")
            benefit = bool(benefit.item())
        self._state["last_benefit"] = None if benefit is None else bool(benefit)
        self._state["last_reason"] = str(reason)
        if benefit is True:
            self._state["positive_observations"] += 1
            self._state["rho"] = min(
                self.MAX_RHO, float(self._state["rho"]) + self.RHO_STEP
            )
        elif benefit is False:
            self._state["negative_observations"] += 1
            self._state["rho"] = max(
                0.0, float(self._state["rho"]) * (1.0 - self.NEGATIVE_DECAY)
            )
        positives = int(self._state["positive_observations"])
        self._state["commit_ready"] = bool(positives >= self.MIN_CONFIRMATIONS)
        self._state["confirmed_active"] = bool(
            self._state["commit_ready"] and float(self._state["rho"]) >= self.MAX_RHO
        )
        if (
            benefit is False
            and int(self._state["negative_observations"]) >= 2
            and (float(self._state["rho"]) <= self.INITIAL_RHO * 0.5)
        ):
            self.cancel("soft_reuse_contrary_evidence")
        return self.state_dict()

    def blend(
        self, current_theta: torch.Tensor, target_theta: torch.Tensor
    ) -> torch.Tensor:
        current = torch.as_tensor(current_theta)
        target = torch.as_tensor(
            target_theta, device=current.device, dtype=current.dtype
        )
        if current.shape != target.shape or current.numel() < 1:
            raise ValueError("soft reuse theta geometry mismatch")
        if current.is_floating_point() and (not bool(torch.isfinite(current).all())):
            raise ValueError("soft reuse current theta is non-finite")
        if target.is_floating_point() and (not bool(torch.isfinite(target).all())):
            raise ValueError("soft reuse target theta is non-finite")
        if bool((current < 0).any()) or bool((target < 0).any()):
            raise ValueError("soft reuse theta must be non-negative")
        if not self.active:
            return current.detach().clone()
        rho = float(max(0.0, min(self.MAX_RHO, self.rho)))
        result = (1.0 - rho) * current + rho * target
        return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-08)

    def rehearsal_weight(self) -> float:
        return float(max(0.0, min(self.MAX_RHO, self.rho)))

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        self.reset()
        if state is None:
            return
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != SCHEMA
            or int(state.get("version", -1)) != VERSION
        ):
            raise ValueError("invalid soft Context reuse state")
        target = state.get("target_context_id")
        if target is not None and (isinstance(target, bool) or int(target) < 0):
            raise ValueError("invalid soft Context reuse target")
        rho = _finite(state.get("rho"))
        if rho is None or rho < 0.0 or rho > self.MAX_RHO:
            raise ValueError("invalid soft Context reuse blend")
        for name in ("active", "commit_ready", "confirmed_active"):
            if not isinstance(state.get(name), bool):
                raise ValueError("invalid soft Context reuse flag")
        for name in ("positive_observations", "negative_observations"):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("invalid soft Context reuse observation count")
        if bool(state.get("active")) and target is None:
            raise ValueError("active soft reuse has no target")
        self._state = deepcopy(dict(state))
        self._state["version"] = VERSION
        self._state["target_context_id"] = None if target is None else int(target)
        self._state["rho"] = float(rho)
