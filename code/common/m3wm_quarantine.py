from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isclose, isfinite
from typing import Any, Dict, Mapping, Optional, Sequence

R3_FEATURE_DIM = 17
FULL_FEATURE_DIM = 19
EVIDENCE_WINDOW = 128
INITIAL_DECISION_DELAY = 160
RETRY_DECISION_DELAY = 288
STATE_VERSION = 4
_ENTRY_MODE_FULL_NOVELTY = "full_novelty"
_ENTRY_MODE_REWARD_ONLY_DISAGREEMENT = "reward_only_disagreement"
_ENTRY_MODE_GROUP_RECALL_CONFLICT = "group_recall_conflict"
_ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE = "unstable_boundary_bridge"
_ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY = "regime_failure_dual_novelty"
_ENTRY_MODES = {
    _ENTRY_MODE_FULL_NOVELTY,
    _ENTRY_MODE_REWARD_ONLY_DISAGREEMENT,
    _ENTRY_MODE_GROUP_RECALL_CONFLICT,
    _ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE,
    _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY,
}
_IDENTITY_ACTIONS = {
    "create_novel",
    "create_ambiguous",
    "recall_confident",
    "recall_legacy",
    "dismiss_active",
    "dismiss_capacity",
    "dismiss_no_slots",
}


def _strict_integer(value: Any, *, minimum: int = 0) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(numeric) or numeric != float(integer) or integer < minimum:
        return None
    return integer


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if isfinite(numeric) else None


@dataclass(frozen=True)
class QuarantineDecision:
    action: str
    reason: str
    transaction_delay: int
    decision_step: Optional[int]
    context_id: Optional[int] = None
    rollback: bool = False
    retry_due_step: Optional[int] = None
    retry_window_start_step: Optional[int] = None
    retry_window_end_step: Optional[int] = None

    @property
    def terminal(self) -> bool:
        return self.action != "quarantine"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["terminal"] = self.terminal
        return result


@dataclass(frozen=True)
class _IdentityEvidence:
    action: str
    best_context_id: Optional[int]
    best_log_posterior: Optional[float]
    new_advantage: float
    recall_advantage: float
    feature_dim: Optional[int]


@dataclass(frozen=True)
class _RewardEvidence:
    persistent: bool
    sample_count: int
    signed_score: float
    energy_score: float


@dataclass(frozen=True)
class _GroupRecallConflictEvidence:
    triggered: bool
    active_context_id: int
    candidate_context_id: int
    fresh_vs_active_advantage: float
    new_context_margin: float
    reward_signed_log_bayes: float
    min_reward_signed_log_bayes: float


@dataclass(frozen=True)
class _UnstableBoundaryBridgeEvidence:
    triggered: bool
    stage: str
    pattern: str
    candidate_context_id: int
    reward_signed_score: float
    reward_energy_score: float
    control_d_z: float
    control_s_post: float
    regime_failure_dual_novelty: bool


@dataclass(frozen=True)
class _ActiveConsensusAmbiguousEvidence:
    triggered: bool
    shadow_best_inactive_context_id: int
    shadow_active_to_inactive_log_gap: float
    reward_signed_score: float
    reward_energy_score: float
    control_d_z: float
    control_s_post: float


_GROUP_RECALL_CONFLICT_SCHEMA = "m3wm_group_recall_conflict"
_GROUP_RECALL_CONFLICT_VERSION = 1
_GROUP_RECALL_CONFLICT_KEYS = {
    "schema",
    "version",
    "valid",
    "triggered",
    "reason",
    "active_context_id",
    "candidate_context_id",
    "full_action",
    "shadow_action",
    "full_best_context_id",
    "shadow_best_context_id",
    "fresh_log_posterior",
    "active_log_posterior",
    "fresh_vs_active_advantage",
    "new_context_margin",
    "reward_signed_fresh_log_likelihood",
    "reward_signed_candidate_log_likelihood",
    "reward_signed_log_bayes",
    "min_reward_signed_log_bayes",
    "reward_evidence_complete",
    "reward_sample_count",
    "reward_signed_score",
    "capacity_available",
}
_UNSTABLE_BOUNDARY_BRIDGE_SCHEMA = "m3wm_unstable_boundary_bridge"
_UNSTABLE_BOUNDARY_BRIDGE_VERSION = 1
_UNSTABLE_BOUNDARY_BRIDGE_PATTERNS = {
    "none",
    "masked_active",
    "dual_novelty",
    "full_novelty_shadow_weak",
    "retry_active_consensus",
    "initial_strong_active_consensus",
}
_UNSTABLE_BOUNDARY_BRIDGE_KEYS = {
    "schema",
    "version",
    "valid",
    "triggered",
    "reason",
    "stage",
    "source",
    "proposal_step",
    "proposal_score",
    "proposal_threshold",
    "active_context_id",
    "candidate_context_id",
    "context_count",
    "capacity_available",
    "frozen_reward_only_context_active",
    "full_action",
    "shadow_action",
    "full_best_context_id",
    "shadow_best_context_id",
    "pattern",
    "reward_persistent",
    "reward_sample_count",
    "reward_signed_score",
    "reward_energy_score",
    "min_abs_reward_signed_score",
    "min_reward_energy_score",
    "control_window_start_step",
    "control_window_end_step",
    "control_sample_count",
    "control_d_z",
    "control_s_post",
    "control_d_z_threshold",
    "control_s_post_threshold",
}
_ACTIVE_CONSENSUS_AMBIGUOUS_SCHEMA = "m3wm_active_consensus_ambiguous"
_ACTIVE_CONSENSUS_AMBIGUOUS_VERSION = 1
_ACTIVE_CONSENSUS_AMBIGUOUS_KEYS = {
    "schema",
    "version",
    "valid",
    "triggered",
    "reason",
    "source",
    "proposal_step",
    "proposal_score",
    "proposal_threshold",
    "active_context_id",
    "context_count",
    "capacity_available",
    "frozen_reward_only_context_active",
    "full_action",
    "shadow_action",
    "full_best_context_id",
    "shadow_best_context_id",
    "shadow_best_inactive_context_id",
    "shadow_active_log_posterior",
    "shadow_best_inactive_log_posterior",
    "shadow_active_to_inactive_log_gap",
    "max_shadow_active_to_inactive_log_gap",
    "reward_persistent",
    "reward_sample_count",
    "reward_signed_score",
    "reward_energy_score",
    "min_abs_reward_signed_score",
    "min_reward_energy_score",
    "control_window_start_step",
    "control_window_end_step",
    "control_sample_count",
    "control_d_z",
    "control_s_post",
    "control_d_z_threshold",
    "control_s_post_threshold",
}
_ACTIVE_CONSENSUS_CONTEXT_STATE_SCHEMA = "m3wm_active_consensus_proposal_context_state"
_ACTIVE_CONSENSUS_CONTEXT_STATE_VERSION = 1
_ACTIVE_CONSENSUS_CONTEXT_STATE_KEYS = {
    "schema",
    "version",
    "proposal_step",
    "active_context_id",
    "active_context_origin",
    "context_count",
    "slot_digests",
    "state_sha256",
}
_CONTEXT_SLOT_DIGEST_KEYS = {
    "schema",
    "version",
    "num_contexts",
    "memory_sha256",
    "slots",
}
_CONTEXT_SLOT_KEYS = {
    "context_id",
    "count",
    "prototype_sha256",
    "theta_sha256",
    "full_slot_sha256",
}


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdef" for character in value))
    )


def _parse_active_consensus_context_state(
    value: Any, *, proposal_step: int
) -> Optional[Dict[str, Any]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _ACTIVE_CONSENSUS_CONTEXT_STATE_KEYS
        or value.get("schema") != _ACTIVE_CONSENSUS_CONTEXT_STATE_SCHEMA
        or (value.get("version") != _ACTIVE_CONSENSUS_CONTEXT_STATE_VERSION)
    ):
        return None
    stored_step = _strict_integer(value.get("proposal_step"), minimum=1)
    active_context_id = _strict_integer(value.get("active_context_id"), minimum=0)
    active_context_origin = value.get("active_context_origin")
    context_count = _strict_integer(value.get("context_count"), minimum=1)
    digests = value.get("slot_digests")
    if (
        stored_step != proposal_step
        or active_context_id is None
        or (not isinstance(active_context_origin, str))
        or (not active_context_origin)
        or (context_count is None)
        or (active_context_id >= context_count)
        or (not isinstance(digests, Mapping))
        or (set(digests) != _CONTEXT_SLOT_DIGEST_KEYS)
        or (digests.get("schema") != "m3wm_context_slot_digests")
        or (digests.get("version") != 1)
        or (_strict_integer(digests.get("num_contexts"), minimum=1) != context_count)
        or (not _is_sha256(digests.get("memory_sha256")))
        or (not isinstance(digests.get("slots"), list))
        or (len(digests["slots"]) != context_count)
        or (not _is_sha256(value.get("state_sha256")))
    ):
        return None
    for expected_context_id, slot in enumerate(digests["slots"]):
        if (
            not isinstance(slot, Mapping)
            or set(slot) != _CONTEXT_SLOT_KEYS
            or _strict_integer(slot.get("context_id"), minimum=0) != expected_context_id
            or (_strict_integer(slot.get("count"), minimum=0) is None)
            or (not _is_sha256(slot.get("prototype_sha256")))
            or (not _is_sha256(slot.get("theta_sha256")))
            or (not _is_sha256(slot.get("full_slot_sha256")))
        ):
            return None
    try:
        serialized = json.dumps(
            {key: value[key] for key in value if key != "state_sha256"},
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return None
    if hashlib.sha256(serialized).hexdigest() != value["state_sha256"]:
        return None
    return {
        "context_count": context_count,
        "active_context_id": active_context_id,
        "active_context_origin": active_context_origin,
    }


def _parse_identity(
    value: Any, *, required_feature_dim: Optional[int] = None
) -> Optional[_IdentityEvidence]:
    if not isinstance(value, Mapping):
        return None
    action = value.get("action")
    if action not in _IDENTITY_ACTIONS:
        return None
    best = value.get("best")
    best_context_id = None
    best_log_posterior = None
    if best is not None:
        if not isinstance(best, Mapping):
            return None
        best_context_id = _strict_integer(best.get("context_id"), minimum=0)
        if best_context_id is None:
            return None
        best_log_posterior = _finite_float(best.get("log_posterior"))
        if best_log_posterior is None:
            return None
    new_advantage = _finite_float(value.get("new_advantage"))
    recall_advantage = _finite_float(value.get("recall_advantage"))
    if new_advantage is None or recall_advantage is None:
        return None
    feature_dim = value.get("feature_dim")
    if feature_dim is not None:
        feature_dim = _strict_integer(feature_dim, minimum=1)
        if feature_dim is None:
            return None
    if required_feature_dim is not None and feature_dim != required_feature_dim:
        return None
    return _IdentityEvidence(
        action=str(action),
        best_context_id=best_context_id,
        best_log_posterior=best_log_posterior,
        new_advantage=new_advantage,
        recall_advantage=recall_advantage,
        feature_dim=feature_dim,
    )


def _parse_reward(value: Any) -> Optional[_RewardEvidence]:
    if not isinstance(value, Mapping):
        return None
    persistent = value.get("persistent")
    if not isinstance(persistent, bool):
        return None
    sample_count = _strict_integer(value.get("sample_count"), minimum=1)
    signed_score = _finite_float(value.get("signed_score"))
    energy_score = _finite_float(value.get("energy_score"))
    if (
        sample_count != EVIDENCE_WINDOW
        or signed_score is None
        or energy_score is None
        or (energy_score < 0.0)
    ):
        return None
    if persistent and (signed_score == 0.0 or energy_score == 0.0):
        return None
    return _RewardEvidence(
        persistent=persistent,
        sample_count=sample_count,
        signed_score=signed_score,
        energy_score=energy_score,
    )


def _parse_group_recall_conflict(
    value: Any,
    *,
    active_context_id: int,
    r3_shadow: _IdentityEvidence,
    full_identity: _IdentityEvidence,
    reward_evidence: _RewardEvidence,
    can_create: bool,
) -> Optional[_GroupRecallConflictEvidence]:
    if not isinstance(value, Mapping) or set(value) != _GROUP_RECALL_CONFLICT_KEYS:
        return None
    if (
        value.get("schema") != _GROUP_RECALL_CONFLICT_SCHEMA
        or value.get("version") != _GROUP_RECALL_CONFLICT_VERSION
        or value.get("valid") is not True
        or (not isinstance(value.get("triggered"), bool))
        or (not isinstance(value.get("reward_evidence_complete"), bool))
        or (not isinstance(value.get("capacity_available"), bool))
    ):
        return None
    integers = {
        name: _strict_integer(value.get(name), minimum=0)
        for name in (
            "active_context_id",
            "candidate_context_id",
            "full_best_context_id",
            "shadow_best_context_id",
            "reward_sample_count",
        )
    }
    numerics = {
        name: _finite_float(value.get(name))
        for name in (
            "fresh_log_posterior",
            "active_log_posterior",
            "fresh_vs_active_advantage",
            "new_context_margin",
            "reward_signed_fresh_log_likelihood",
            "reward_signed_candidate_log_likelihood",
            "reward_signed_log_bayes",
            "min_reward_signed_log_bayes",
            "reward_signed_score",
        )
    }
    if any((item is None for item in integers.values())) or any(
        (item is None for item in numerics.values())
    ):
        return None
    candidate = integers["candidate_context_id"]
    fresh_advantage = numerics["fresh_vs_active_advantage"]
    signed_log_bayes = numerics["reward_signed_log_bayes"]
    margin = numerics["new_context_margin"]
    minimum_bayes = numerics["min_reward_signed_log_bayes"]
    complete = value["reward_evidence_complete"]
    capacity = value["capacity_available"]
    triggered = value["triggered"]
    expected_trigger = bool(
        integers["active_context_id"] == active_context_id
        and candidate != active_context_id
        and (candidate == integers["full_best_context_id"])
        and (integers["full_best_context_id"] == candidate)
        and (integers["shadow_best_context_id"] == candidate)
        and (value.get("full_action") == "recall_confident")
        and (value.get("shadow_action") == "recall_confident")
        and (full_identity.action == "recall_confident")
        and (r3_shadow.action == "recall_confident")
        and (full_identity.best_context_id == candidate)
        and (r3_shadow.best_context_id == candidate)
        and complete
        and (integers["reward_sample_count"] == EVIDENCE_WINDOW)
        and reward_evidence.persistent
        and (reward_evidence.sample_count == EVIDENCE_WINDOW)
        and (numerics["reward_signed_score"] == reward_evidence.signed_score)
        and (capacity == can_create)
        and can_create
        and (margin > 0.0)
        and (minimum_bayes > 0.0)
        and (fresh_advantage > margin)
        and (signed_log_bayes >= minimum_bayes)
    )
    relations_hold = bool(
        integers["active_context_id"] == active_context_id
        and candidate == integers["full_best_context_id"]
        and (value.get("full_action") == full_identity.action)
        and (value.get("shadow_action") == r3_shadow.action)
        and (integers["full_best_context_id"] == full_identity.best_context_id)
        and (integers["shadow_best_context_id"] == r3_shadow.best_context_id)
        and complete
        and (integers["reward_sample_count"] == EVIDENCE_WINDOW)
        and reward_evidence.persistent
        and (reward_evidence.sample_count == EVIDENCE_WINDOW)
        and (numerics["reward_signed_score"] == reward_evidence.signed_score)
        and (capacity == can_create)
        and (margin > 0.0)
        and (minimum_bayes > 0.0)
        and isclose(
            numerics["fresh_log_posterior"] - numerics["active_log_posterior"],
            fresh_advantage,
            rel_tol=1e-06,
            abs_tol=2e-06,
        )
        and isclose(
            numerics["reward_signed_fresh_log_likelihood"]
            - numerics["reward_signed_candidate_log_likelihood"],
            signed_log_bayes,
            rel_tol=1e-06,
            abs_tol=2e-06,
        )
    )
    if not relations_hold or triggered != expected_trigger:
        return None
    expected_reason = (
        "strong_inactive_recall_group_conflict"
        if expected_trigger
        else "group_recall_conflict_not_strong"
    )
    if value.get("reason") != expected_reason:
        return None
    return _GroupRecallConflictEvidence(
        triggered=triggered,
        active_context_id=active_context_id,
        candidate_context_id=candidate,
        fresh_vs_active_advantage=fresh_advantage,
        new_context_margin=margin,
        reward_signed_log_bayes=signed_log_bayes,
        min_reward_signed_log_bayes=minimum_bayes,
    )


def _shadow_active_best_inactive(
    diagnostics: Any, *, active_context_id: int, r3_shadow: _IdentityEvidence
) -> Optional[Dict[str, Any]]:
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("feature_dim") != R3_FEATURE_DIM
        or (not isinstance(diagnostics.get("slots"), Sequence))
        or isinstance(diagnostics.get("slots"), (str, bytes))
    ):
        return None
    parsed = []
    seen = set()
    for slot in diagnostics["slots"]:
        if not isinstance(slot, Mapping):
            return None
        context_id = _strict_integer(slot.get("context_id"), minimum=0)
        log_posterior = _finite_float(slot.get("log_posterior"))
        if context_id is None or log_posterior is None or context_id in seen:
            return None
        seen.add(context_id)
        parsed.append((context_id, log_posterior))
    if len(parsed) < 2 or active_context_id not in seen:
        return None
    ranked = sorted(parsed, key=lambda item: (-item[1], item[0]))
    if (
        ranked[0][0] != active_context_id
        or r3_shadow.best_context_id != active_context_id
        or r3_shadow.best_log_posterior is None
        or (
            not isclose(
                ranked[0][1], r3_shadow.best_log_posterior, rel_tol=1e-07, abs_tol=1e-07
            )
        )
    ):
        return None
    inactive = next((item for item in ranked if item[0] != active_context_id), None)
    if inactive is None:
        return None
    gap = ranked[0][1] - inactive[1]
    if not isfinite(gap) or gap < 0.0:
        return None
    return {
        "context_count": len(parsed),
        "active_log_posterior": ranked[0][1],
        "best_inactive_context_id": inactive[0],
        "best_inactive_log_posterior": inactive[1],
        "active_to_inactive_log_gap": gap,
    }


def _parse_active_consensus_ambiguous(
    value: Any,
    *,
    proposal_step: int,
    decision_step: int,
    active_context_id: int,
    r3_shadow: _IdentityEvidence,
    r3_shadow_diagnostics: Any,
    full_identity: _IdentityEvidence,
    reward_evidence: _RewardEvidence,
    can_create: bool,
    frozen_reward_only_context_active: bool,
    trusted_source: Any,
    trusted_proposal_score: Any,
    trusted_proposal_threshold: Any,
    trusted_control_metrics: Any,
    trusted_proposal_context_state: Any,
    trusted_due_context_state: Any,
    trusted_max_shadow_log_gap: Any,
    trusted_min_abs_reward_signed_score: Any,
    trusted_min_reward_energy_score: Any,
    trusted_control_d_z_threshold: Any,
    trusted_control_s_post_threshold: Any,
) -> Optional[_ActiveConsensusAmbiguousEvidence]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _ACTIVE_CONSENSUS_AMBIGUOUS_KEYS
        or value.get("schema") != _ACTIVE_CONSENSUS_AMBIGUOUS_SCHEMA
        or (value.get("version") != _ACTIVE_CONSENSUS_AMBIGUOUS_VERSION)
        or (value.get("valid") is not True)
        or (not isinstance(value.get("triggered"), bool))
        or (not isinstance(value.get("capacity_available"), bool))
        or (not isinstance(value.get("frozen_reward_only_context_active"), bool))
        or (not isinstance(value.get("reward_persistent"), bool))
    ):
        return None
    integers = {
        name: _strict_integer(value.get(name), minimum=0)
        for name in (
            "proposal_step",
            "active_context_id",
            "context_count",
            "full_best_context_id",
            "shadow_best_context_id",
            "shadow_best_inactive_context_id",
            "reward_sample_count",
            "control_window_start_step",
            "control_window_end_step",
            "control_sample_count",
        )
    }
    numerics = {
        name: _finite_float(value.get(name))
        for name in (
            "proposal_score",
            "proposal_threshold",
            "shadow_active_log_posterior",
            "shadow_best_inactive_log_posterior",
            "shadow_active_to_inactive_log_gap",
            "max_shadow_active_to_inactive_log_gap",
            "reward_signed_score",
            "reward_energy_score",
            "min_abs_reward_signed_score",
            "min_reward_energy_score",
            "control_d_z",
            "control_s_post",
            "control_d_z_threshold",
            "control_s_post_threshold",
        )
    }
    trusted = {
        "max_gap": _finite_float(trusted_max_shadow_log_gap),
        "min_signed": _finite_float(trusted_min_abs_reward_signed_score),
        "min_energy": _finite_float(trusted_min_reward_energy_score),
        "d_z": _finite_float(trusted_control_d_z_threshold),
        "s_post": _finite_float(trusted_control_s_post_threshold),
    }
    trusted_score = _finite_float(trusted_proposal_score)
    trusted_score_threshold = _finite_float(trusted_proposal_threshold)
    trusted_proposal_context = _parse_active_consensus_context_state(
        trusted_proposal_context_state, proposal_step=proposal_step
    )
    trusted_due_context = _parse_active_consensus_context_state(
        trusted_due_context_state, proposal_step=proposal_step
    )
    if (
        not isinstance(trusted_source, str)
        or not trusted_source
        or trusted_score is None
        or (trusted_score_threshold is None)
        or (trusted_score_threshold <= 0.0)
        or (trusted_proposal_context is None)
        or (trusted_due_context is None)
        or (trusted_proposal_context_state != trusted_due_context_state)
        or (trusted_proposal_context["active_context_id"] != active_context_id)
        or (trusted_due_context["active_context_id"] != active_context_id)
        or (not isinstance(trusted_control_metrics, Mapping))
        or (
            set(trusted_control_metrics)
            != {"window_start_step", "window_end_step", "sample_count", "d_z", "s_post"}
        )
    ):
        return None
    trusted_control_integers = {
        name: _strict_integer(trusted_control_metrics.get(name), minimum=0)
        for name in ("window_start_step", "window_end_step", "sample_count")
    }
    trusted_control_numerics = {
        name: _finite_float(trusted_control_metrics.get(name))
        for name in ("d_z", "s_post")
    }
    if (
        any((item is None for item in integers.values()))
        or any((item is None for item in numerics.values()))
        or any((item is None or item <= 0.0 for item in trusted.values()))
        or any((item is None for item in trusted_control_integers.values()))
        or any((item is None for item in trusted_control_numerics.values()))
    ):
        return None
    shadow = _shadow_active_best_inactive(
        r3_shadow_diagnostics, active_context_id=active_context_id, r3_shadow=r3_shadow
    )
    if shadow is None:
        return None
    active = integers["active_context_id"]
    gap = numerics["shadow_active_to_inactive_log_gap"]
    if (
        not isinstance(value.get("source"), str)
        or not value.get("source")
        or value.get("source") != trusted_source
        or (integers["proposal_step"] != proposal_step)
        or (
            not isclose(
                numerics["proposal_score"], trusted_score, rel_tol=1e-12, abs_tol=1e-12
            )
        )
        or (
            not isclose(
                numerics["proposal_threshold"],
                trusted_score_threshold,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        or (active != active_context_id)
        or (integers["context_count"] != shadow["context_count"])
        or (integers["context_count"] != trusted_proposal_context["context_count"])
        or (integers["context_count"] != trusted_due_context["context_count"])
        or (not 0 <= active < integers["context_count"])
        or (not 0 <= integers["full_best_context_id"] < integers["context_count"])
        or (not 0 <= integers["shadow_best_context_id"] < integers["context_count"])
        or (
            not 0
            <= integers["shadow_best_inactive_context_id"]
            < integers["context_count"]
        )
        or (integers["shadow_best_inactive_context_id"] == active)
        or (value.get("full_action") != full_identity.action)
        or (value.get("shadow_action") != r3_shadow.action)
        or (integers["full_best_context_id"] != full_identity.best_context_id)
        or (integers["shadow_best_context_id"] != r3_shadow.best_context_id)
        or (
            integers["shadow_best_inactive_context_id"]
            != shadow["best_inactive_context_id"]
        )
        or (value["capacity_available"] != can_create)
        or (
            value["frozen_reward_only_context_active"]
            != frozen_reward_only_context_active
        )
        or (value["reward_persistent"] != reward_evidence.persistent)
        or (integers["reward_sample_count"] != reward_evidence.sample_count)
        or (numerics["reward_signed_score"] != reward_evidence.signed_score)
        or (numerics["reward_energy_score"] != reward_evidence.energy_score)
        or (numerics["max_shadow_active_to_inactive_log_gap"] != trusted["max_gap"])
        or (numerics["min_abs_reward_signed_score"] != trusted["min_signed"])
        or (numerics["min_reward_energy_score"] != trusted["min_energy"])
        or (numerics["control_d_z_threshold"] != trusted["d_z"])
        or (numerics["control_s_post_threshold"] != trusted["s_post"])
        or (integers["control_window_end_step"] != decision_step - 1)
        or (
            integers["control_window_start_step"]
            != trusted_control_integers["window_start_step"]
        )
        or (
            integers["control_window_end_step"]
            != trusted_control_integers["window_end_step"]
        )
        or (
            integers["control_sample_count"] != trusted_control_integers["sample_count"]
        )
        or (
            not isclose(
                numerics["control_d_z"],
                trusted_control_numerics["d_z"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        or (
            not isclose(
                numerics["control_s_post"],
                trusted_control_numerics["s_post"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        or (integers["control_sample_count"] != EVIDENCE_WINDOW)
        or (
            integers["control_window_end_step"]
            - integers["control_window_start_step"]
            + 1
            != EVIDENCE_WINDOW
        )
        or (
            not isclose(
                numerics["shadow_active_log_posterior"],
                shadow["active_log_posterior"],
                rel_tol=1e-07,
                abs_tol=1e-07,
            )
        )
        or (
            not isclose(
                numerics["shadow_best_inactive_log_posterior"],
                shadow["best_inactive_log_posterior"],
                rel_tol=1e-07,
                abs_tol=1e-07,
            )
        )
        or (
            not isclose(
                gap,
                numerics["shadow_active_log_posterior"]
                - numerics["shadow_best_inactive_log_posterior"],
                rel_tol=1e-07,
                abs_tol=1e-07,
            )
        )
        or (
            not isclose(
                gap, shadow["active_to_inactive_log_gap"], rel_tol=1e-07, abs_tol=1e-07
            )
        )
    ):
        return None
    score = numerics["proposal_score"]
    score_threshold = numerics["proposal_threshold"]
    signed = numerics["reward_signed_score"]
    energy = numerics["reward_energy_score"]
    d_z = numerics["control_d_z"]
    s_post = numerics["control_s_post"]
    expected_trigger = bool(
        can_create
        and (not frozen_reward_only_context_active)
        and (value.get("source") == "multiscale")
        and (full_identity.action == "dismiss_active")
        and (r3_shadow.action == "dismiss_active")
        and (full_identity.best_context_id == active_context_id)
        and (r3_shadow.best_context_id == active_context_id)
        and (score_threshold > 0.0)
        and (score > score_threshold)
        and reward_evidence.persistent
        and (reward_evidence.sample_count == EVIDENCE_WINDOW)
        and (abs(signed) >= trusted["min_signed"])
        and (energy >= trusted["min_energy"])
        and (d_z > trusted["d_z"])
        and (s_post >= trusted["s_post"])
        and (0.0 <= gap <= trusted["max_gap"])
    )
    if value["triggered"] != expected_trigger:
        return None
    expected_reason = (
        "active_consensus_ambiguous_supported"
        if expected_trigger
        else "active_consensus_ambiguous_not_supported"
    )
    if value.get("reason") != expected_reason:
        return None
    return _ActiveConsensusAmbiguousEvidence(
        triggered=expected_trigger,
        shadow_best_inactive_context_id=integers["shadow_best_inactive_context_id"],
        shadow_active_to_inactive_log_gap=gap,
        reward_signed_score=signed,
        reward_energy_score=energy,
        control_d_z=d_z,
        control_s_post=s_post,
    )


def _parse_unstable_boundary_bridge(
    value: Any,
    *,
    stage: str,
    proposal_step: int,
    decision_step: int,
    active_context_id: int,
    r3_shadow: _IdentityEvidence,
    full_identity: _IdentityEvidence,
    reward_evidence: _RewardEvidence,
    can_create: bool,
    frozen_reward_only_context_active: bool,
    trusted_min_abs_reward_signed_score: Any,
    trusted_min_reward_energy_score: Any,
    trusted_control_d_z_threshold: Any,
    trusted_control_s_post_threshold: Any,
    trusted_retry_active_consensus_enabled: Any = False,
    trusted_initial_strong_active_consensus_enabled: Any = False,
    trusted_initial_masked_active_near_energy_enabled: Any = False,
    trusted_initial_masked_active_min_energy_ratio: Any = 1.0,
    trusted_initial_masked_active_min_signed_surplus_ratio: Any = 1.0,
    trusted_initial_dual_novel_regime_failure_enabled: Any = False,
    trusted_initial_dual_novel_min_abs_reward_signed_score: Any = None,
    trusted_initial_dual_novel_min_reward_energy_score: Any = None,
    trusted_regime_failure_dual_novelty_retry: Any = False,
) -> Optional[_UnstableBoundaryBridgeEvidence]:
    if (
        stage not in {"initial", "retry"}
        or not isinstance(value, Mapping)
        or set(value) != _UNSTABLE_BOUNDARY_BRIDGE_KEYS
        or (value.get("schema") != _UNSTABLE_BOUNDARY_BRIDGE_SCHEMA)
        or (value.get("version") != _UNSTABLE_BOUNDARY_BRIDGE_VERSION)
        or (value.get("valid") is not True)
        or (not isinstance(value.get("triggered"), bool))
        or (not isinstance(value.get("capacity_available"), bool))
        or (not isinstance(value.get("frozen_reward_only_context_active"), bool))
        or (not isinstance(value.get("reward_persistent"), bool))
        or (not isinstance(trusted_retry_active_consensus_enabled, bool))
        or (not isinstance(trusted_initial_strong_active_consensus_enabled, bool))
        or (not isinstance(trusted_initial_masked_active_near_energy_enabled, bool))
        or (not isinstance(trusted_initial_dual_novel_regime_failure_enabled, bool))
        or (not isinstance(trusted_regime_failure_dual_novelty_retry, bool))
    ):
        return None
    integers = {
        name: _strict_integer(value.get(name), minimum=0)
        for name in (
            "proposal_step",
            "active_context_id",
            "candidate_context_id",
            "context_count",
            "full_best_context_id",
            "shadow_best_context_id",
            "reward_sample_count",
            "control_window_start_step",
            "control_window_end_step",
            "control_sample_count",
        )
    }
    numerics = {
        name: _finite_float(value.get(name))
        for name in (
            "proposal_score",
            "proposal_threshold",
            "reward_signed_score",
            "reward_energy_score",
            "min_abs_reward_signed_score",
            "min_reward_energy_score",
            "control_d_z",
            "control_s_post",
            "control_d_z_threshold",
            "control_s_post_threshold",
        )
    }
    if any((item is None for item in integers.values())) or any(
        (item is None for item in numerics.values())
    ):
        return None
    trusted_thresholds = {
        "min_abs_reward_signed_score": _finite_float(
            trusted_min_abs_reward_signed_score
        ),
        "min_reward_energy_score": _finite_float(trusted_min_reward_energy_score),
        "control_d_z_threshold": _finite_float(trusted_control_d_z_threshold),
        "control_s_post_threshold": _finite_float(trusted_control_s_post_threshold),
    }
    near_energy_ratio = _finite_float(trusted_initial_masked_active_min_energy_ratio)
    near_signed_surplus_ratio = _finite_float(
        trusted_initial_masked_active_min_signed_surplus_ratio
    )
    dual_novel_min_signed = _finite_float(
        trusted_initial_dual_novel_min_abs_reward_signed_score
    )
    dual_novel_min_energy = _finite_float(
        trusted_initial_dual_novel_min_reward_energy_score
    )
    if any(
        (
            threshold is None or threshold <= 0.0
            for threshold in trusted_thresholds.values()
        )
    ) or (
        near_energy_ratio is None
        or near_signed_surplus_ratio is None
        or (not 0.0 < near_energy_ratio <= 1.0)
        or (near_signed_surplus_ratio < 1.0)
        or (
            (
                trusted_initial_dual_novel_regime_failure_enabled
                or trusted_regime_failure_dual_novelty_retry
            )
            and (
                dual_novel_min_signed is None
                or dual_novel_min_energy is None
                or dual_novel_min_signed <= 0.0
                or (dual_novel_min_energy <= 0.0)
                or (
                    dual_novel_min_signed
                    > trusted_thresholds["min_abs_reward_signed_score"]
                )
                or (
                    dual_novel_min_energy
                    > trusted_thresholds["min_reward_energy_score"]
                )
            )
        )
    ):
        return None
    active = integers["active_context_id"]
    candidate = integers["candidate_context_id"]
    context_count = integers["context_count"]
    full_best = integers["full_best_context_id"]
    shadow_best = integers["shadow_best_context_id"]
    pattern = value.get("pattern")
    masked_active = bool(
        full_identity.action == "dismiss_active"
        and full_best == active
        and (r3_shadow.action == "dismiss_active")
        and (shadow_best != active)
    )
    dual_novelty = bool(
        full_identity.action == "create_novel" and r3_shadow.action == "create_novel"
    )
    initial_dual_novel_regime_failure = bool(
        trusted_initial_dual_novel_regime_failure_enabled
        and stage == "initial"
        and dual_novelty
        and (numerics["control_d_z"] > trusted_thresholds["control_d_z_threshold"])
        and (
            numerics["control_s_post"] >= trusted_thresholds["control_s_post_threshold"]
        )
    )
    retry_dual_novel_regime_failure = bool(
        trusted_regime_failure_dual_novelty_retry and stage == "retry" and dual_novelty
    )
    regime_failure_dual_novelty = bool(
        initial_dual_novel_regime_failure or retry_dual_novel_regime_failure
    )
    strict_min_signed = trusted_thresholds["min_abs_reward_signed_score"]
    strict_min_energy = trusted_thresholds["min_reward_energy_score"]
    use_initial_near_energy = bool(
        trusted_initial_masked_active_near_energy_enabled
        and stage == "initial"
        and masked_active
        and (reward_evidence.energy_score < strict_min_energy)
    )
    effective_min_signed = (
        dual_novel_min_signed
        if regime_failure_dual_novelty
        else (
            strict_min_signed * near_signed_surplus_ratio
            if use_initial_near_energy
            else strict_min_signed
        )
    )
    effective_min_energy = (
        dual_novel_min_energy
        if regime_failure_dual_novelty
        else (
            strict_min_energy * near_energy_ratio
            if use_initial_near_energy
            else strict_min_energy
        )
    )
    if (
        value.get("stage") != stage
        or not isinstance(value.get("source"), str)
        or (not value.get("source"))
        or (pattern not in _UNSTABLE_BOUNDARY_BRIDGE_PATTERNS)
        or (integers["proposal_step"] != proposal_step)
        or (active != active_context_id)
        or (context_count <= 0)
        or (not 0 <= active < context_count)
        or (not 0 <= candidate < context_count)
        or (not 0 <= full_best < context_count)
        or (not 0 <= shadow_best < context_count)
        or (value.get("full_action") != full_identity.action)
        or (value.get("shadow_action") != r3_shadow.action)
        or (full_best != full_identity.best_context_id)
        or (shadow_best != r3_shadow.best_context_id)
        or (value["capacity_available"] != can_create)
        or (
            value["frozen_reward_only_context_active"]
            != frozen_reward_only_context_active
        )
        or (value["reward_persistent"] != reward_evidence.persistent)
        or (integers["reward_sample_count"] != reward_evidence.sample_count)
        or (numerics["reward_signed_score"] != reward_evidence.signed_score)
        or (numerics["reward_energy_score"] != reward_evidence.energy_score)
        or (numerics["min_abs_reward_signed_score"] != effective_min_signed)
        or (numerics["min_reward_energy_score"] != effective_min_energy)
        or (
            numerics["control_d_z_threshold"]
            != trusted_thresholds["control_d_z_threshold"]
        )
        or (
            numerics["control_s_post_threshold"]
            != trusted_thresholds["control_s_post_threshold"]
        )
        or (
            integers["control_window_end_step"]
            != (decision_step - 1 if stage == "initial" else decision_step)
        )
        or (integers["control_sample_count"] != EVIDENCE_WINDOW)
        or (
            integers["control_window_end_step"]
            - integers["control_window_start_step"]
            + 1
            != EVIDENCE_WINDOW
        )
    ):
        return None
    full_novelty_shadow_weak = bool(
        stage == "retry"
        and full_identity.action == "create_novel"
        and (r3_shadow.action == "dismiss_active")
        and (shadow_best != active)
    )
    retry_active_consensus = bool(
        trusted_retry_active_consensus_enabled
        and stage == "retry"
        and (full_identity.action == "dismiss_active")
        and (r3_shadow.action == "dismiss_active")
        and (full_best == active)
        and (shadow_best == active)
    )
    initial_strong_active_consensus = bool(
        trusted_initial_strong_active_consensus_enabled
        and stage == "initial"
        and (context_count >= 2)
        and (full_identity.action == "dismiss_active")
        and (r3_shadow.action == "dismiss_active")
        and (full_best == active)
        and (shadow_best == active)
    )
    expected_pattern = (
        "masked_active"
        if masked_active
        else (
            "dual_novelty"
            if dual_novelty
            else (
                "full_novelty_shadow_weak"
                if full_novelty_shadow_weak
                else (
                    "retry_active_consensus"
                    if retry_active_consensus
                    else (
                        "initial_strong_active_consensus"
                        if initial_strong_active_consensus
                        else "none"
                    )
                )
            )
        )
    )
    expected_candidate = (
        shadow_best if masked_active or full_novelty_shadow_weak else active
    )
    if pattern != expected_pattern or candidate != expected_candidate:
        return None
    score = numerics["proposal_score"]
    score_threshold = numerics["proposal_threshold"]
    signed = numerics["reward_signed_score"]
    energy = numerics["reward_energy_score"]
    min_signed = numerics["min_abs_reward_signed_score"]
    min_energy = numerics["min_reward_energy_score"]
    d_z = numerics["control_d_z"]
    s_post = numerics["control_s_post"]
    d_z_threshold = numerics["control_d_z_threshold"]
    s_post_threshold = numerics["control_s_post_threshold"]
    allowed_pattern = bool(
        pattern in {"masked_active", "dual_novelty", "initial_strong_active_consensus"}
        or (
            stage == "retry"
            and pattern in {"full_novelty_shadow_weak", "retry_active_consensus"}
        )
    )
    initial_reward_admission = bool(abs(signed) >= min_signed or energy >= min_energy)
    strict_reward_confirmation = bool(
        abs(signed) >= min_signed and energy >= min_energy
    )
    expected_trigger = bool(
        can_create
        and (not frozen_reward_only_context_active)
        and (value.get("source") == "multiscale")
        and (score_threshold > 0.0)
        and (score > score_threshold)
        and reward_evidence.persistent
        and (reward_evidence.sample_count == EVIDENCE_WINDOW)
        and (min_signed > 0.0)
        and (min_energy > 0.0)
        and (
            initial_reward_admission
            if initial_dual_novel_regime_failure
            else strict_reward_confirmation
        )
        and (d_z_threshold > 0.0)
        and (s_post_threshold > 0.0)
        and (d_z > d_z_threshold)
        and (
            stage == "retry"
            or s_post >= s_post_threshold
            or initial_dual_novel_regime_failure
        )
        and (pattern != "retry_active_consensus" or s_post >= s_post_threshold)
        and allowed_pattern
    )
    if value["triggered"] != expected_trigger:
        return None
    expected_reason = (
        "unstable_boundary_bridge_supported"
        if expected_trigger
        else "unstable_boundary_bridge_not_supported"
    )
    if value.get("reason") != expected_reason:
        return None
    return _UnstableBoundaryBridgeEvidence(
        triggered=expected_trigger,
        stage=stage,
        pattern=pattern,
        candidate_context_id=candidate,
        reward_signed_score=signed,
        reward_energy_score=energy,
        control_d_z=d_z,
        control_s_post=s_post,
        regime_failure_dual_novelty=regime_failure_dual_novelty,
    )


def _valid_context_id(value: Any) -> Optional[int]:
    return _strict_integer(value, minimum=0)


class M3wmOneShotQuarantine:

    def __init__(self) -> None:
        self._active: Optional[Dict[str, Any]] = None
        self._last_resolved_proposal_step: Optional[int] = None

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def retry_due_step(self) -> Optional[int]:
        return None if self._active is None else self._active["retry_due_step"]

    def reset(self) -> None:
        self._active = None
        self._last_resolved_proposal_step = None

    @staticmethod
    def _window_is_valid(
        start_step: Any, end_step: Any, expected_end_step: int
    ) -> bool:
        start = _strict_integer(start_step, minimum=0)
        end = _strict_integer(end_step, minimum=0)
        return (
            start is not None
            and end is not None
            and (end == expected_end_step)
            and (end - start + 1 == EVIDENCE_WINDOW)
        )

    @staticmethod
    def _decision(
        action: str,
        reason: str,
        delay: int,
        decision_step: Optional[int],
        *,
        context_id: Optional[int] = None,
        rollback: bool = False,
        retry_due_step: Optional[int] = None,
        retry_window_start_step: Optional[int] = None,
        retry_window_end_step: Optional[int] = None,
    ) -> QuarantineDecision:
        return QuarantineDecision(
            action=action,
            reason=reason,
            transaction_delay=delay,
            decision_step=decision_step,
            context_id=context_id,
            rollback=rollback,
            retry_due_step=retry_due_step,
            retry_window_start_step=retry_window_start_step,
            retry_window_end_step=retry_window_end_step,
        )

    def _resolve_initial_dismiss(
        self, reason: str, proposal_step: Optional[int], decision_step: Optional[int]
    ) -> QuarantineDecision:
        if proposal_step is not None:
            self._last_resolved_proposal_step = max(
                proposal_step,
                (
                    -1
                    if self._last_resolved_proposal_step is None
                    else self._last_resolved_proposal_step
                ),
            )
        return self._decision(
            "dismiss", reason, INITIAL_DECISION_DELAY, decision_step, rollback=True
        )

    def initial_decision(
        self,
        *,
        proposal_step: Any,
        decision_step: Any,
        window_start_step: Any,
        window_end_step: Any,
        active_context_id: Any,
        r3_shadow: Mapping[str, Any],
        r3_shadow_diagnostics: Any = None,
        full_identity: Mapping[str, Any],
        reward_evidence: Mapping[str, Any],
        can_create: Any,
        reward_only_disagreement: Any = False,
        group_recall_conflict_enabled: Any = False,
        group_recall_conflict_evidence: Any = None,
        recall_consensus_enabled: Any = False,
        active_consensus_ambiguous_direct_create_enabled: Any = False,
        active_consensus_ambiguous_evidence: Any = None,
        active_consensus_ambiguous_max_shadow_log_gap: Any = None,
        active_consensus_ambiguous_source: Any = None,
        active_consensus_ambiguous_proposal_score: Any = None,
        active_consensus_ambiguous_proposal_threshold: Any = None,
        active_consensus_ambiguous_control_metrics: Any = None,
        active_consensus_ambiguous_proposal_context_state: Any = None,
        active_consensus_ambiguous_due_context_state: Any = None,
        unstable_boundary_bridge_enabled: Any = False,
        unstable_boundary_bridge_evidence: Any = None,
        frozen_reward_only_context_active: Any = False,
        unstable_boundary_bridge_min_abs_signed_score: Any = None,
        unstable_boundary_bridge_min_energy_score: Any = None,
        unstable_boundary_bridge_control_d_z_threshold: Any = None,
        unstable_boundary_bridge_control_s_post_threshold: Any = None,
        unstable_boundary_bridge_initial_strong_active_consensus_enabled: Any = False,
        unstable_boundary_bridge_initial_masked_active_near_energy_enabled: Any = False,
        unstable_boundary_bridge_initial_masked_active_min_energy_ratio: Any = 1.0,
        unstable_boundary_bridge_initial_masked_active_min_signed_surplus_ratio: Any = 1.0,
        unstable_boundary_bridge_initial_dual_novel_regime_failure_enabled: Any = False,
        unstable_boundary_bridge_initial_dual_novel_min_abs_signed_score: Any = None,
        unstable_boundary_bridge_initial_dual_novel_min_energy_score: Any = None,
    ) -> QuarantineDecision:
        proposal = _strict_integer(proposal_step, minimum=0)
        due = _strict_integer(decision_step, minimum=0)
        active_context = _valid_context_id(active_context_id)
        if self._active is not None:
            return self._decision(
                "dismiss",
                "quarantine_already_active",
                INITIAL_DECISION_DELAY,
                due,
                rollback=True,
            )
        if (
            proposal is None
            or due is None
            or due - proposal != INITIAL_DECISION_DELAY
            or (not self._window_is_valid(window_start_step, window_end_step, due))
        ):
            return self._resolve_initial_dismiss(
                "invalid_initial_window", proposal, due
            )
        if (
            self._last_resolved_proposal_step is not None
            and proposal <= self._last_resolved_proposal_step
        ):
            return self._decision(
                "dismiss",
                "proposal_not_monotonic",
                INITIAL_DECISION_DELAY,
                due,
                rollback=True,
            )
        if active_context is None:
            return self._resolve_initial_dismiss(
                "invalid_active_context", proposal, due
            )
        if not isinstance(can_create, bool):
            return self._resolve_initial_dismiss("invalid_capacity_flag", proposal, due)
        if not isinstance(reward_only_disagreement, bool):
            return self._resolve_initial_dismiss(
                "invalid_reward_only_disagreement_flag", proposal, due
            )
        if not isinstance(group_recall_conflict_enabled, bool):
            return self._resolve_initial_dismiss(
                "invalid_group_recall_conflict_enabled_flag", proposal, due
            )
        if not isinstance(recall_consensus_enabled, bool):
            return self._resolve_initial_dismiss(
                "invalid_recall_consensus_enabled_flag", proposal, due
            )
        if not isinstance(active_consensus_ambiguous_direct_create_enabled, bool):
            return self._resolve_initial_dismiss(
                "invalid_active_consensus_ambiguous_enabled_flag", proposal, due
            )
        if not isinstance(unstable_boundary_bridge_enabled, bool):
            return self._resolve_initial_dismiss(
                "invalid_unstable_boundary_bridge_enabled_flag", proposal, due
            )
        if not isinstance(
            unstable_boundary_bridge_initial_strong_active_consensus_enabled, bool
        ):
            return self._resolve_initial_dismiss(
                "invalid_unstable_boundary_bridge_initial_strong_active_consensus_enabled_flag",
                proposal,
                due,
            )
        if (
            unstable_boundary_bridge_initial_strong_active_consensus_enabled
            and active_consensus_ambiguous_direct_create_enabled
        ):
            return self._resolve_initial_dismiss(
                "conflicting_initial_strong_active_consensus_paths", proposal, due
            )
        if not isinstance(
            unstable_boundary_bridge_initial_masked_active_near_energy_enabled, bool
        ):
            return self._resolve_initial_dismiss(
                "invalid_unstable_boundary_bridge_initial_masked_active_near_energy_enabled_flag",
                proposal,
                due,
            )
        if not isinstance(
            unstable_boundary_bridge_initial_dual_novel_regime_failure_enabled, bool
        ):
            return self._resolve_initial_dismiss(
                "invalid_initial_dual_novel_regime_failure_enabled_flag", proposal, due
            )
        near_energy_ratio = _finite_float(
            unstable_boundary_bridge_initial_masked_active_min_energy_ratio
        )
        near_signed_surplus_ratio = _finite_float(
            unstable_boundary_bridge_initial_masked_active_min_signed_surplus_ratio
        )
        if (
            near_energy_ratio is None
            or near_signed_surplus_ratio is None
            or (not 0.0 < near_energy_ratio <= 1.0)
            or (near_signed_surplus_ratio < 1.0)
        ):
            return self._resolve_initial_dismiss(
                "invalid_unstable_boundary_bridge_initial_near_energy_ratios",
                proposal,
                due,
            )
        if not isinstance(frozen_reward_only_context_active, bool):
            return self._resolve_initial_dismiss(
                "invalid_frozen_reward_only_context_active_flag", proposal, due
            )
        r3 = _parse_identity(
            r3_shadow,
            required_feature_dim=(
                R3_FEATURE_DIM
                if group_recall_conflict_enabled
                or recall_consensus_enabled
                or active_consensus_ambiguous_direct_create_enabled
                or unstable_boundary_bridge_enabled
                else None
            ),
        )
        if r3 is None:
            return self._resolve_initial_dismiss(
                "invalid_r3_classification", proposal, due
            )
        full = None
        reward = None
        if (
            unstable_boundary_bridge_enabled
            or active_consensus_ambiguous_direct_create_enabled
        ):
            full = _parse_identity(full_identity, required_feature_dim=FULL_FEATURE_DIM)
            reward = _parse_reward(reward_evidence)
            if full is None or reward is None:
                return self._resolve_initial_dismiss(
                    (
                        "invalid_active_consensus_ambiguous_inputs"
                        if active_consensus_ambiguous_direct_create_enabled
                        else "invalid_unstable_boundary_bridge_inputs"
                    ),
                    proposal,
                    due,
                )
        if active_consensus_ambiguous_direct_create_enabled:
            active_shape = bool(
                full.action == "dismiss_active"
                and full.best_context_id == active_context
                and (r3.action == "dismiss_active")
                and (r3.best_context_id == active_context)
            )
            shadow_gap = (
                _shadow_active_best_inactive(
                    r3_shadow_diagnostics,
                    active_context_id=active_context,
                    r3_shadow=r3,
                )
                if active_shape
                else None
            )
            if shadow_gap is not None:
                ambiguous = _parse_active_consensus_ambiguous(
                    active_consensus_ambiguous_evidence,
                    proposal_step=proposal,
                    decision_step=due,
                    active_context_id=active_context,
                    r3_shadow=r3,
                    r3_shadow_diagnostics=r3_shadow_diagnostics,
                    full_identity=full,
                    reward_evidence=reward,
                    can_create=can_create,
                    frozen_reward_only_context_active=frozen_reward_only_context_active,
                    trusted_source=active_consensus_ambiguous_source,
                    trusted_proposal_score=active_consensus_ambiguous_proposal_score,
                    trusted_proposal_threshold=active_consensus_ambiguous_proposal_threshold,
                    trusted_control_metrics=active_consensus_ambiguous_control_metrics,
                    trusted_proposal_context_state=active_consensus_ambiguous_proposal_context_state,
                    trusted_due_context_state=active_consensus_ambiguous_due_context_state,
                    trusted_max_shadow_log_gap=active_consensus_ambiguous_max_shadow_log_gap,
                    trusted_min_abs_reward_signed_score=unstable_boundary_bridge_min_abs_signed_score,
                    trusted_min_reward_energy_score=unstable_boundary_bridge_min_energy_score,
                    trusted_control_d_z_threshold=unstable_boundary_bridge_control_d_z_threshold,
                    trusted_control_s_post_threshold=unstable_boundary_bridge_control_s_post_threshold,
                )
                if ambiguous is None:
                    return self._resolve_initial_dismiss(
                        "invalid_active_consensus_ambiguous_evidence", proposal, due
                    )
                if ambiguous.triggered:
                    self._last_resolved_proposal_step = proposal
                    return self._decision(
                        "create",
                        "strong_active_consensus_ambiguous_create",
                        INITIAL_DECISION_DELAY,
                        due,
                    )
        if unstable_boundary_bridge_enabled:
            bridge = _parse_unstable_boundary_bridge(
                unstable_boundary_bridge_evidence,
                stage="initial",
                proposal_step=proposal,
                decision_step=due,
                active_context_id=active_context,
                r3_shadow=r3,
                full_identity=full,
                reward_evidence=reward,
                can_create=can_create,
                frozen_reward_only_context_active=frozen_reward_only_context_active,
                trusted_min_abs_reward_signed_score=unstable_boundary_bridge_min_abs_signed_score,
                trusted_min_reward_energy_score=unstable_boundary_bridge_min_energy_score,
                trusted_control_d_z_threshold=unstable_boundary_bridge_control_d_z_threshold,
                trusted_control_s_post_threshold=unstable_boundary_bridge_control_s_post_threshold,
                trusted_initial_strong_active_consensus_enabled=unstable_boundary_bridge_initial_strong_active_consensus_enabled,
                trusted_initial_masked_active_near_energy_enabled=unstable_boundary_bridge_initial_masked_active_near_energy_enabled,
                trusted_initial_masked_active_min_energy_ratio=near_energy_ratio,
                trusted_initial_masked_active_min_signed_surplus_ratio=near_signed_surplus_ratio,
                trusted_initial_dual_novel_regime_failure_enabled=unstable_boundary_bridge_initial_dual_novel_regime_failure_enabled,
                trusted_initial_dual_novel_min_abs_reward_signed_score=unstable_boundary_bridge_initial_dual_novel_min_abs_signed_score,
                trusted_initial_dual_novel_min_reward_energy_score=unstable_boundary_bridge_initial_dual_novel_min_energy_score,
            )
            if bridge is None:
                return self._resolve_initial_dismiss(
                    "invalid_unstable_boundary_bridge_evidence", proposal, due
                )
            if bridge.triggered:
                retry_due = proposal + RETRY_DECISION_DELAY
                self._active = {
                    "proposal_step": proposal,
                    "initial_due_step": due,
                    "retry_due_step": retry_due,
                    "initial_window_start_step": int(window_start_step),
                    "initial_window_end_step": int(window_end_step),
                    "active_context_id": active_context,
                    "initial_best_context_id": bridge.candidate_context_id,
                    "initial_r3_action": r3.action,
                    "initial_full_action": full.action,
                    "initial_reward_signed_score": reward.signed_score,
                    "initial_reward_energy_score": reward.energy_score,
                    "entry_mode": (
                        _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY
                        if bridge.regime_failure_dual_novelty
                        else _ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE
                    ),
                    "initial_bridge_pattern": bridge.pattern,
                    "retry_attempts": 0,
                }
                return self._decision(
                    "quarantine",
                    (
                        "dual_novel_regime_failure_quarantine"
                        if bridge.regime_failure_dual_novelty
                        else "unstable_multiscale_boundary_bridge_quarantine"
                    ),
                    RETRY_DECISION_DELAY,
                    due,
                    retry_due_step=retry_due,
                    retry_window_start_step=due + 1,
                    retry_window_end_step=retry_due,
                )
        if r3.action in {"create_novel", "create_ambiguous"}:
            if not can_create:
                return self._resolve_initial_dismiss(
                    "capacity_unavailable", proposal, due
                )
            self._last_resolved_proposal_step = proposal
            return self._decision(
                "create", "r3_accepted_create", INITIAL_DECISION_DELAY, due
            )
        if r3.action == "recall_confident":
            if r3.best_context_id is None or r3.best_context_id == active_context:
                return self._resolve_initial_dismiss(
                    "invalid_r3_recall_target", proposal, due
                )
            if recall_consensus_enabled or group_recall_conflict_enabled:
                if full is None:
                    full = _parse_identity(
                        full_identity, required_feature_dim=FULL_FEATURE_DIM
                    )
                if full is None:
                    return self._resolve_initial_dismiss(
                        (
                            "invalid_recall_consensus_full_classification"
                            if recall_consensus_enabled
                            else "invalid_group_recall_conflict_inputs"
                        ),
                        proposal,
                        due,
                    )
            if recall_consensus_enabled and (
                not (
                    full.action == "recall_confident"
                    and full.best_context_id == r3.best_context_id
                    and (full.best_context_id != active_context)
                )
            ):
                return self._resolve_initial_dismiss(
                    "recall_consensus_not_met", proposal, due
                )
            if group_recall_conflict_enabled:
                if reward is None:
                    reward = _parse_reward(reward_evidence)
                if reward is None:
                    return self._resolve_initial_dismiss(
                        "invalid_group_recall_conflict_inputs", proposal, due
                    )
                conflict = _parse_group_recall_conflict(
                    group_recall_conflict_evidence,
                    active_context_id=active_context,
                    r3_shadow=r3,
                    full_identity=full,
                    reward_evidence=reward,
                    can_create=can_create,
                )
                if conflict is None:
                    return self._resolve_initial_dismiss(
                        "invalid_group_recall_conflict_evidence", proposal, due
                    )
                if conflict.triggered:
                    retry_due = proposal + RETRY_DECISION_DELAY
                    self._active = {
                        "proposal_step": proposal,
                        "initial_due_step": due,
                        "retry_due_step": retry_due,
                        "initial_window_start_step": int(window_start_step),
                        "initial_window_end_step": int(window_end_step),
                        "active_context_id": active_context,
                        "initial_best_context_id": conflict.candidate_context_id,
                        "initial_r3_action": r3.action,
                        "initial_full_action": full.action,
                        "initial_reward_signed_score": reward.signed_score,
                        "initial_reward_energy_score": reward.energy_score,
                        "entry_mode": _ENTRY_MODE_GROUP_RECALL_CONFLICT,
                        "initial_bridge_pattern": None,
                        "retry_attempts": 0,
                    }
                    return self._decision(
                        "quarantine",
                        "confident_recall_with_reward_signed_group_conflict",
                        RETRY_DECISION_DELAY,
                        due,
                        retry_due_step=retry_due,
                        retry_window_start_step=due + 1,
                        retry_window_end_step=retry_due,
                    )
            self._last_resolved_proposal_step = proposal
            return self._decision(
                "recall",
                "r3_confident_recall",
                INITIAL_DECISION_DELAY,
                due,
                context_id=r3.best_context_id,
            )
        weak_dismiss = (
            r3.action == "dismiss_active"
            and r3.best_context_id is not None
            and (r3.best_context_id != active_context)
        )
        if not weak_dismiss:
            return self._resolve_initial_dismiss("r3_not_weak_dismiss", proposal, due)
        if not can_create:
            return self._resolve_initial_dismiss("capacity_unavailable", proposal, due)
        if full is None:
            full = _parse_identity(full_identity, required_feature_dim=FULL_FEATURE_DIM)
        if full is None:
            return self._resolve_initial_dismiss(
                "invalid_full_classification", proposal, due
            )
        full_novelty = (
            full.action == "create_novel" and full.best_context_id is not None
        )
        if not full_novelty and (not reward_only_disagreement):
            return self._resolve_initial_dismiss(
                "full_identity_not_novel", proposal, due
            )
        if reward is None:
            reward = _parse_reward(reward_evidence)
        if reward is None:
            return self._resolve_initial_dismiss(
                "invalid_reward_evidence", proposal, due
            )
        if not reward.persistent:
            return self._resolve_initial_dismiss(
                "reward_evidence_not_persistent", proposal, due
            )
        retry_due = proposal + RETRY_DECISION_DELAY
        retry_start = due + 1
        retry_end = retry_due
        entry_mode = (
            _ENTRY_MODE_FULL_NOVELTY
            if full_novelty
            else _ENTRY_MODE_REWARD_ONLY_DISAGREEMENT
        )
        initial_best_context_id = (
            full.best_context_id if full_novelty else r3.best_context_id
        )
        self._active = {
            "proposal_step": proposal,
            "initial_due_step": due,
            "retry_due_step": retry_due,
            "initial_window_start_step": int(window_start_step),
            "initial_window_end_step": int(window_end_step),
            "active_context_id": active_context,
            "initial_best_context_id": initial_best_context_id,
            "initial_r3_action": r3.action,
            "initial_full_action": full.action,
            "initial_reward_signed_score": reward.signed_score,
            "initial_reward_energy_score": reward.energy_score,
            "entry_mode": entry_mode,
            "initial_bridge_pattern": None,
            "retry_attempts": 0,
        }
        reason = (
            "weak_dismiss_with_persistent_reward_and_full_novelty"
            if full_novelty
            else "weak_dismiss_with_persistent_reward_only_disagreement"
        )
        return self._decision(
            "quarantine",
            reason,
            RETRY_DECISION_DELAY,
            due,
            retry_due_step=retry_due,
            retry_window_start_step=retry_start,
            retry_window_end_step=retry_end,
        )

    def retry_decision(
        self,
        *,
        proposal_step: Any,
        decision_step: Any,
        window_start_step: Any,
        window_end_step: Any,
        full_identity: Mapping[str, Any],
        reward_evidence: Mapping[str, Any],
        can_create: Any,
        reward_only_disagreement: Any = False,
        r3_shadow: Any = None,
        group_recall_conflict_enabled: Any = False,
        group_recall_conflict_evidence: Any = None,
        group_recall_conflict_dual_novelty_retry_enabled: Any = False,
        group_recall_conflict_dual_ambiguous_retry_enabled: Any = False,
        group_recall_conflict_absolute_novelty_retry_enabled: Any = False,
        retry_transaction_state_coherent: Any = False,
        recall_consensus_enabled: Any = False,
        unstable_boundary_bridge_enabled: Any = False,
        unstable_boundary_bridge_evidence: Any = None,
        unstable_boundary_bridge_transaction_state_coherent: Any = False,
        unstable_boundary_bridge_retry_active_consensus_enabled: Any = False,
        unstable_boundary_bridge_initial_strong_active_consensus_enabled: Any = False,
        frozen_reward_only_context_active: Any = False,
        unstable_boundary_bridge_min_abs_signed_score: Any = None,
        unstable_boundary_bridge_min_energy_score: Any = None,
        unstable_boundary_bridge_control_d_z_threshold: Any = None,
        unstable_boundary_bridge_control_s_post_threshold: Any = None,
        unstable_boundary_bridge_regime_failure_dual_novel_min_abs_signed_score: Any = None,
        unstable_boundary_bridge_regime_failure_dual_novel_min_energy_score: Any = None,
    ) -> QuarantineDecision:
        proposal = _strict_integer(proposal_step, minimum=0)
        due = _strict_integer(decision_step, minimum=0)
        if self._active is None:
            return self._decision(
                "dismiss",
                "no_active_quarantine",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        active = dict(self._active)
        self._active = None
        stored_proposal = active["proposal_step"]
        self._last_resolved_proposal_step = stored_proposal
        if (
            proposal != stored_proposal
            or due != active["retry_due_step"]
            or due is None
            or (due - stored_proposal != RETRY_DECISION_DELAY)
            or (
                not self._window_is_valid(
                    window_start_step, window_end_step, active["retry_due_step"]
                )
            )
            or (
                _strict_integer(window_start_step, minimum=0)
                != active["initial_window_end_step"] + 1
            )
        ):
            return self._decision(
                "dismiss",
                "invalid_or_overlapping_retry_window",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(can_create, bool):
            return self._decision(
                "dismiss",
                "invalid_capacity_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(reward_only_disagreement, bool):
            return self._decision(
                "dismiss",
                "invalid_reward_only_disagreement_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(group_recall_conflict_enabled, bool):
            return self._decision(
                "dismiss",
                "invalid_group_recall_conflict_enabled_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(group_recall_conflict_dual_novelty_retry_enabled, bool):
            return self._decision(
                "dismiss",
                "invalid_group_dual_novelty_retry_enabled_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(group_recall_conflict_dual_ambiguous_retry_enabled, bool):
            return self._decision(
                "dismiss",
                "invalid_group_dual_ambiguous_retry_enabled_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(group_recall_conflict_absolute_novelty_retry_enabled, bool):
            return self._decision(
                "dismiss",
                "invalid_group_recall_conflict_absolute_novelty_retry_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(retry_transaction_state_coherent, bool):
            return self._decision(
                "dismiss",
                "invalid_retry_transaction_state_coherent_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(recall_consensus_enabled, bool):
            return self._decision(
                "dismiss",
                "invalid_recall_consensus_enabled_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(unstable_boundary_bridge_enabled, bool):
            return self._decision(
                "dismiss",
                "invalid_unstable_boundary_bridge_enabled_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(unstable_boundary_bridge_transaction_state_coherent, bool):
            return self._decision(
                "dismiss",
                "invalid_unstable_boundary_bridge_state_coherent_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(
            unstable_boundary_bridge_retry_active_consensus_enabled, bool
        ):
            return self._decision(
                "dismiss",
                "invalid_unstable_boundary_bridge_retry_active_consensus_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(
            unstable_boundary_bridge_initial_strong_active_consensus_enabled, bool
        ):
            return self._decision(
                "dismiss",
                "invalid_unstable_boundary_bridge_initial_strong_active_consensus_enabled_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not isinstance(frozen_reward_only_context_active, bool):
            return self._decision(
                "dismiss",
                "invalid_frozen_reward_only_context_active_flag",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not can_create:
            return self._decision(
                "dismiss",
                "capacity_unavailable",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        full = _parse_identity(full_identity, required_feature_dim=FULL_FEATURE_DIM)
        if full is None:
            return self._decision(
                "dismiss",
                "invalid_full_classification",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        reward = _parse_reward(reward_evidence)
        if reward is None:
            return self._decision(
                "dismiss",
                "invalid_reward_evidence",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if not reward.persistent:
            return self._decision(
                "dismiss",
                "reward_evidence_not_persistent",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if active["initial_reward_signed_score"] * reward.signed_score <= 0.0:
            return self._decision(
                "dismiss",
                "reward_evidence_direction_changed",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if active["entry_mode"] in {
            _ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE,
            _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY,
        }:
            if not unstable_boundary_bridge_enabled:
                return self._decision(
                    "dismiss",
                    "unstable_boundary_bridge_not_enabled",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            if not unstable_boundary_bridge_transaction_state_coherent:
                return self._decision(
                    "dismiss",
                    "unstable_boundary_bridge_state_incoherent",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            r3 = _parse_identity(r3_shadow, required_feature_dim=R3_FEATURE_DIM)
            if r3 is None:
                return self._decision(
                    "dismiss",
                    "invalid_unstable_boundary_bridge_shadow",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            bridge = _parse_unstable_boundary_bridge(
                unstable_boundary_bridge_evidence,
                stage="retry",
                proposal_step=stored_proposal,
                decision_step=due,
                active_context_id=active["active_context_id"],
                r3_shadow=r3,
                full_identity=full,
                reward_evidence=reward,
                can_create=can_create,
                frozen_reward_only_context_active=frozen_reward_only_context_active,
                trusted_min_abs_reward_signed_score=unstable_boundary_bridge_min_abs_signed_score,
                trusted_min_reward_energy_score=unstable_boundary_bridge_min_energy_score,
                trusted_control_d_z_threshold=unstable_boundary_bridge_control_d_z_threshold,
                trusted_control_s_post_threshold=unstable_boundary_bridge_control_s_post_threshold,
                trusted_retry_active_consensus_enabled=unstable_boundary_bridge_retry_active_consensus_enabled,
                trusted_initial_strong_active_consensus_enabled=unstable_boundary_bridge_initial_strong_active_consensus_enabled,
                trusted_initial_dual_novel_min_abs_reward_signed_score=unstable_boundary_bridge_regime_failure_dual_novel_min_abs_signed_score,
                trusted_initial_dual_novel_min_reward_energy_score=unstable_boundary_bridge_regime_failure_dual_novel_min_energy_score,
                trusted_regime_failure_dual_novelty_retry=active["entry_mode"]
                == _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY,
            )
            if bridge is None:
                return self._decision(
                    "dismiss",
                    "invalid_unstable_boundary_bridge_evidence",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            if not bridge.triggered:
                return self._decision(
                    "dismiss",
                    "unstable_boundary_bridge_not_persistent",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            if active["entry_mode"] == _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY:
                if bridge.pattern != "dual_novelty":
                    return self._decision(
                        "dismiss",
                        "regime_failure_dual_novelty_not_persistent",
                        RETRY_DECISION_DELAY,
                        due,
                        rollback=True,
                    )
                return self._decision(
                    "create",
                    "persistent_regime_failure_dual_novelty",
                    RETRY_DECISION_DELAY,
                    due,
                )
            if bridge.pattern == "retry_active_consensus":
                initial_consensus = bool(
                    unstable_boundary_bridge_initial_strong_active_consensus_enabled
                    and active["initial_bridge_pattern"]
                    == "initial_strong_active_consensus"
                    and (active["initial_full_action"] == "dismiss_active")
                    and (active["initial_r3_action"] == "dismiss_active")
                    and (
                        active["initial_best_context_id"] == active["active_context_id"]
                    )
                )
                initial_masked_active = bool(
                    active["initial_bridge_pattern"] == "masked_active"
                    and active["initial_full_action"] == "dismiss_active"
                    and (active["initial_r3_action"] == "dismiss_active")
                    and (
                        active["initial_best_context_id"] != active["active_context_id"]
                    )
                )
                if not (initial_masked_active or initial_consensus):
                    return self._decision(
                        "dismiss",
                        "unstable_boundary_bridge_active_consensus_initial_not_masked",
                        RETRY_DECISION_DELAY,
                        due,
                        rollback=True,
                    )
                return self._decision(
                    "create",
                    "persistent_unstable_boundary_bridge_to_active_consensus",
                    RETRY_DECISION_DELAY,
                    due,
                )
            if (
                bridge.pattern in {"masked_active", "full_novelty_shadow_weak"}
                and bridge.candidate_context_id != active["initial_best_context_id"]
            ):
                return self._decision(
                    "dismiss",
                    "unstable_boundary_bridge_candidate_changed",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            return self._decision(
                "create",
                "persistent_unstable_boundary_bridge",
                RETRY_DECISION_DELAY,
                due,
            )
        if active["entry_mode"] == _ENTRY_MODE_REWARD_ONLY_DISAGREEMENT:
            if not reward_only_disagreement:
                return self._decision(
                    "dismiss",
                    "reward_only_disagreement_not_confirmed",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            return self._decision(
                "create",
                "persistent_reward_only_disagreement",
                RETRY_DECISION_DELAY,
                due,
            )
        if active["entry_mode"] == _ENTRY_MODE_GROUP_RECALL_CONFLICT:
            if not group_recall_conflict_enabled:
                return self._decision(
                    "dismiss",
                    "group_recall_conflict_guard_not_enabled",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            r3 = _parse_identity(r3_shadow, required_feature_dim=R3_FEATURE_DIM)
            if r3 is None:
                return self._decision(
                    "dismiss",
                    "invalid_group_recall_conflict_shadow",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            conflict = _parse_group_recall_conflict(
                group_recall_conflict_evidence,
                active_context_id=active["active_context_id"],
                r3_shadow=r3,
                full_identity=full,
                reward_evidence=reward,
                can_create=can_create,
            )
            if conflict is None:
                return self._decision(
                    "dismiss",
                    "invalid_group_recall_conflict_evidence",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            if (
                group_recall_conflict_dual_novelty_retry_enabled
                or group_recall_conflict_dual_ambiguous_retry_enabled
                or group_recall_conflict_absolute_novelty_retry_enabled
            ) and (not retry_transaction_state_coherent):
                return self._decision(
                    "dismiss",
                    "group_retry_transaction_state_incoherent",
                    RETRY_DECISION_DELAY,
                    due,
                    rollback=True,
                )
            if group_recall_conflict_absolute_novelty_retry_enabled:
                min_signed = _finite_float(
                    unstable_boundary_bridge_min_abs_signed_score
                )
                min_energy = _finite_float(unstable_boundary_bridge_min_energy_score)
                if (
                    min_signed is None
                    or min_energy is None
                    or min_signed <= 0.0
                    or (min_energy <= 0.0)
                ):
                    return self._decision(
                        "dismiss",
                        "invalid_group_absolute_novelty_thresholds",
                        RETRY_DECISION_DELAY,
                        due,
                        rollback=True,
                    )
                absolute_novelty = bool(
                    abs(reward.signed_score) >= min_signed
                    and reward.energy_score >= min_energy
                    and (abs(active["initial_reward_signed_score"]) >= min_signed)
                    and (active["initial_reward_energy_score"] >= min_energy)
                )
            else:
                absolute_novelty = True
            if (
                conflict.triggered
                and conflict.candidate_context_id == active["initial_best_context_id"]
                and absolute_novelty
            ):
                return self._decision(
                    "create",
                    "persistent_group_recall_conflict",
                    RETRY_DECISION_DELAY,
                    due,
                )
            dual_novelty = bool(
                group_recall_conflict_dual_novelty_retry_enabled
                and (not conflict.triggered)
                and (full.action == "create_novel")
                and (r3.action == "create_novel")
                and (full.best_context_id == active["active_context_id"])
                and (r3.best_context_id == active["active_context_id"])
                and (conflict.candidate_context_id == active["active_context_id"])
                and (conflict.fresh_vs_active_advantage > conflict.new_context_margin)
                and (full.new_advantage > conflict.new_context_margin)
                and (r3.new_advantage > conflict.new_context_margin)
                and isclose(
                    conflict.fresh_vs_active_advantage,
                    full.new_advantage,
                    rel_tol=1e-06,
                    abs_tol=2e-06,
                )
                and (
                    conflict.reward_signed_log_bayes
                    >= conflict.min_reward_signed_log_bayes
                )
                and absolute_novelty
            )
            if dual_novelty:
                return self._decision(
                    "create",
                    "persistent_group_conflict_to_dual_novelty",
                    RETRY_DECISION_DELAY,
                    due,
                )
            initial_candidate = active["initial_best_context_id"]
            dual_ambiguous = bool(
                group_recall_conflict_dual_ambiguous_retry_enabled
                and (not conflict.triggered)
                and (full.action == "create_ambiguous")
                and (r3.action == "create_ambiguous")
                and (initial_candidate != active["active_context_id"])
                and (full.best_context_id == initial_candidate)
                and (r3.best_context_id == initial_candidate)
                and (conflict.candidate_context_id == initial_candidate)
                and (conflict.fresh_vs_active_advantage > conflict.new_context_margin)
                and (
                    conflict.reward_signed_log_bayes
                    >= conflict.min_reward_signed_log_bayes
                )
                and absolute_novelty
            )
            if dual_ambiguous:
                return self._decision(
                    "create",
                    "persistent_group_conflict_to_dual_ambiguous",
                    RETRY_DECISION_DELAY,
                    due,
                )
            return self._decision(
                "dismiss",
                "group_recall_conflict_not_persistent",
                RETRY_DECISION_DELAY,
                due,
                rollback=True,
            )
        if full.action == "create_novel":
            return self._decision(
                "create", "persistent_full_novelty", RETRY_DECISION_DELAY, due
            )
        if (
            full.action == "recall_confident"
            and full.best_context_id is not None
            and (full.best_context_id != active["active_context_id"])
            and (full.best_context_id == active["initial_best_context_id"])
        ):
            if recall_consensus_enabled:
                r3 = _parse_identity(r3_shadow, required_feature_dim=R3_FEATURE_DIM)
                if not (
                    r3 is not None
                    and r3.action == "recall_confident"
                    and (r3.best_context_id == full.best_context_id)
                ):
                    return self._decision(
                        "dismiss",
                        "retry_recall_consensus_not_met",
                        RETRY_DECISION_DELAY,
                        due,
                        rollback=True,
                    )
            return self._decision(
                "recall",
                "consistent_confident_inactive_match",
                RETRY_DECISION_DELAY,
                due,
                context_id=full.best_context_id,
            )
        return self._decision(
            "dismiss",
            "retry_identity_not_persistent",
            RETRY_DECISION_DELAY,
            due,
            rollback=True,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "window_size": EVIDENCE_WINDOW,
            "initial_decision_delay": INITIAL_DECISION_DELAY,
            "retry_decision_delay": RETRY_DECISION_DELAY,
            "active": None if self._active is None else dict(self._active),
            "last_resolved_proposal_step": self._last_resolved_proposal_step,
        }

    @staticmethod
    def _validated_state(state: Any) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
        if not isinstance(state, Mapping):
            raise ValueError("quarantine state must be a mapping")
        required_top = {
            "version",
            "window_size",
            "initial_decision_delay",
            "retry_decision_delay",
            "active",
            "last_resolved_proposal_step",
        }
        if set(state.keys()) != required_top:
            raise ValueError("quarantine state keys mismatch")
        protocol_values = (
            (state["version"], STATE_VERSION),
            (state["window_size"], EVIDENCE_WINDOW),
            (state["initial_decision_delay"], INITIAL_DECISION_DELAY),
            (state["retry_decision_delay"], RETRY_DECISION_DELAY),
        )
        if any(
            (
                type(actual) is not int or actual != expected
                for (actual, expected) in protocol_values
            )
        ):
            raise ValueError("quarantine state protocol mismatch")
        raw_last = state["last_resolved_proposal_step"]
        last = None if raw_last is None else _strict_integer(raw_last, minimum=0)
        if raw_last is not None and last is None:
            raise ValueError("invalid last resolved proposal step")
        raw_active = state["active"]
        if raw_active is None:
            return (None, last)
        if not isinstance(raw_active, Mapping):
            raise ValueError("active quarantine state must be a mapping")
        required_active = {
            "proposal_step",
            "initial_due_step",
            "retry_due_step",
            "initial_window_start_step",
            "initial_window_end_step",
            "active_context_id",
            "initial_best_context_id",
            "initial_r3_action",
            "initial_full_action",
            "initial_reward_signed_score",
            "initial_reward_energy_score",
            "entry_mode",
            "initial_bridge_pattern",
            "retry_attempts",
        }
        if set(raw_active.keys()) != required_active:
            raise ValueError("active quarantine state keys mismatch")
        integer_fields = {
            key: _strict_integer(raw_active[key], minimum=0)
            for key in (
                "proposal_step",
                "initial_due_step",
                "retry_due_step",
                "initial_window_start_step",
                "initial_window_end_step",
                "active_context_id",
                "initial_best_context_id",
                "retry_attempts",
            )
        }
        if any((value is None for value in integer_fields.values())):
            raise ValueError("invalid active quarantine integer")
        proposal = integer_fields["proposal_step"]
        initial_due = integer_fields["initial_due_step"]
        retry_due = integer_fields["retry_due_step"]
        start = integer_fields["initial_window_start_step"]
        end = integer_fields["initial_window_end_step"]
        if (
            initial_due - proposal != INITIAL_DECISION_DELAY
            or retry_due - proposal != RETRY_DECISION_DELAY
            or end != initial_due
            or (end - start + 1 != EVIDENCE_WINDOW)
            or (integer_fields["retry_attempts"] != 0)
        ):
            raise ValueError("incoherent active quarantine state")
        entry_mode = raw_active["entry_mode"]
        initial_bridge_pattern = raw_active["initial_bridge_pattern"]
        initial_r3_action = raw_active["initial_r3_action"]
        initial_full_action = raw_active["initial_full_action"]
        if (
            entry_mode not in _ENTRY_MODES
            or (
                entry_mode
                not in {
                    _ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE,
                    _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY,
                }
                and initial_bridge_pattern is not None
            )
            or (
                entry_mode
                in {
                    _ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE,
                    _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY,
                }
                and initial_bridge_pattern
                not in {
                    "masked_active",
                    "dual_novelty",
                    "initial_strong_active_consensus",
                }
            )
            or (initial_r3_action not in _IDENTITY_ACTIONS)
            or (initial_full_action not in _IDENTITY_ACTIONS)
            or (
                entry_mode
                not in {
                    _ENTRY_MODE_GROUP_RECALL_CONFLICT,
                    _ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE,
                    _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY,
                }
                and initial_r3_action != "dismiss_active"
            )
            or (
                entry_mode == _ENTRY_MODE_FULL_NOVELTY
                and initial_full_action != "create_novel"
            )
            or (
                entry_mode == _ENTRY_MODE_REWARD_ONLY_DISAGREEMENT
                and initial_full_action == "create_novel"
            )
            or (
                entry_mode == _ENTRY_MODE_GROUP_RECALL_CONFLICT
                and (
                    initial_r3_action != "recall_confident"
                    or initial_full_action != "recall_confident"
                )
            )
            or (
                entry_mode
                in {
                    _ENTRY_MODE_UNSTABLE_BOUNDARY_BRIDGE,
                    _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY,
                }
                and (
                    not (
                        initial_bridge_pattern == "masked_active"
                        and initial_r3_action == "dismiss_active"
                        and (initial_full_action == "dismiss_active")
                        and (
                            integer_fields["initial_best_context_id"]
                            != integer_fields["active_context_id"]
                        )
                        or (
                            initial_bridge_pattern == "dual_novelty"
                            and initial_r3_action == "create_novel"
                            and (initial_full_action == "create_novel")
                            and (
                                integer_fields["initial_best_context_id"]
                                == integer_fields["active_context_id"]
                            )
                        )
                        or (
                            initial_bridge_pattern == "initial_strong_active_consensus"
                            and initial_r3_action == "dismiss_active"
                            and (initial_full_action == "dismiss_active")
                            and (
                                integer_fields["initial_best_context_id"]
                                == integer_fields["active_context_id"]
                            )
                        )
                    )
                )
            )
            or (
                entry_mode == _ENTRY_MODE_REGIME_FAILURE_DUAL_NOVELTY
                and initial_bridge_pattern != "dual_novelty"
            )
        ):
            raise ValueError("incoherent active quarantine entry mode")
        if last is not None and proposal <= last:
            raise ValueError("active proposal is not newer than resolved state")
        signed = _finite_float(raw_active["initial_reward_signed_score"])
        energy = _finite_float(raw_active["initial_reward_energy_score"])
        if signed is None or signed == 0.0 or energy is None or (energy <= 0.0):
            raise ValueError("invalid active reward evidence")
        active = dict(raw_active)
        active.update(integer_fields)
        active["initial_reward_signed_score"] = signed
        active["initial_reward_energy_score"] = energy
        active["entry_mode"] = entry_mode
        active["initial_bridge_pattern"] = initial_bridge_pattern
        return (active, last)

    def load_state_dict(self, state: Any) -> None:
        if state is None:
            self.reset()
            return
        try:
            active, last = self._validated_state(state)
        except (KeyError, TypeError, ValueError, OverflowError):
            self.reset()
            raise
        self._active = None if active is None else dict(active)
        self._last_resolved_proposal_step = last
