from collections import deque
from collections.abc import Mapping
from copy import deepcopy
import numpy as np
import json
import os
import time
import torch
import torch.nn.functional as F

_CUDAGRAPH_MARK_ENABLED = True


def _safe_cudagraph_mark_step_begin():
    global _CUDAGRAPH_MARK_ENABLED
    return


from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.m3wm_context_memory import (
    MultiscaleShiftDetector,
    batched_conditioned_action_response_log_norm,
    conditioned_action_response_distance,
    conditioned_action_response_log_norm,
    linear_action_response_log_norm,
    supervised_context_separation_loss,
    system_id_routing_targets,
)
from common.m3wm_quarantine import (
    EVIDENCE_WINDOW as M3WM_QUARANTINE_EVIDENCE_WINDOW,
    M3wmOneShotQuarantine,
)
from common.m3wm_diagnostics import (
    M3wmDiagnosticRecorder,
    canonical_json_sha256,
    canonical_context_slot_digests,
    finite_or_none,
    runtime_step_coordinates,
    signed_reward_residual_summary,
    tensor_snapshot,
)
from common.m3wm_objective_detector import ProposalLocalObjectiveChangeEvaluator
from common.m3wm_identity_event_artifact import identity_event_payload_sha256
from common.m3wm_soft_context_reuse import SoftContextReuseController
from common.m3wm_reversible_reuse_trial import (
    VERSION as M3WM_REVERSIBLE_REUSE_TRIAL_VERSION,
)
from common.m3wm_factorized_law_id import empty_shadow_reference_state
from common.m3wm_context_evidence_stream import ContextEvidenceStream
from common.m3wm_detector_calibration import EmpiricalNullCalibrator
from common.m3wm_v66_core_bank import empty_core_bank

_M3WM_REWARD_IDENTITY_WINDOW = 128
_M3WM_EPISODIC_THETA_REWARD_BLOCK = 128
_M3WM_EPISODIC_THETA_MIN_BLOCKS = 4
_M3WM_REWARD_IDENTITY_MAD_SCALE = 1.4826
_M3WM_REWARD_IDENTITY_WINSOR_Z = 8.0
_M3WM_REWARD_IDENTITY_MIN_SCALE = 1e-06
_M3WM_REWARD_IDENTITY_WEIGHTS = frozenset({0.25, 0.5, 1.0})
_M3WM_CONTEXT_LAW_TRANSFER_RIDGE = 0.01
_M3WM_CONTEXT_LAW_TRANSFER_MIN_STD = 0.001
_M3WM_ARCHITECTURE_IDENTITY_MAX_LATENT_GROUPS = 4
_M3WM_SEMANTIC_BOCPD_MODES = frozenset(
    {"bocpd_fixed_three_stream", "bocpd_live_three_stream"}
)
_M3WM_LIVE_MULTISCALE_MODES = frozenset({"multiscale_live_error_shift"})
_M3WM_FIXED_LIVE_IDENTITY_MODES = (
    _M3WM_SEMANTIC_BOCPD_MODES | _M3WM_LIVE_MULTISCALE_MODES
)
_M3WM_LIVE_MEMORY_FEATURE_MODE = "architecture_live_identity"
M3WM_EPISODE_IDENTITY_AUDIT_STEP = M3WM_QUARANTINE_EVIDENCE_WINDOW
M3WM_EPISODE_IDENTITY_AUDIT_SOURCE = "identity_audit"
_M3WM_EPISODE_IDENTITY_AUDIT_REASONS = frozenset(
    {"group_recall_conflict", "dual_identity_novelty", "reward_only_disagreement"}
)
_M3WM_EVAL_MPC_ACTION_MODES = frozenset({"sample_elite", "weighted_mean"})
_M3WM_DETECTOR_PROPOSAL_SOURCES = frozenset({"bocpd", "multiscale"})
_M3WM_V57_SHADOW_STATE_CODES = {
    "DISABLED": -2,
    "UNKNOWN": -1,
    "UNRESOLVED": 0,
    "INSUFFICIENT": 1,
    "OBSERVING_OWNERS": 2,
    "STAY_EVIDENCE": 3,
    "RECALL_EVIDENCE": 4,
    "REPLICATION_CONFIRMED": 5,
    "COVERAGE_GAP": 6,
    "INVALID": 7,
    "TERMINAL_NEGATIVE": 8,
    "TERMINAL_UNIDENTIFIABLE": 9,
}


def _m3wm_v57_shadow_receipt(reason, **values):
    return {
        "schema": "m3wm_v57_production_shadow",
        "version": 1,
        "valid": False,
        "state": "UNRESOLVED",
        "reason": str(reason),
        **values,
        "shadow_only": True,
        "decision_inert": True,
        "authority_eligible": False,
        "revision_binding_authoritative": False,
        "legacy_identity_authority": "V56",
        "may_create": False,
        "may_recall": False,
        "may_stay": False,
        "may_unlock": False,
        "may_write_official_bank": False,
        "may_write_escrow_bank": False,
        "may_mutate_context_memory": False,
        "may_mutate_replay": False,
        "may_mutate_optimizer": False,
        "may_change_action": False,
    }


_M3WM_V57_SHADOW_SIDECAR_SCHEMA = "m3wm_v57_shadow_sidecar"
_M3WM_V57_SHADOW_SIDECAR_VERSION = 1
_M3WM_V57_SHADOW_SIDECAR_CAPACITY = 8
_M3WM_V57_SHADOW_SIDECAR_EVENTS = frozenset({"OWNER_EVALUATION", "DUE_REGISTRATION"})
_M3WM_V57_DENIED_CAPABILITIES = (
    "may_create",
    "may_recall",
    "may_stay",
    "may_unlock",
    "may_write_official_bank",
    "may_write_escrow_bank",
    "may_mutate_context_memory",
    "may_mutate_replay",
    "may_mutate_optimizer",
    "may_change_action",
)


def validate_m3wm_v57_shadow_sidecar(artifact):
    required = {
        "schema",
        "version",
        "record_index",
        "event_type",
        "proposal_step",
        "due_step",
        "runtime_step",
        "active_context_id",
        "context_count",
        "shadow_only",
        "decision_inert",
        "authority_eligible",
        "revision_binding_authoritative",
        "legacy_identity_authority",
        "durable",
        "durability_reason",
        "queue_capacity",
        "source_fingerprint",
        "config_fingerprint",
        "source_binding",
        "registration",
        "candidate",
        "owner_ledger",
        "owner_evaluation",
        "candidate_self_consistency",
        "payload_sha256",
        *_M3WM_V57_DENIED_CAPABILITIES,
    }
    if (
        not isinstance(artifact, dict)
        or set(artifact) != required
        or artifact.get("schema") != _M3WM_V57_SHADOW_SIDECAR_SCHEMA
        or (artifact.get("version") != _M3WM_V57_SHADOW_SIDECAR_VERSION)
        or (artifact.get("event_type") not in _M3WM_V57_SHADOW_SIDECAR_EVENTS)
    ):
        raise ValueError("Invalid V57 shadow sidecar schema")
    payload = {
        key: value for (key, value) in artifact.items() if key != "payload_sha256"
    }
    if artifact.get("payload_sha256") != identity_event_payload_sha256(payload):
        raise ValueError("V57 shadow sidecar digest mismatch")
    if (
        isinstance(artifact.get("record_index"), bool)
        or not isinstance(artifact.get("record_index"), int)
        or artifact["record_index"] < 0
        or any(
            (
                isinstance(artifact.get(name), bool)
                or not isinstance(artifact.get(name), int)
                or artifact[name] < (0 if name in {"active_context_id"} else 1)
                for name in (
                    "proposal_step",
                    "due_step",
                    "runtime_step",
                    "active_context_id",
                    "context_count",
                )
            )
        )
        or (artifact["due_step"] < artifact["proposal_step"])
        or (artifact.get("shadow_only") is not True)
        or (artifact.get("decision_inert") is not True)
        or (artifact.get("authority_eligible") is not False)
        or (artifact.get("revision_binding_authoritative") is not False)
        or (artifact.get("legacy_identity_authority") != "V56")
        or (artifact.get("durable") is not False)
        or (
            artifact.get("durability_reason")
            != "agent_queue_requires_external_consumer"
        )
        or (artifact.get("queue_capacity") != _M3WM_V57_SHADOW_SIDECAR_CAPACITY)
        or (artifact["active_context_id"] >= artifact["context_count"])
        or (
            artifact["event_type"] == "DUE_REGISTRATION"
            and artifact["runtime_step"] != artifact["due_step"]
        )
        or any(
            (artifact.get(name) is not False for name in _M3WM_V57_DENIED_CAPABILITIES)
        )
    ):
        raise ValueError("Invalid V57 shadow sidecar authority binding")
    source = artifact.get("source_fingerprint")
    if (
        not isinstance(source, dict)
        or source.get("authoritative") is not False
        or source.get("reason") != "trainer_run_manifest_source_not_bound"
        or (source.get("source_map_sha256") is not None)
        or (
            source.get("payload_sha256")
            != identity_event_payload_sha256(
                {
                    key: value
                    for (key, value) in source.items()
                    if key != "payload_sha256"
                }
            )
        )
    ):
        raise ValueError("Invalid V57 shadow source fingerprint")
    config = artifact.get("config_fingerprint")
    if (
        not isinstance(config, dict)
        or set(config) != {"values", "payload_sha256"}
        or (not isinstance(config.get("values"), dict))
        or (
            config.get("payload_sha256")
            != identity_event_payload_sha256({"values": config.get("values")})
        )
    ):
        raise ValueError("Invalid V57 shadow config fingerprint")
    registration = artifact.get("registration")
    owner_evaluation = artifact.get("owner_evaluation")
    for name, receipt in (
        ("registration", registration),
        ("owner_evaluation", owner_evaluation),
    ):
        if receipt is None:
            continue
        if (
            not isinstance(receipt, dict)
            or receipt.get("shadow_only") is not True
            or receipt.get("decision_inert") is not True
            or (receipt.get("authority_eligible") is not False)
            or (receipt.get("legacy_identity_authority") != "V56")
            or any(
                (
                    receipt.get(field) is not False
                    for field in _M3WM_V57_DENIED_CAPABILITIES
                )
            )
        ):
            raise ValueError(f"Invalid V57 shadow {name} capabilities")
    for name in ("candidate", "owner_ledger", "candidate_self_consistency"):
        carrier = artifact.get(name)
        if carrier is None:
            continue
        if (
            not isinstance(carrier, dict)
            or not isinstance(carrier.get("payload_sha256"), str)
            or carrier.get("payload_sha256")
            != identity_event_payload_sha256(
                {
                    key: value
                    for (key, value) in carrier.items()
                    if key != "payload_sha256"
                }
            )
        ):
            raise ValueError(f"Invalid V57 shadow {name} digest")
        if (
            carrier.get("decision_inert") is not True
            or carrier.get("may_create") is not False
            or carrier.get("may_recall") is not False
            or (carrier.get("may_mutate_context_memory") is not False)
            or (carrier.get("mutation_performed") is not False)
        ):
            raise ValueError(f"Invalid V57 shadow {name} authority")
    if isinstance(owner_evaluation, dict):
        source_evaluation = owner_evaluation.get("source_evaluation")
        if source_evaluation is not None and (
            not isinstance(source_evaluation, dict)
            or source_evaluation.get("payload_sha256")
            != identity_event_payload_sha256(
                {
                    key: value
                    for (key, value) in source_evaluation.items()
                    if key != "payload_sha256"
                }
            )
            or owner_evaluation.get("source_evaluation_payload_sha256")
            != source_evaluation.get("payload_sha256")
            or (source_evaluation.get("decision_inert") is not True)
            or (source_evaluation.get("authority_eligible") is not False)
            or (source_evaluation.get("may_create") is not False)
            or (source_evaluation.get("may_recall") is not False)
            or (source_evaluation.get("may_mutate_context_memory") is not False)
            or (source_evaluation.get("mutation_performed") is not False)
        ):
            raise ValueError("Invalid V57 nested owner evaluation digest")
    if artifact["event_type"] == "DUE_REGISTRATION":
        binding = artifact.get("source_binding")
        pending = binding.get("pending") if isinstance(binding, dict) else None
        assignment = (
            binding.get("legacy_v56_assignment") if isinstance(binding, dict) else None
        )
        if (
            registration is None
            or owner_evaluation is not None
            or (not isinstance(pending, dict))
            or (not isinstance(assignment, dict))
            or (pending.get("proposal_step") != artifact["proposal_step"])
            or (
                int(pending.get("confirmation_due_step", artifact["due_step"]))
                != artifact["due_step"]
            )
            or (
                binding.get("legacy_v56_assignment_sha256")
                != identity_event_payload_sha256({"legacy_v56_assignment": assignment})
            )
        ):
            raise ValueError("Invalid V57 due sidecar source binding")
        candidate = artifact.get("candidate")
        if candidate is not None and registration.get(
            "candidate_payload_sha256"
        ) not in {None, candidate.get("payload_sha256")}:
            raise ValueError("Invalid V57 due registration candidate binding")
    else:
        candidate = artifact.get("candidate")
        ledger = artifact.get("owner_ledger")
        binding = artifact.get("source_binding")
        if (
            candidate is None
            or owner_evaluation is None
            or (not isinstance(binding, dict))
            or (
                binding.get("candidate_payload_sha256")
                != candidate.get("payload_sha256")
            )
        ):
            raise ValueError("Invalid V57 owner sidecar source binding")
        if ledger is not None and ledger.get(
            "candidate_payload_sha256"
        ) != candidate.get("payload_sha256"):
            raise ValueError("Invalid V57 owner ledger candidate binding")
        source_evaluation = owner_evaluation.get("source_evaluation")
        if source_evaluation is not None and (
            source_evaluation.get("candidate_payload_sha256")
            != candidate.get("payload_sha256")
            or ledger is None
            or source_evaluation.get("ledger_payload_sha256")
            != ledger.get("payload_sha256")
        ):
            raise ValueError("Invalid V57 owner evaluation carrier binding")
        self_consistency = artifact.get("candidate_self_consistency")
        if self_consistency is not None and (
            self_consistency.get("candidate_payload_sha256")
            != candidate.get("payload_sha256")
            or ledger is None
            or self_consistency.get("ledger_payload_sha256")
            != ledger.get("payload_sha256")
        ):
            raise ValueError("Invalid V57 self-consistency carrier binding")
    return artifact


def _multiscale_detector_fsm_metrics(detector):
    fields = (
        "multiscale_fsm_armed",
        "multiscale_fsm_cooldown_remaining",
        "multiscale_fsm_above_threshold_count",
        "multiscale_fsm_below_threshold_count",
    )
    if not isinstance(detector, MultiscaleShiftDetector):
        return {field: float("nan") for field in fields}
    return {
        "multiscale_fsm_armed": float(detector.armed),
        "multiscale_fsm_cooldown_remaining": float(detector.cooldown_remaining),
        "multiscale_fsm_above_threshold_count": float(detector.above_threshold_count),
        "multiscale_fsm_below_threshold_count": float(detector.below_threshold_count),
    }


def _live_bocpd_edge_gate_metrics(gate):
    fields = (
        "bocpd_edge_armed",
        "bocpd_edge_stable_low_steps",
        "bocpd_edge_suppressed_count",
        "bocpd_edge_rearm_count",
    )
    if gate is None:
        return {field: float("nan") for field in fields}
    return {
        "bocpd_edge_armed": float(gate.armed),
        "bocpd_edge_stable_low_steps": float(gate.stable_low_steps),
        "bocpd_edge_suppressed_count": float(gate.suppressed_count),
        "bocpd_edge_rearm_count": float(gate.rearm_count),
    }


def _validated_signed_reward_slow_config(cfg, change_mode, reward_residual_available):
    if not isinstance(False, (bool, np.bool_)):
        raise ValueError("m3wm_change_signed_reward_slow_enabled must be boolean")
    raw_window = cfg.get("m3wm_change_signed_reward_slow_window", 32)
    if (
        isinstance(raw_window, (bool, np.bool_))
        or not isinstance(raw_window, (int, np.integer))
        or int(raw_window) < 2
    ):
        raise ValueError(
            "m3wm_change_signed_reward_slow_window must be an integer >= 2"
        )
    raw_consistency = cfg.get("m3wm_change_signed_reward_slow_min_consistency", 0.75)
    if (
        isinstance(raw_consistency, (str, bool, np.bool_))
        or not np.isfinite(float(raw_consistency))
        or (not 0.5 < float(raw_consistency) <= 1.0)
    ):
        raise ValueError(
            "m3wm_change_signed_reward_slow_min_consistency must be in (0.5, 1]"
        )
    short_window = cfg.get(
        "m3wm_change_short_window", cfg.get("m3wm_context_short_window", 8)
    )
    return (False, int(raw_window), float(raw_consistency))


def _validated_m3wm_eval_mpc_action_mode(value):
    if not isinstance(value, str):
        raise ValueError(
            "m3wm_eval_mpc_action_mode must be one of {'sample_elite', 'weighted_mean'}"
        )
    mode = value.lower()
    if mode not in _M3WM_EVAL_MPC_ACTION_MODES:
        raise ValueError(
            f"m3wm_eval_mpc_action_mode must be one of {{'sample_elite', 'weighted_mean'}}, received {value!r}"
        )
    return mode


def _validated_m3wm_branchwise_theta_trust_region(enabled, radius):
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("m3wm_branchwise_theta_trust_region_enabled must be boolean")
    if isinstance(radius, (str, bool, np.bool_)):
        raise ValueError(
            "m3wm_branchwise_theta_trust_region_radius must be a finite non-negative number"
        )
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "m3wm_branchwise_theta_trust_region_radius must be a finite non-negative number"
        ) from error
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError(
            "m3wm_branchwise_theta_trust_region_radius must be a finite non-negative number"
        )
    if bool(enabled) and radius <= 0.0:
        raise ValueError(
            "enabled branchwise theta trust region requires a positive radius"
        )
    if not bool(enabled) and radius != 0.0:
        raise ValueError("disabled branchwise theta trust region requires radius 0")
    return (bool(enabled), radius)


def _m3wm_validated_branch_simplex(theta, name, expected_shape=None):
    if (
        not torch.is_tensor(theta)
        or not torch.is_floating_point(theta)
        or theta.ndim not in {1, 2}
        or (theta.shape[-1] < 2)
        or (theta.numel() < 2)
        or (expected_shape is not None and tuple(theta.shape) != expected_shape)
        or (not bool(torch.isfinite(theta).all()))
        or bool((theta < 0.0).any())
        or (
            not bool(
                torch.allclose(
                    theta.sum(dim=-1),
                    torch.ones_like(theta.sum(dim=-1)),
                    atol=1e-05,
                    rtol=0.0,
                )
            )
        )
    ):
        raise ValueError(f"{name} is not a finite branch simplex")
    return theta.detach().clone()


def _m3wm_branchwise_centered_logit_l2(theta, anchor):
    try:
        candidate = _m3wm_validated_branch_simplex(theta, "candidate theta")
        reference = _m3wm_validated_branch_simplex(
            anchor, "anchor theta", expected_shape=tuple(candidate.shape)
        ).to(device=candidate.device, dtype=candidate.dtype)
        candidate_rows = candidate.reshape(-1, candidate.shape[-1]).double()
        anchor_rows = reference.reshape(-1, reference.shape[-1]).double()
        candidate_logits = candidate_rows.clamp_min(1e-08).log()
        anchor_logits = anchor_rows.clamp_min(1e-08).log()
        candidate_logits -= candidate_logits.mean(dim=-1, keepdim=True)
        anchor_logits -= anchor_logits.mean(dim=-1, keepdim=True)
        distance = torch.linalg.vector_norm(candidate_logits - anchor_logits, dim=-1)
        if not bool(torch.isfinite(distance).all()):
            raise ValueError("centered-logit distance is non-finite")
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise RuntimeError(
            f"Branchwise centered-logit trust region failed closed: {error}"
        ) from error
    return distance.reshape(theta.shape[:-1]).to(device=theta.device, dtype=theta.dtype)


def _m3wm_project_branchwise_centered_logit_trust_region(theta, anchor, radius):
    try:
        if isinstance(radius, (str, bool, np.bool_)):
            raise ValueError("radius is not a finite positive number")
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius is not a finite positive number")
        candidate = _m3wm_validated_branch_simplex(theta, "candidate theta")
        reference = _m3wm_validated_branch_simplex(
            anchor, "anchor theta", expected_shape=tuple(candidate.shape)
        ).to(device=candidate.device, dtype=candidate.dtype)
        candidate_rows = candidate.reshape(-1, candidate.shape[-1])
        anchor_rows = reference.reshape(-1, reference.shape[-1])
        work_candidate = candidate_rows.double()
        work_anchor = anchor_rows.double()
        candidate_logits = work_candidate.clamp_min(1e-08).log()
        anchor_logits = work_anchor.clamp_min(1e-08).log()
        candidate_logits -= candidate_logits.mean(dim=-1, keepdim=True)
        anchor_logits -= anchor_logits.mean(dim=-1, keepdim=True)
        delta = candidate_logits - anchor_logits
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        if not bool(torch.isfinite(norm).all()):
            raise ValueError("centered-logit distance is non-finite")
        outside = norm.squeeze(-1) > radius
        if not bool(outside.any()):
            return candidate
        scale = (radius / norm.clamp_min(1e-16)).clamp(max=1.0)
        projected_logits = anchor_logits + delta * scale
        projected_rows = torch.softmax(projected_logits, dim=-1)
        result_rows = candidate_rows.clone()
        result_rows[outside] = projected_rows[outside].to(result_rows)
        result = result_rows.reshape(candidate.shape)
        result = _m3wm_validated_branch_simplex(
            result, "projected theta", expected_shape=tuple(candidate.shape)
        )
        projected_distance = _m3wm_branchwise_centered_logit_l2(result, reference)
        if bool((projected_distance > radius + 1e-05).any()):
            raise ValueError("finite-precision projection exceeded its radius")
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        if isinstance(error, RuntimeError) and str(error).startswith(
            "Branchwise centered-logit trust region failed closed:"
        ):
            raise
        raise RuntimeError(
            f"Branchwise centered-logit trust region failed closed: {error}"
        ) from error
    return result


def _context_reward_residual_index(model):
    if getattr(model, "m3wm_context_feature_mode", "legacy") == "legacy" or not bool(
        getattr(model, "m3wm_use_reward_residual", False)
    ):
        return None
    transform = getattr(model, "m3wm_context_residual_transform", None)
    latent_dim = int(model.cfg.latent_dim)
    action_dim = int(model.cfg.action_dim)
    if transform == "architecture":
        return 2 * latent_dim + action_dim + 2
    if transform == "abs_action_norm":
        return latent_dim + 2
    if transform == "stats_group_rms":
        return (
            9
            + int(model.m3wm_context_group_count)
            + 3 * int(bool(model.m3wm_use_action_gain_feature))
        )
    return action_dim + latent_dim + 1


def _diagnostic_reward_residual_index(model):
    return _context_reward_residual_index(model)


def _validated_reward_identity_weight(value):
    weight = float(value)
    if weight not in _M3WM_REWARD_IDENTITY_WEIGHTS:
        raise ValueError(
            "m3wm_dp_reward_feature_weight must be one of {0.25, 0.5, 1.0}"
        )
    return weight


def _validated_reward_only_score_threshold(value, name):
    if isinstance(value, (str, bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return threshold


def _validated_reward_only_frozen_theta(value, model):
    if value is None:
        return None
    try:
        theta = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "m3wm_reward_only_frozen_theta must be a finite composition"
        ) from error
    branches = (
        int(model.m3wm_num_composition_branches)
        if bool(model.m3wm_separate_composition_heads)
        else 1
    )
    primitives = int(model.m3wm_num_primitives)
    if (
        not bool(model.m3wm_separate_composition_heads)
        or branches != 3
        or primitives != 4
    ):
        raise ValueError(
            "m3wm_reward_only_frozen_theta requires separate 3x4 composition heads"
        )
    expected_shape = (branches, primitives) if branches > 1 else (primitives,)
    if tuple(theta.shape) != expected_shape:
        raise ValueError(
            f"m3wm_reward_only_frozen_theta must have shape {expected_shape}, got {tuple(theta.shape)}"
        )
    if (
        not bool(torch.isfinite(theta).all())
        or bool((theta < 0.0).any())
        or (
            not bool(
                torch.allclose(
                    theta.sum(dim=-1),
                    torch.ones_like(theta.sum(dim=-1)),
                    atol=1e-05,
                    rtol=0.0,
                )
            )
        )
    ):
        raise ValueError(
            "m3wm_reward_only_frozen_theta must contain nonnegative, finite branches that each sum to one"
        )
    return theta / theta.sum(dim=-1, keepdim=True)


def _reward_identity_null_location_scale(values, min_samples=128):
    values = list(values)
    if len(values) < int(min_samples):
        return None
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.numel() < int(min_samples) or not bool(torch.isfinite(tensor).all()):
        return None
    median = torch.quantile(tensor, 0.5)
    mad = torch.quantile((tensor - median).abs(), 0.5)
    scale = _M3WM_REWARD_IDENTITY_MAD_SCALE * mad
    if (
        not bool(torch.isfinite(scale))
        or float(scale) <= _M3WM_REWARD_IDENTITY_MIN_SCALE
    ):
        return None
    return (float(median), float(mad))


_M3WM_SEQUENTIAL_HISTORY_FIXED_UNRESOLVED_REASONS = frozenset(
    {
        "active_provisional_escrow_out_of_coverage",
        "active_provisional_escrow_reject_requires_h_new",
        "active_provisional_escrow_calibration_unresolved",
        "active_provisional_calibration",
        "crossfit_residual_rank_change_unconfirmed",
        "active_calibration_unreliable",
        "active_provisional_escrow_reject_calibration_insufficient",
        "active_predictive_unresolved",
        "active_calibration_insufficient",
    }
)


class M3wmPendingChangeTracker:

    def __init__(self) -> None:
        self._pending: dict | None = None
        self.drop_count = 0

    @staticmethod
    def _valid_integer(value, minimum=0):
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            numeric = float(value)
            integer = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not np.isfinite(numeric)
            or numeric != float(integer)
            or integer < int(minimum)
        ):
            return None
        return integer

    @staticmethod
    def _finite_scalar(value):
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return numeric if np.isfinite(numeric) else None

    @classmethod
    def _finite_payload(cls, value, seen=None, depth=0):
        if depth > 64:
            return False
        if value is None or isinstance(value, (str, bool, np.bool_)):
            return True
        if isinstance(value, (int, float, np.number)):
            try:
                return bool(np.isfinite(float(value)))
            except (TypeError, ValueError, OverflowError):
                return False
        if torch.is_tensor(value):
            return bool(torch.isfinite(value.detach()).all())
        if isinstance(value, np.ndarray):
            return bool(np.isfinite(value).all())
        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            return False
        if isinstance(value, dict):
            if not all((isinstance(key, str) for key in value)):
                return False
            seen.add(identity)
            valid = all(
                (cls._finite_payload(item, seen, depth + 1) for item in value.values())
            )
            seen.remove(identity)
            return valid
        if isinstance(value, (list, tuple)):
            seen.add(identity)
            valid = all((cls._finite_payload(item, seen, depth + 1) for item in value))
            seen.remove(identity)
            return valid
        return False

    @classmethod
    def _validated_quarantine(cls, value, original_due, current_due):
        required = {
            "retry_count",
            "deferred_step",
            "retry_confirmation_due_step",
            "active_context_id",
            "candidate_context_id",
            "evidence",
            "evidence_sha256",
            "quarantine_state",
        }
        if not isinstance(value, dict) or set(value) != required:
            return None
        retry_count = cls._valid_integer(value["retry_count"], minimum=0)
        deferred_step = cls._valid_integer(value["deferred_step"])
        retry_due = cls._valid_integer(value["retry_confirmation_due_step"])
        active_id = cls._valid_integer(value["active_context_id"])
        candidate_id = cls._valid_integer(value["candidate_context_id"])
        evidence = value["evidence"]
        evidence_sha256 = value["evidence_sha256"]
        quarantine_state = value["quarantine_state"]
        try:
            computed_evidence_sha256 = canonical_json_sha256(evidence)
        except (TypeError, ValueError, OverflowError):
            computed_evidence_sha256 = None
        if (
            retry_count != 1
            or deferred_step != original_due
            or retry_due != current_due
            or (current_due != original_due + 128)
            or (active_id is None)
            or (candidate_id is None)
            or (not isinstance(evidence, dict))
            or (not evidence)
            or (not cls._finite_payload(evidence))
            or (not isinstance(evidence_sha256, str))
            or (len(evidence_sha256) != 64)
            or any(
                (character not in "0123456789abcdef" for character in evidence_sha256)
            )
            or (evidence_sha256 != computed_evidence_sha256)
            or (
                quarantine_state is not None
                and (not isinstance(quarantine_state, dict))
            )
            or (not cls._finite_payload(quarantine_state))
        ):
            return None
        return {
            "retry_count": 1,
            "deferred_step": original_due,
            "retry_confirmation_due_step": current_due,
            "active_context_id": active_id,
            "candidate_context_id": candidate_id,
            "evidence": deepcopy(evidence),
            "evidence_sha256": evidence_sha256,
            "quarantine_state": deepcopy(quarantine_state),
        }

    @classmethod
    def _validated_pending(cls, pending):
        legacy_required = {
            "source",
            "proposal_step",
            "proposal_probability",
            "proposal_map_run_length",
            "confirmation_due_step",
            "confirmation_step",
            "confirmation_probability",
        }
        modern_optional = {
            "original_confirmation_due_step",
            "quarantine",
            "proposal_threshold",
            "runtime_theta_history",
            "same_version_proposal_raw_evidence",
            "physical_response_proposal_evidence",
        }
        if (
            not isinstance(pending, dict)
            or not legacy_required.issubset(pending)
            or (not set(pending).issubset(legacy_required | modern_optional))
        ):
            return None
        source = pending["source"]
        proposal = cls._valid_integer(pending["proposal_step"])
        probability = cls._finite_scalar(pending["proposal_probability"])
        proposal_threshold = pending.get("proposal_threshold")
        if proposal_threshold is not None:
            proposal_threshold = cls._finite_scalar(proposal_threshold)
        map_run_length = cls._valid_integer(pending["proposal_map_run_length"])
        current_due = cls._valid_integer(pending["confirmation_due_step"])
        quarantine = pending.get("quarantine")
        if "original_confirmation_due_step" in pending:
            original_due = cls._valid_integer(pending["original_confirmation_due_step"])
        elif quarantine is None:
            original_due = current_due
        else:
            original_due = None
        if (
            not isinstance(source, str)
            or not source
            or proposal is None
            or (probability is None)
            or (
                pending.get("proposal_threshold") is not None
                and (proposal_threshold is None or proposal_threshold <= 0.0)
            )
            or (map_run_length is None)
            or (current_due is None)
            or (original_due is None)
            or (current_due < proposal)
        ):
            return None
        if quarantine is None:
            original_due = current_due
            normalized_quarantine = None
        else:
            if original_due < proposal:
                return None
            normalized_quarantine = cls._validated_quarantine(
                quarantine, original_due, current_due
            )
            if normalized_quarantine is None:
                return None
        confirmation_step = pending["confirmation_step"]
        confirmation_probability = pending["confirmation_probability"]
        if confirmation_step is None and confirmation_probability is None:
            normalized_confirmation_step = None
            normalized_confirmation_probability = None
        else:
            normalized_confirmation_step = cls._valid_integer(confirmation_step)
            normalized_confirmation_probability = cls._finite_scalar(
                confirmation_probability
            )
            if (
                normalized_confirmation_step is None
                or normalized_confirmation_probability is None
                or normalized_confirmation_step < current_due
            ):
                return None
        normalized = {
            "source": source,
            "proposal_step": proposal,
            "proposal_probability": probability,
            "proposal_threshold": proposal_threshold,
            "proposal_map_run_length": map_run_length,
            "confirmation_due_step": current_due,
            "original_confirmation_due_step": original_due,
            "quarantine": normalized_quarantine,
            "confirmation_step": normalized_confirmation_step,
            "confirmation_probability": normalized_confirmation_probability,
        }
        if "runtime_theta_history" in pending:
            runtime_theta_history = pending["runtime_theta_history"]
            if not isinstance(runtime_theta_history, list):
                return None
            normalized_history = []
            for sample in runtime_theta_history:
                if not isinstance(sample, dict) or set(sample) != {
                    "runtime_step",
                    "theta_flat",
                }:
                    return None
                runtime_step = cls._valid_integer(sample["runtime_step"], minimum=1)
                theta_flat = sample["theta_flat"]
                if (
                    runtime_step is None
                    or not torch.is_tensor(theta_flat)
                    or theta_flat.ndim != 1
                ):
                    return None
                normalized_history.append(
                    {
                        "runtime_step": runtime_step,
                        "theta_flat": theta_flat.detach().clone(),
                    }
                )
            normalized["runtime_theta_history"] = normalized_history
        if "same_version_proposal_raw_evidence" in pending:
            evidence = cls._validated_same_version_proposal_raw_evidence(
                pending["same_version_proposal_raw_evidence"], proposal_step=proposal
            )
            if evidence is None:
                return None
            normalized["same_version_proposal_raw_evidence"] = evidence
        if "physical_response_proposal_evidence" in pending:
            evidence = cls._validated_physical_response_proposal_evidence(
                pending["physical_response_proposal_evidence"], proposal_step=proposal
            )
            if evidence is None:
                return None
            normalized["physical_response_proposal_evidence"] = evidence
        return normalized

    @classmethod
    def _validated_physical_response_proposal_evidence(cls, value, *, proposal_step):
        required = {
            "schema",
            "version",
            "proposal_step",
            "window",
            "detector_guard_window",
            "calibration_block_count",
            "future_window_count",
            "required_support_count",
            "mutual_required_count",
            "ridge",
            "min_std",
            "action_dim",
            "observation_dim",
            "episode_segment_id",
            "current_episode_start_step",
            "pre_start_step",
            "pre_end_step",
            "response_matrices",
            "effective_ranks",
            "null_scores",
            "null_threshold",
        }
        raw_version = value.get("version") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value.get("schema") != "m3wm_physical_response_proposal_evidence"
            or isinstance(raw_version, (bool, np.bool_))
            or (not isinstance(raw_version, (int, np.integer)))
            or (int(raw_version) not in {1, 2})
        ):
            return None
        version = int(raw_version)
        bound_step = cls._valid_integer(value.get("proposal_step"), minimum=1)
        window = cls._valid_integer(value.get("window"), minimum=2)
        guard = cls._valid_integer(value.get("detector_guard_window"), minimum=1)
        block_count = cls._valid_integer(
            value.get("calibration_block_count"), minimum=3
        )
        future_count = cls._valid_integer(value.get("future_window_count"), minimum=2)
        support_count = cls._valid_integer(
            value.get("required_support_count"), minimum=1
        )
        mutual_count = cls._valid_integer(value.get("mutual_required_count"), minimum=1)
        action_dim = cls._valid_integer(value.get("action_dim"), minimum=1)
        observation_dim = cls._valid_integer(value.get("observation_dim"), minimum=1)
        segment_id = cls._valid_integer(value.get("episode_segment_id"), minimum=0)
        current_start = cls._valid_integer(
            value.get("current_episode_start_step"), minimum=1
        )
        pre_start = cls._valid_integer(value.get("pre_start_step"), minimum=1)
        pre_end = cls._valid_integer(value.get("pre_end_step"), minimum=1)
        ridge = cls._finite_scalar(value.get("ridge"))
        min_std = cls._finite_scalar(value.get("min_std"))
        threshold = cls._finite_scalar(value.get("null_threshold"))
        if (
            bound_step != int(proposal_step)
            or window is None
            or guard is None
            or (block_count is None)
            or (block_count != 5)
            or (future_count != 4)
            or (support_count != 3)
            or (mutual_count != (2 if version == 1 else 3))
            or (action_dim is None)
            or (observation_dim is None)
            or (segment_id is None)
            or (current_start is None)
            or (pre_start is None)
            or (pre_end is None)
            or (pre_end != bound_step - guard)
            or (pre_end - pre_start + 1 != block_count * window)
            or (ridge is None)
            or (ridge <= 0.0)
            or (min_std is None)
            or (min_std <= 0.0)
            or (threshold is None)
            or (threshold <= 0.0)
        ):
            return None
        try:
            matrices = (
                torch.as_tensor(value["response_matrices"])
                .detach()
                .float()
                .cpu()
                .clone()
            )
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None
        if matrices.shape != (block_count, action_dim, observation_dim) or not bool(
            torch.isfinite(matrices).all()
        ):
            return None
        ranks = value.get("effective_ranks")
        if not isinstance(ranks, list) or len(ranks) != block_count:
            return None
        normalized_ranks = []
        for raw_rank in ranks:
            rank = cls._valid_integer(raw_rank, minimum=1)
            if rank is None or rank != action_dim:
                return None
            normalized_ranks.append(rank)
        raw_scores = value.get("null_scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != block_count - 1:
            return None
        scores = []
        try:
            for index, raw_score in enumerate(raw_scores):
                score = cls._finite_scalar(raw_score)
                recomputed = float(
                    conditioned_action_response_distance(
                        matrices[index], matrices[index + 1]
                    )
                )
                if (
                    score is None
                    or score < 0.0
                    or (not np.isclose(score, recomputed, rtol=1e-06, atol=1e-07))
                ):
                    return None
                scores.append(float(score))
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None
        if not np.isclose(threshold, max(scores), rtol=1e-06, atol=1e-07):
            return None
        return {
            "schema": value["schema"],
            "version": 2,
            "proposal_step": bound_step,
            "window": window,
            "detector_guard_window": guard,
            "calibration_block_count": block_count,
            "future_window_count": future_count,
            "required_support_count": support_count,
            "mutual_required_count": future_count - 1,
            "ridge": float(ridge),
            "min_std": float(min_std),
            "action_dim": action_dim,
            "observation_dim": observation_dim,
            "episode_segment_id": segment_id,
            "current_episode_start_step": current_start,
            "pre_start_step": pre_start,
            "pre_end_step": pre_end,
            "response_matrices": matrices,
            "effective_ranks": normalized_ranks,
            "null_scores": scores,
            "null_threshold": float(threshold),
        }

    @classmethod
    def _validated_same_version_proposal_raw_evidence(cls, value, *, proposal_step):
        required = {
            "schema",
            "version",
            "kind",
            "proposal_step",
            "short_window",
            "long_window",
            "episode_segment_id",
            "current_episode_start_step",
            "previous_episode_start_step",
            "reference_samples",
            "recent_samples",
        }
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value.get("schema") != "m3wm_same_version_proposal_raw_evidence"
            or (value.get("version") != 1)
            or (value.get("kind") not in {"rolling", "episode_phase"})
        ):
            return None
        bound_step = cls._valid_integer(value.get("proposal_step"), minimum=1)
        short_window = cls._valid_integer(value.get("short_window"), minimum=1)
        long_window = cls._valid_integer(value.get("long_window"), minimum=2)
        segment_id = cls._valid_integer(value.get("episode_segment_id"), minimum=0)
        current_start = cls._valid_integer(
            value.get("current_episode_start_step"), minimum=1
        )
        previous_start = value.get("previous_episode_start_step")
        if previous_start is not None:
            previous_start = cls._valid_integer(previous_start, minimum=1)
        if (
            bound_step != int(proposal_step)
            or short_window is None
            or long_window is None
            or (short_window >= long_window)
            or (segment_id is None)
            or (current_start is None)
            or (value["kind"] == "episode_phase" and previous_start is None)
            or (value["kind"] == "rolling" and previous_start is not None)
        ):
            return None

        def validated_samples(raw_samples, expected_length):
            if not isinstance(raw_samples, list) or len(raw_samples) != expected_length:
                return None
            cloned = []
            previous = None
            for sample in raw_samples:
                if not isinstance(sample, (list, tuple)) or len(sample) != 6:
                    return None
                step = cls._valid_integer(sample[0], minimum=1)
                try:
                    values = [
                        torch.as_tensor(sample[index])
                        .detach()
                        .reshape(-1)
                        .cpu()
                        .clone()
                        for index in range(1, 5)
                    ]
                    task = (
                        None
                        if sample[5] is None
                        else torch.as_tensor(sample[5])
                        .detach()
                        .reshape(-1)
                        .cpu()
                        .clone()
                    )
                except (TypeError, ValueError, RuntimeError, OverflowError):
                    return None
                if (
                    step is None
                    or (previous is not None and step != previous + 1)
                    or any((item.numel() < 1 for item in values))
                    or any(
                        (
                            not bool(torch.isfinite(item.float()).all())
                            for item in values
                        )
                    )
                    or (
                        task is not None
                        and (
                            task.numel() != 1
                            or not bool(torch.isfinite(task.float()).all())
                        )
                    )
                ):
                    return None
                cloned.append((step, *values, task))
                previous = step
            return cloned

        reference = validated_samples(
            value["reference_samples"], long_window - short_window
        )
        recent = validated_samples(value["recent_samples"], short_window)
        if reference is None or recent is None:
            return None
        if recent[-1][0] != bound_step:
            return None
        if value["kind"] == "rolling":
            if (
                reference[0][0] != bound_step - long_window + 1
                or reference[-1][0] != bound_step - short_window
                or recent[0][0] != bound_step - short_window + 1
                or (reference[-1][0] + 1 != recent[0][0])
            ):
                return None
        else:
            if long_window != 2 * short_window:
                return None
            phase_offset = recent[0][0] - current_start
            if (
                phase_offset < 0
                or reference[0][0] != previous_start + phase_offset
                or reference[-1][0] != previous_start + phase_offset + short_window - 1
            ):
                return None
        return {
            "schema": value["schema"],
            "version": 1,
            "kind": value["kind"],
            "proposal_step": bound_step,
            "short_window": short_window,
            "long_window": long_window,
            "episode_segment_id": segment_id,
            "current_episode_start_step": current_start,
            "previous_episode_start_step": previous_start,
            "reference_samples": reference,
            "recent_samples": recent,
        }

    def store(
        self,
        source,
        proposal_step,
        probability,
        map_run_length,
        confirmation_due_step,
        original_confirmation_due_step=None,
        quarantine=None,
        proposal_threshold=None,
        same_version_proposal_raw_evidence=None,
        physical_response_proposal_evidence=None,
    ) -> bool:
        if self._pending is not None and self._pending.get("confirmation_step") is None:
            self.drop_count += 1
            return False
        original_due = (
            confirmation_due_step
            if original_confirmation_due_step is None
            else original_confirmation_due_step
        )
        candidate = {
            "source": source,
            "proposal_step": proposal_step,
            "proposal_probability": probability,
            "proposal_threshold": proposal_threshold,
            "proposal_map_run_length": map_run_length,
            "confirmation_due_step": confirmation_due_step,
            "original_confirmation_due_step": original_due,
            "quarantine": deepcopy(quarantine),
            "confirmation_step": None,
            "confirmation_probability": None,
        }
        if same_version_proposal_raw_evidence is not None:
            candidate["same_version_proposal_raw_evidence"] = deepcopy(
                same_version_proposal_raw_evidence
            )
        if physical_response_proposal_evidence is not None:
            candidate["physical_response_proposal_evidence"] = deepcopy(
                physical_response_proposal_evidence
            )
        validated = self._validated_pending(candidate)
        if validated is None:
            return False
        self._pending = validated
        return True

    def supersede(
        self,
        source,
        proposal_step,
        probability,
        map_run_length,
        confirmation_due_step,
        original_confirmation_due_step=None,
        quarantine=None,
        proposal_threshold=None,
        same_version_proposal_raw_evidence=None,
        physical_response_proposal_evidence=None,
    ):
        old = self.get()
        if (
            not self.active
            or not isinstance(old, dict)
            or int(proposal_step) <= int(old.get("proposal_step", -1))
        ):
            return None
        old_snapshot = deepcopy(old)
        original = self._pending
        self._pending = None
        accepted = self.store(
            source=source,
            proposal_step=proposal_step,
            probability=probability,
            map_run_length=map_run_length,
            confirmation_due_step=confirmation_due_step,
            original_confirmation_due_step=original_confirmation_due_step,
            quarantine=quarantine,
            proposal_threshold=proposal_threshold,
            same_version_proposal_raw_evidence=same_version_proposal_raw_evidence,
            physical_response_proposal_evidence=physical_response_proposal_evidence,
        )
        if not accepted:
            self._pending = original
            return None
        old_snapshot["terminal_outcome"] = "UNKNOWN_SUPERSEDED"
        old_snapshot["superseded_by_proposal_step"] = int(proposal_step)
        return old_snapshot

    def confirm(self, step, probability) -> bool:
        if self._pending is None or self._pending.get("confirmation_step") is not None:
            return False
        confirmation_step = self._valid_integer(step)
        confirmation_probability = self._finite_scalar(probability)
        if (
            confirmation_step is None
            or confirmation_probability is None
            or confirmation_step < self._pending["confirmation_due_step"]
        ):
            return False
        self._pending["confirmation_step"] = confirmation_step
        self._pending["confirmation_probability"] = confirmation_probability
        return True

    def defer_quarantine(
        self,
        step,
        *,
        active_context_id,
        candidate_context_id,
        evidence,
        quarantine_state=None,
    ) -> bool:
        if not self.active:
            return False
        current_due = int(self._pending["confirmation_due_step"])
        original_due = int(self._pending["original_confirmation_due_step"])
        requested_step = self._valid_integer(step)
        if (
            requested_step != current_due
            or current_due != original_due
            or self._pending.get("quarantine") is not None
        ):
            return False
        retry_due = original_due + 128
        try:
            evidence_sha256 = canonical_json_sha256(evidence)
        except (TypeError, ValueError, OverflowError):
            return False
        metadata = {
            "retry_count": 1,
            "deferred_step": original_due,
            "retry_confirmation_due_step": retry_due,
            "active_context_id": active_context_id,
            "candidate_context_id": candidate_context_id,
            "evidence": deepcopy(evidence),
            "evidence_sha256": evidence_sha256,
            "quarantine_state": deepcopy(quarantine_state),
        }
        validated = self._validated_quarantine(metadata, original_due, retry_due)
        if validated is None:
            return False
        self._pending["confirmation_due_step"] = retry_due
        self._pending["quarantine"] = validated
        return True

    def defer_unresolved_identity(self, step, *, interval, max_horizon) -> bool:
        if not self.active or self._pending.get("quarantine") is not None:
            return False
        requested_step = self._valid_integer(step, minimum=1)
        interval = self._valid_integer(interval, minimum=1)
        max_horizon = self._valid_integer(max_horizon, minimum=1)
        if requested_step is None or interval is None or max_horizon is None:
            return False
        proposal = int(self._pending["proposal_step"])
        current_due = int(self._pending["confirmation_due_step"])
        if (
            requested_step != current_due
            or current_due < proposal + interval
            or (current_due - proposal) % interval != 0
        ):
            return False
        proposal_deadline = proposal + max_horizon
        next_due = min(current_due + interval, proposal_deadline)
        if next_due <= current_due or next_due > proposal + max_horizon:
            return False
        self._pending["confirmation_due_step"] = int(next_due)
        self._pending["original_confirmation_due_step"] = int(next_due)
        return True

    def start_runtime_theta_history(self) -> bool:
        if not self.active:
            return False
        self._pending["runtime_theta_history"] = []
        return True

    def append_runtime_theta(self, runtime_step, theta, maxlen) -> bool:
        window = self._valid_integer(maxlen, minimum=1)
        step = self._valid_integer(runtime_step, minimum=1)
        if (
            not self.active
            or window is None
            or step is None
            or ("runtime_theta_history" not in self._pending)
            or (not torch.is_tensor(theta))
        ):
            return False
        history = self._pending["runtime_theta_history"]
        history.append(
            {"runtime_step": step, "theta_flat": theta.detach().reshape(-1).clone()}
        )
        if len(history) > window:
            del history[:-window]
        return True

    def get_runtime_theta_history(self):
        if self._pending is None:
            return None
        history = self._pending.get("runtime_theta_history")
        return None if history is None else tuple(history)

    def consume_confirmed(self) -> dict | None:
        if self._pending is None or self._pending.get("confirmation_step") is None:
            return None
        result = self._pending
        self._pending = None
        result.pop("runtime_theta_history", None)
        return result

    def clear(self) -> None:
        self.cancel_pending()
        self.drop_count = 0

    def cancel_pending(self) -> None:
        self._pending = None

    def state_dict(self) -> dict:
        return {
            "pending": None if self._pending is None else deepcopy(self._pending),
            "drop_count": int(self.drop_count),
        }

    def load_state_dict(self, state) -> None:
        if not isinstance(state, dict):
            self.clear()
            return
        pending = deepcopy(state.get("pending"))
        drop_count = self._valid_integer(state.get("drop_count", 0))
        if drop_count is None:
            self.clear()
            return
        if pending is None:
            self._pending = None
            self.drop_count = drop_count
            return
        validated = self._validated_pending(pending)
        if validated is None:
            self.clear()
            return
        self._pending = validated
        self.drop_count = drop_count

    @property
    def active(self) -> bool:
        return (
            self._pending is not None and self._pending.get("confirmation_step") is None
        )

    def get(self) -> dict | None:
        return self._pending


def _none_to_nan(value):
    return float("nan") if value is None else float(value)


def _validated_pending_persistence_runtime_contract(cfg, confirmed_departure_enabled):
    if not isinstance(False, (bool, np.bool_)):
        raise ValueError("m3wm_dp_pending_persistence_required must be boolean")
    return False


def _context_exemplar_not_evaluated_receipt(
    *, proposal_step=None, source_context_id=None
):
    bound_step = None if proposal_step is None else int(proposal_step)
    bound_source = None if source_context_id is None else int(source_context_id)
    return {
        "valid": False,
        "evaluated_this_transaction": False,
        "action": "not_evaluated",
        "predictive_action": "not_evaluated",
        "reason": "not_evaluated",
        "failure_stage": "not_evaluated",
        "selected_context_id": None,
        "supported_context_ids": [],
        "contexts": {},
        "proposal_token": None if bound_step is None else ("m3wm-proposal", bound_step),
        "proposal_step": -1 if bound_step is None else bound_step,
        "due_step": -1,
        "source_context_id": bound_source,
    }


class TDMPC2(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._m3wm_eval_mpc_action_mode = _validated_m3wm_eval_mpc_action_mode(
            self.cfg.get("m3wm_eval_mpc_action_mode", "sample_elite")
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_episode_reset_from_memory_theta must be boolean")
        self._m3wm_episode_reset_from_memory_theta = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_new_context_inherit_pending_runtime_theta must be boolean"
            )
        self._m3wm_new_context_inherit_pending_runtime_theta = False
        raw_inherit_theta_window = self.cfg.get(
            "m3wm_new_context_inherit_theta_window", 128
        )
        if (
            isinstance(raw_inherit_theta_window, (bool, np.bool_))
            or not isinstance(raw_inherit_theta_window, (int, np.integer))
            or int(raw_inherit_theta_window) < 1
        ):
            raise ValueError(
                "m3wm_new_context_inherit_theta_window must be a positive integer"
            )
        self._m3wm_new_context_inherit_theta_window = int(raw_inherit_theta_window)
        raw_inherit_theta_max_std = self.cfg.get(
            "m3wm_new_context_inherit_theta_max_std", 0.15
        )
        if isinstance(raw_inherit_theta_max_std, (str, bool, np.bool_)):
            raise ValueError(
                "m3wm_new_context_inherit_theta_max_std must be a finite non-negative number"
            )
        try:
            inherit_theta_max_std = float(raw_inherit_theta_max_std)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "m3wm_new_context_inherit_theta_max_std must be a finite non-negative number"
            ) from error
        if not np.isfinite(inherit_theta_max_std) or inherit_theta_max_std < 0.0:
            raise ValueError(
                "m3wm_new_context_inherit_theta_max_std must be a finite non-negative number"
            )
        self._m3wm_new_context_inherit_theta_max_std = inherit_theta_max_std
        if not isinstance(True, (bool, np.bool_)):
            raise ValueError(
                "m3wm_online_theta_writeback_require_unlatched must be boolean"
            )
        self._m3wm_online_theta_writeback_require_unlatched = True
        raw_post_terminal_writeback_clean_steps = self.cfg.get(
            "m3wm_post_terminal_writeback_clean_steps", 0
        )
        if (
            isinstance(raw_post_terminal_writeback_clean_steps, (bool, np.bool_))
            or not isinstance(
                raw_post_terminal_writeback_clean_steps, (int, np.integer)
            )
            or int(raw_post_terminal_writeback_clean_steps) < 0
        ):
            raise ValueError(
                "m3wm_post_terminal_writeback_clean_steps must be a non-negative integer"
            )
        self._m3wm_post_terminal_writeback_clean_steps = int(
            raw_post_terminal_writeback_clean_steps
        )
        raw_clean_episode_update_limit = self.cfg.get(
            "m3wm_online_theta_clean_episode_update_limit", 0
        )
        if (
            isinstance(raw_clean_episode_update_limit, (bool, np.bool_))
            or not isinstance(raw_clean_episode_update_limit, (int, np.integer))
            or int(raw_clean_episode_update_limit) < 0
        ):
            raise ValueError(
                "m3wm_online_theta_clean_episode_update_limit must be a non-negative integer"
            )
        self._m3wm_online_theta_clean_episode_update_limit = int(
            raw_clean_episode_update_limit
        )
        raw_transaction_update_limit = self.cfg.get(
            "m3wm_online_theta_transaction_update_limit", 0
        )
        if (
            isinstance(raw_transaction_update_limit, (bool, np.bool_))
            or not isinstance(raw_transaction_update_limit, (int, np.integer))
            or int(raw_transaction_update_limit) < 0
        ):
            raise ValueError(
                "m3wm_online_theta_transaction_update_limit must be a non-negative integer"
            )
        self._m3wm_online_theta_transaction_update_limit = int(
            raw_transaction_update_limit
        )
        if not isinstance(True, (bool, np.bool_)):
            raise ValueError("m3wm_episodic_theta_reward_guard_enabled must be boolean")
        self._m3wm_episodic_theta_reward_guard_enabled = True
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_episodic_theta_baseline_first_enabled must be boolean"
            )
        self._m3wm_episodic_theta_baseline_first_enabled = False
        raw_guard_min_tail_samples = self.cfg.get(
            "m3wm_episodic_theta_reward_guard_min_tail_samples",
            _M3WM_EPISODIC_THETA_REWARD_BLOCK * _M3WM_EPISODIC_THETA_MIN_BLOCKS,
        )
        if (
            isinstance(raw_guard_min_tail_samples, (bool, np.bool_))
            or not isinstance(raw_guard_min_tail_samples, (int, np.integer))
            or int(raw_guard_min_tail_samples)
            < _M3WM_EPISODIC_THETA_REWARD_BLOCK * _M3WM_EPISODIC_THETA_MIN_BLOCKS
            or (
                int(raw_guard_min_tail_samples) % _M3WM_EPISODIC_THETA_REWARD_BLOCK != 0
            )
        ):
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_min_tail_samples must be an integer containing at least four complete 128-step blocks"
            )
        self._m3wm_episodic_theta_reward_guard_min_tail_samples = int(
            raw_guard_min_tail_samples
        )
        raw_guard_margin = self.cfg.get(
            "m3wm_episodic_theta_reward_guard_improvement_margin", 0.005
        )
        if isinstance(raw_guard_margin, (str, bool, np.bool_)):
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_improvement_margin must be a finite non-negative number"
            )
        try:
            guard_margin = float(raw_guard_margin)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_improvement_margin must be a finite non-negative number"
            ) from error
        if not np.isfinite(guard_margin) or guard_margin < 0.0:
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_improvement_margin must be a finite non-negative number"
            )
        self._m3wm_episodic_theta_reward_guard_improvement_margin = guard_margin
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_episodic_theta_lock_after_accept must be boolean")
        self._m3wm_episodic_theta_lock_after_accept = False
        raw_lock_gain = self.cfg.get(
            "m3wm_episodic_theta_reward_guard_lock_cumulative_gain", 0.03
        )
        if isinstance(raw_lock_gain, (str, bool, np.bool_)):
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_lock_cumulative_gain must be a finite non-negative number"
            )
        try:
            lock_gain = float(raw_lock_gain)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_lock_cumulative_gain must be a finite non-negative number"
            ) from error
        if not np.isfinite(lock_gain) or lock_gain < 0.0:
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_lock_cumulative_gain must be a finite non-negative number"
            )
        self._m3wm_episodic_theta_reward_guard_lock_cumulative_gain = lock_gain
        if not isinstance(True, (bool, np.bool_)):
            raise ValueError(
                "m3wm_post_create_quiet_prototype_maturation_enabled must be boolean"
            )
        self._m3wm_post_create_quiet_prototype_maturation_enabled = True
        if self._m3wm_online_theta_clean_episode_update_limit <= 0:
            raise ValueError(
                "m3wm_episodic_theta_reward_guard_enabled requires a positive m3wm_online_theta_clean_episode_update_limit"
            )
        (
            self._m3wm_branchwise_theta_trust_region_enabled,
            self._m3wm_branchwise_theta_trust_region_radius,
        ) = _validated_m3wm_branchwise_theta_trust_region(
            False, self.cfg.get("m3wm_branchwise_theta_trust_region_radius", 0.0)
        )
        offline_replay_device = self.cfg.get("m3wm_identity_replay_device", None)
        if offline_replay_device is not None:
            if self.cfg.get("m3wm_identity_replay_mode", False) is not True or str(
                offline_replay_device
            ) not in {"cpu", "cuda:0"}:
                raise ValueError("Invalid offline identity replay device")
            self.device = torch.device(str(offline_replay_device))
        else:
            self.device = torch.device("cuda:0")
        self.model = WorldModel(cfg).to(self.device)
        enable_m3wm = self.cfg.get("enable_m3wm", False)
        system_id_routing_loss = torch.zeros((), device=self.device)
        if enable_m3wm:
            warmup_steps = int(self.cfg.get("m3wm_warmup_steps", 5000))
            specialized_lr = 0.0 if warmup_steps > 0 else self.cfg.lr
            if self.model.m3wm_composition_mode in {"modular", "hybrid_modular"}:
                if self.model.m3wm_modular_value:
                    q_base_params = []
                    q_primitive_params = []
                    for name, parameter in self.model._Qs.named_parameters():
                        if "residual_bank" in name:
                            q_primitive_params.append(parameter)
                        else:
                            q_base_params.append(parameter)
                    q_param_groups = [
                        {
                            "params": q_base_params,
                            "lr": self.cfg.lr,
                            "name": "m3wm_value_base",
                        },
                        {
                            "params": q_primitive_params,
                            "lr": specialized_lr,
                            "name": "m3wm_value_primitives",
                        },
                    ]
                else:
                    q_param_groups = [
                        {"params": self.model._Qs.parameters(), "name": "q_functions"}
                    ]
                model_param_groups = [
                    {
                        "params": self.model._encoder.parameters(),
                        "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                        "name": "encoder",
                    },
                    {
                        "params": self.model._dynamics.base.parameters(),
                        "lr": self.cfg.lr,
                        "name": "m3wm_dynamics_base",
                    },
                    {
                        "params": self.model._dynamics.residual_bank.parameters(),
                        "lr": specialized_lr,
                        "name": "m3wm_dynamics_primitives",
                    },
                    {
                        "params": self.model._reward.base.parameters(),
                        "lr": self.cfg.lr,
                        "name": "m3wm_reward_base",
                    },
                    {
                        "params": self.model._reward.residual_bank.parameters(),
                        "lr": specialized_lr,
                        "name": "m3wm_reward_primitives",
                    },
                    {
                        "params": self.model._m3wm_context.parameters(),
                        "lr": specialized_lr,
                        "name": "m3wm_context",
                    },
                    {
                        "params": (
                            self.model._termination.parameters()
                            if self.cfg.episodic
                            else []
                        ),
                        "name": "termination",
                    },
                    *q_param_groups,
                    {"params": [], "name": "task_embedding"},
                ]
            else:
                model_param_groups = [
                    {
                        "params": self.model._encoder.parameters(),
                        "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                        "name": "encoder",
                    },
                    {
                        "params": self.model._dynamics.parameters(),
                        "lr": self.cfg.lr,
                        "name": "baseline_dynamics",
                    },
                    {
                        "params": self.model._reward.parameters(),
                        "lr": self.cfg.lr,
                        "name": "baseline_reward",
                    },
                    {
                        "params": self.model._m3wm_context.parameters(),
                        "lr": specialized_lr,
                        "name": "m3wm_context",
                    },
                    {
                        "params": (
                            self.model._termination.parameters()
                            if self.cfg.episodic
                            else []
                        ),
                        "name": "termination",
                    },
                    {
                        "params": self.model._Qs.parameters(),
                        "lr": self.cfg.lr,
                        "name": "baseline_q_functions",
                    },
                    {"params": [], "name": "task_embedding"},
                ]
        else:
            model_param_groups = [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {
                    "params": (
                        self.model._termination.parameters()
                        if self.cfg.episodic
                        else []
                    )
                },
                {"params": self.model._Qs.parameters()},
                {"params": []},
            ]
        self.optim = torch.optim.Adam(
            model_param_groups, lr=self.cfg.lr, capturable=True
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-05, capturable=True
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_conservative_actor_update_enabled must be boolean")
        self._m3wm_conservative_actor_update_enabled = False
        raw_actor_interval = self.cfg.get("m3wm_actor_update_interval", 1)
        if (
            isinstance(raw_actor_interval, (bool, np.bool_))
            or not isinstance(raw_actor_interval, (int, np.integer))
            or int(raw_actor_interval) < 1
        ):
            raise ValueError("m3wm_actor_update_interval must be a positive integer")
        self._m3wm_actor_update_interval = int(raw_actor_interval)
        for key, default in (
            ("m3wm_conservative_q_beta", 0.0),
            ("m3wm_actor_ema_tau", 0.01),
            ("m3wm_actor_ema_coef", 0.0),
        ):
            raw_value = self.cfg.get(key, default)
            if isinstance(raw_value, (str, bool, np.bool_)):
                raise ValueError(f"{key} must be a finite non-negative number")
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{key} must be a finite non-negative number"
                ) from error
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be a finite non-negative number")
            if key == "m3wm_actor_ema_tau" and value > 1.0:
                raise ValueError("m3wm_actor_ema_tau must be in [0, 1]")
            setattr(self, f"_{key}", value)
        self._m3wm_control_update_count = 0
        self._m3wm_last_pi_update_applied = 0.0
        self._m3wm_last_pi_q_std = 0.0
        self._m3wm_last_pi_ema_penalty = 0.0
        self._m3wm_actor_ema = None
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)
        self.discount = self._get_discount(cfg.episode_length)
        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)
        self._prev_mean = torch.nn.Buffer(
            torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        )
        self._plain_incumbent_proposal_enabled = True
        self._plain_incumbent_proposal_tolerance = float(
            self.cfg.get("m3wm_plain_incumbent_proposal_tolerance", 1e-06)
        )
        if (
            not np.isfinite(self._plain_incumbent_proposal_tolerance)
            or self._plain_incumbent_proposal_tolerance < 0.0
        ):
            raise ValueError(
                "m3wm_plain_incumbent_proposal_tolerance must be finite and non-negative"
            )
        self._plain_incumbent_proposal_warmup_steps = int(
            self.cfg.get("m3wm_plain_proposal_warmup_steps", 0)
        )
        self._plain_incumbent_proposal_min_accept_interval = int(
            self.cfg.get("m3wm_plain_proposal_min_accept_interval", 0)
        )
        if (
            self._plain_incumbent_proposal_warmup_steps < 0
            or self._plain_incumbent_proposal_min_accept_interval < 0
        ):
            raise ValueError(
                "proposal warmup and minimum accept interval must be non-negative"
            )
        self._plain_incumbent_last_accept_step = -(10**18)
        self._plain_incumbent_plan_result = None
        self._plain_incumbent_proposal_metrics = {
            "proposal_enabled": 1,
            "proposal_valid": 0,
            "proposal_accepted": 0,
            "proposal_rejected_reason": "not_evaluated",
            "proposal_context_id": -1,
            "proposal_theta_version": -1,
            "proposal_model_version": -1,
            "proposal_count": 0,
            "proposal_control_step": -1,
            "proposal_last_accept_step": -1,
            "paired_plain_min_delta": float("nan"),
            "paired_plain_score_delta_per_head": "[]",
            "plain_q_head_count": int(self.cfg.num_q),
            "plain_q_head_indices": json.dumps(list(range(int(self.cfg.num_q)))),
            "paired_plain_score_shape": "[]",
            "proposal_plain_rescore_ms": 0.0,
            "executed_action_source": "plain",
        }
        self._plain_incumbent_score_generator = None
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self._plain_incumbent_score_generator = torch.Generator(device=generator_device)
        self._plain_incumbent_score_generator.manual_seed(
            int(self.cfg.get("seed", 0)) + 104729
        )
        (
            self._m3wm_signed_reward_slow_enabled,
            self._m3wm_signed_reward_slow_window,
            self._m3wm_signed_reward_slow_min_consistency,
        ) = _validated_signed_reward_slow_config(
            self.cfg,
            self.cfg.get("m3wm_change_statistic", ""),
            getattr(self.model, "m3wm_use_reward_residual", False),
        )
        self._m3wm_context_history = deque(
            maxlen=(
                max(
                    int(self.cfg.get("m3wm_context_window", 32)),
                    int(self.cfg.get("m3wm_memory_min_history", 1)),
                )
                if self.cfg.get("enable_m3wm", False)
                else 1
            )
        )
        context_raw_maxlen = max(
            int(self.cfg.get("m3wm_context_window", 32)),
            int(self.cfg.get("m3wm_memory_min_history", 1)),
            int(
                self.cfg.get(
                    "m3wm_memory_window", self.cfg.get("m3wm_context_window", 32)
                )
            ),
            int(
                self.cfg.get(
                    "m3wm_memory_trigger_window",
                    self.cfg.get("m3wm_context_short_window", 8),
                )
            ),
            int(self.cfg.get("m3wm_context_long_window", 32)),
        )
        self._m3wm_context_raw_history = deque(
            maxlen=context_raw_maxlen if self.cfg.get("enable_m3wm", False) else 1
        )
        self._m3wm_ut_trace = None
        self._m3wm_bocpd_fixed_null_samples = []
        self._m3wm_bocpd_fixed_location = None
        self._m3wm_bocpd_fixed_scale = None
        self._m3wm_bocpd_fixed_calibrated = False
        self._m3wm_change_history = deque(
            maxlen=(
                max(
                    int(
                        self.cfg.get(
                            "m3wm_change_long_window",
                            self.cfg.get("m3wm_context_long_window", 32),
                        )
                    ),
                    int(self.cfg.get("m3wm_memory_trigger_window", 8)),
                    int(self.cfg.get("m3wm_trigger_confirmation_delay", 0)),
                    (
                        2 * self._m3wm_signed_reward_slow_window
                        if self._m3wm_signed_reward_slow_enabled
                        else 1
                    ),
                )
                if self.cfg.get("enable_m3wm", False)
                else 1
            )
        )
        self._m3wm_current_episode_change_history = []
        self._m3wm_previous_episode_change_history = []
        self._m3wm_current_episode_system_id = []
        self._m3wm_previous_episode_system_id = []
        self._m3wm_local_system_id_history = deque(
            maxlen=(
                2
                * max(
                    2,
                    int(
                        self.cfg.get(
                            "m3wm_local_system_id_window",
                            self.cfg.get("m3wm_system_id_window", 32),
                        )
                    ),
                )
                if self.cfg.get("enable_m3wm", False)
                else 1
            )
        )
        detector_raw_long_window = max(
            int(self.cfg.get("m3wm_change_long_window", 32)),
            2 * int(getattr(self, "_m3wm_signed_reward_slow_window", 32)),
        )
        self._m3wm_detector_raw_history = deque(
            maxlen=(
                max(
                    1,
                    int(self.cfg.get("m3wm_regime_confirmation_delay", 160))
                    + detector_raw_long_window
                    + 1,
                )
                if self.cfg.get("enable_m3wm", False)
                else 1
            )
        )
        self._m3wm_detector_episode_segment_id = 0
        self._m3wm_detector_episode_start_step = 1
        self._m3wm_detector_episode_prefix_valid = True
        self._m3wm_detector_current_episode_prefix = []
        self._m3wm_detector_previous_episode_start_step = None
        self._m3wm_detector_previous_episode_end_step = None
        self._m3wm_detector_previous_episode_prefix_valid = False
        self._m3wm_detector_previous_episode_prefix = []
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_context_exemplar_identity_enabled must be boolean")
        self._m3wm_context_exemplar_identity_enabled = False
        self._m3wm_context_exemplar_support_mode = str(
            self.cfg.get("m3wm_context_exemplar_support_mode", "gaussian")
        ).lower()
        if self._m3wm_context_exemplar_support_mode not in {
            "gaussian",
            "nearest_exemplar",
        }:
            raise ValueError("Invalid raw Context exemplar support mode")
        self._m3wm_context_exemplar_windows_per_segment = int(
            self.cfg.get("m3wm_context_exemplar_windows_per_segment", 1)
        )
        self._m3wm_context_exemplar_min_segments = int(
            self.cfg.get("m3wm_context_exemplar_min_segments", 1)
        )
        self._m3wm_context_exemplar_query_window_count = int(
            self.cfg.get("m3wm_context_exemplar_query_window_count", 1)
        )
        self._m3wm_context_exemplar_query_required_support = int(
            self.cfg.get("m3wm_context_exemplar_query_required_support", 1)
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context run calibration switch must be boolean")
        self._m3wm_context_run_calibration_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context conditional-law authority switch must be boolean")
        self._m3wm_context_conditional_law_authority_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context residual-law authority switch must be boolean")
        self._m3wm_context_residual_law_authority_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context residual-rank authority switch must be boolean")
        self._m3wm_context_residual_rank_authority_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context sequential-identity switch must be boolean")
        self._m3wm_context_sequential_identity_enabled = False
        self._m3wm_context_sequential_identity_interval = int(
            self.cfg.get("m3wm_context_sequential_identity_interval", 160)
        )
        self._m3wm_context_sequential_identity_max_horizon = int(
            self.cfg.get("m3wm_context_sequential_identity_max_horizon", 960)
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Canonical identity authority switch must be boolean")
        self._m3wm_canonical_identity_authority_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_stable_context_selector_enabled must be boolean")
        self._m3wm_stable_context_selector_enabled = False
        stable_candidate_validations = int(
            self.cfg.get("m3wm_stable_context_candidate_min_validations", 2)
        )
        stable_recall_validations = int(
            self.cfg.get("m3wm_stable_context_recall_min_validations", 2)
        )
        stable_margin = float(self.cfg.get("m3wm_stable_context_min_margin", 0.05))
        stable_control_gain = float(
            self.cfg.get("m3wm_stable_context_min_control_gain", 0.0)
        )
        stable_halo_capacity = int(self.cfg.get("m3wm_stable_context_halo_capacity", 8))
        if (
            stable_candidate_validations < 2
            or stable_recall_validations < 1
            or stable_margin < 0.0
            or (stable_control_gain < 0.0)
            or (stable_halo_capacity < 1)
        ):
            raise ValueError("Invalid stable Context selector configuration")
        self._m3wm_stable_context_selector = None
        soft_reuse_enabled = self.cfg.get("m3wm_jump_drift_soft_reuse_enabled", False)
        if not isinstance(soft_reuse_enabled, (bool, np.bool_)):
            raise ValueError("m3wm_jump_drift_soft_reuse_enabled must be boolean")
        self._m3wm_jump_drift_soft_reuse_enabled = bool(soft_reuse_enabled)
        self._m3wm_soft_context_reuse = (
            SoftContextReuseController()
            if self._m3wm_jump_drift_soft_reuse_enabled
            else None
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_reversible_reuse_trial_enabled must be boolean")
        self._m3wm_reversible_reuse_trial_enabled = False
        self._m3wm_reversible_reuse_trial = None
        self._m3wm_reversible_trial_nomination_count = 0
        self._m3wm_reversible_trial_commit_count = 0
        self._m3wm_reversible_trial_abort_count = 0
        self._m3wm_last_reversible_trial_receipt = {
            "schema": "m3wm_reversible_reuse_trial",
            "version": M3WM_REVERSIBLE_REUSE_TRIAL_VERSION,
            "state": "INACTIVE",
            "active": False,
            "applied": False,
            "committed_target_context_id": None,
            "target_memory_writes": 0,
            "official_active_context_changed": False,
            "reason": "disabled",
        }
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context active-probe switch must be boolean")
        self._m3wm_context_active_probe_enabled = False
        self._m3wm_last_active_probe_receipt = {
            "schema": "m3wm_active_context_probe_runtime",
            "version": 1,
            "enabled": False,
            "override_applied": False,
            "reason": "not_evaluated",
        }
        self._m3wm_context_run_calibration_alpha = float(
            self.cfg.get("m3wm_context_run_calibration_alpha", 0.2)
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context identity event capture switch must be boolean")
        self._m3wm_context_identity_event_capture_enabled = False
        self._m3wm_last_identity_event_artifact = None
        if (
            self._m3wm_context_exemplar_windows_per_segment < 1
            or self._m3wm_context_exemplar_min_segments < 1
            or self._m3wm_context_exemplar_query_window_count < 1
            or (self._m3wm_context_exemplar_query_required_support < 1)
            or (
                self._m3wm_context_exemplar_query_required_support
                > self._m3wm_context_exemplar_query_window_count
            )
            or (not 0.0 < self._m3wm_context_run_calibration_alpha < 1.0)
        ):
            raise ValueError("Invalid raw Context exemplar coverage contract")
        self._m3wm_context_exemplar_bank = None
        self._m3wm_context_identity_model_snapshot = None
        self._m3wm_factorized_law_shadow_reference_state = (
            empty_shadow_reference_state()
        )
        for name, value in (
            ("m3wm_v66_faci_shadow_enabled", False),
            ("m3wm_v66_faci_authority_enabled", False),
            ("m3wm_v66_probe_enabled", False),
            ("m3wm_factorized_law_identity_shadow_enabled", False),
            ("m3wm_factorized_law_identity_authority_enabled", False),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        self._m3wm_v66_faci_shadow_enabled = False
        self._m3wm_v66_faci_authority_enabled = False
        self._m3wm_v66_probe_enabled = False
        self._m3wm_factorized_law_identity_shadow_enabled = False
        self._m3wm_factorized_law_identity_authority_enabled = False
        self._m3wm_factorized_law_identity_recall_margin = float(
            self.cfg.get("m3wm_factorized_law_identity_recall_margin", 0.5)
        )
        self._m3wm_factorized_law_identity_create_margin = float(
            self.cfg.get("m3wm_factorized_law_identity_create_margin", 0.5)
        )
        self._m3wm_factorized_law_identity_factor_margin = float(
            self.cfg.get("m3wm_factorized_law_identity_factor_margin", 0.25)
        )
        self._m3wm_factorized_law_identity_absolute_nll_ceiling = float(
            self.cfg.get("m3wm_factorized_law_identity_absolute_nll_ceiling", 50.0)
        )
        if any(
            (
                value <= 0.0
                for value in (
                    self._m3wm_factorized_law_identity_recall_margin,
                    self._m3wm_factorized_law_identity_create_margin,
                    self._m3wm_factorized_law_identity_factor_margin,
                    self._m3wm_factorized_law_identity_absolute_nll_ceiling,
                )
            )
        ):
            raise ValueError(
                "Factorized physical-law authority margins must be positive"
            )
        v66_latency_limit = self.cfg.get("m3wm_v66_create_latency_limit", 2000)
        if (
            isinstance(v66_latency_limit, (bool, np.bool_))
            or not isinstance(v66_latency_limit, (int, np.integer))
            or int(v66_latency_limit) != 2000
        ):
            raise ValueError(
                "m3wm_v66_create_latency_limit must equal the frozen 2000-step gate"
            )
        self._m3wm_v66_create_latency_limit = int(v66_latency_limit)
        self._m3wm_v66_exclusive_suspect_lease_steps = 320
        self._m3wm_v66_last_faci_shadow = {
            "valid": False,
            "action": "UNKNOWN",
            "reason": "feature_flag_disabled",
            "shadow_only": True,
            "decision_inert": True,
        }
        self._m3wm_v66_probe_ledger = None
        self._m3wm_v66_last_probe_certificate = None
        self._m3wm_v66_last_terminal = None
        self._m3wm_v66_faci_core_bank = empty_core_bank()
        self._m3wm_v66_pending_core_commit = None
        self._m3wm_v66_coverage_bank_cache = None
        self._m3wm_v66_coverage_bank_signature = None
        self._m3wm_v66_coverage_bank_context_count = None
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_v66_recall_wrapper_enabled must be boolean")
        self._m3wm_v66_recall_wrapper_enabled = False
        self._m3wm_v66_recall_holdout_steps = int(
            self.cfg.get("m3wm_v66_recall_holdout_steps", 128)
        )
        self._m3wm_v66_recall_max_trials = int(
            self.cfg.get("m3wm_v66_recall_max_trials", 2)
        )
        self._m3wm_v66_recall_min_relative_improvement = float(
            self.cfg.get("m3wm_v66_recall_min_relative_improvement", 0.05)
        )
        self._m3wm_v66_recall_pending = None
        self._m3wm_v66_last_recall_wrapper_receipt = {
            "schema": "m3wm_v66_recall_wrapper_decision",
            "version": 1,
            "valid": True,
            "applied": False,
            "reason": "feature_flag_disabled",
        }
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_v57_owner_shadow_enabled must be boolean")
        self._m3wm_v57_owner_shadow_enabled = False
        v63_owner_authority = self.cfg.get("m3wm_v63_owner_authority_enabled", False)
        if not isinstance(v63_owner_authority, (bool, np.bool_)):
            raise ValueError("m3wm_v63_owner_authority_enabled must be boolean")
        self._m3wm_v63_owner_authority_enabled = bool(v63_owner_authority)
        if self._m3wm_v63_owner_authority_enabled:
            raise ValueError("V63 owner authority requires the V57 evidence path")
        v65_fast2o_authority = self.cfg.get("m3wm_v65_fast2o_authority_enabled", False)
        if not isinstance(v65_fast2o_authority, (bool, np.bool_)):
            raise ValueError("m3wm_v65_fast2o_authority_enabled must be boolean")
        self._m3wm_v65_fast2o_authority_enabled = bool(v65_fast2o_authority)
        if self._m3wm_v65_fast2o_authority_enabled:
            raise ValueError(
                "V65 Fast2O authority requires the unified V57 carrier and V63 atomic authority chain"
            )
        self._m3wm_last_context_exemplar_assignment = (
            _context_exemplar_not_evaluated_receipt()
        )
        self._m3wm_v57_shadow_registration = _m3wm_v57_shadow_receipt(
            "feature_flag_disabled", state="DISABLED"
        )
        self._m3wm_v57_shadow_owner_ledger = None
        self._m3wm_v57_shadow_owner_evaluation = _m3wm_v57_shadow_receipt(
            "feature_flag_disabled", state="DISABLED"
        )
        self._m3wm_v57_shadow_candidate_self_consistency = None
        self._m3wm_v64_terminal_proof_cache = {}
        self._m3wm_v64_terminal_proof_cache_capacity = 16
        self._m3wm_v64_terminal_proof_cache_hits = 0
        self._m3wm_v64_terminal_proof_cache_misses = 0
        self._m3wm_v63_last_owner_authority = {
            "valid": False,
            "state": "NOT_EVALUATED",
            "reason": (
                "not_evaluated"
                if self._m3wm_v63_owner_authority_enabled
                else "feature_flag_disabled"
            ),
        }
        self._m3wm_v65_fast2o_last = {
            "valid": False,
            "state": "NOT_EVALUATED",
            "reason": "feature_flag_disabled",
            "may_create": False,
            "may_mutate_context_memory": False,
        }
        self._m3wm_v65_provisional_overlay = None
        self._m3wm_v65_last_fast_commit = {
            "valid": False,
            "state": "NOT_EVALUATED",
            "reason": (
                "not_evaluated"
                if self._m3wm_v65_fast2o_authority_enabled
                else "feature_flag_disabled"
            ),
        }
        self._m3wm_v57_shadow_fail_stopped = False
        self._m3wm_v57_shadow_sidecar_queue = deque()
        self._m3wm_v57_shadow_sidecar_sequence = 0
        self._m3wm_v57_shadow_last_registration_sidecar = None
        self._m3wm_v57_shadow_last_owner_sidecar_key = None
        self._m3wm_online_adaptation_history = deque(
            maxlen=max(1, int(self.cfg.get("m3wm_online_theta_window", 1)))
        )
        self._m3wm_episode_change_latched = False
        self._m3wm_proposal_supersede_count = 0
        self._m3wm_last_proposal_superseded_from_step = -1
        self._m3wm_last_proposal_superseded_to_step = -1
        self._m3wm_last_proposal_superseded_reason = "none"
        self._m3wm_last_superseded_proposal_receipt = {
            "schema": "m3wm_superseded_proposal_receipt",
            "version": 1,
            "outcome": "NONE",
        }
        self._m3wm_change_skip_episodes_remaining = 0
        self._m3wm_initial_null_episodes_remaining = 0
        self._m3wm_initial_null_steps_remaining = 0
        self._m3wm_episode_is_null_calibration = False
        self._m3wm_episode_is_initial_null_calibration = False
        self._m3wm_episode_null_max_statistic = float("-inf")
        self._m3wm_episode_null_statistics = []
        self._m3wm_initial_null_statistics = []
        self._m3wm_initial_null_dynamics_statistics = []
        self._m3wm_initial_null_reward_statistics = []
        self._m3wm_last_null_reference_statistic = float("nan")
        self._m3wm_adaptive_change_threshold = float(
            self.cfg.get("m3wm_change_shift_threshold", 2.5)
        )
        self._m3wm_change_null_calibration_mode = str(
            self.cfg.get("m3wm_change_null_calibration_mode", "fixed_floor")
        ).lower()
        if self._m3wm_change_null_calibration_mode not in {
            "fixed_floor",
            "empirical_quantile",
        }:
            raise ValueError(
                f"Unknown m3wm_change_null_calibration_mode: {self._m3wm_change_null_calibration_mode!r}"
            )
        self._m3wm_change_null_calibrator = EmpiricalNullCalibrator(
            alpha=float(self.cfg.get("m3wm_change_null_alpha", 0.01)),
            min_samples=int(self.cfg.get("m3wm_change_null_min_samples", 256)),
            max_samples=int(self.cfg.get("m3wm_change_null_max_samples", 4096)),
            fallback_threshold=float(
                self.cfg.get(
                    "m3wm_change_null_fallback_threshold",
                    self.cfg.get("m3wm_change_shift_threshold", 1.8),
                )
            ),
            update_interval=int(self.cfg.get("m3wm_change_null_update_interval", 32)),
            context_local_enabled=True,
            robot_baseline_frozen=True,
        )
        self._m3wm_last_change_null_metadata = {
            "mode": self._m3wm_change_null_calibration_mode,
            "alpha": float(self._m3wm_change_null_calibrator.alpha),
            "effective_threshold": float(
                self._m3wm_change_null_calibrator.fallback_threshold
            ),
            "source": "fallback",
            "ready": False,
            "sample_count": 0,
            "context_sample_count": 0,
            "robot_sample_count": 0,
            "percentile": float("nan"),
            "tail_probability": float("nan"),
            "dynamics_percentile": float("nan"),
            "dynamics_tail_probability": float("nan"),
            "reward_percentile": float("nan"),
            "reward_tail_probability": float("nan"),
            "frozen": False,
            "clean_observation_count": 0,
            "rejected_observation_count": 0,
        }
        self._m3wm_change_null_std_floor_hits = 0
        self._m3wm_change_null_statistic_count = 0
        self._m3wm_raw_theta = None
        self._m3wm_last_router_proposal = None
        self._m3wm_last_memory_theta = None
        self._m3wm_last_router_blend = float("nan")
        self._m3wm_smoothed_theta = None
        self._m3wm_last_theta = None
        self._m3wm_last_context_feature = None
        self._m3wm_context_version = 0
        self._m3wm_runtime_step = 0
        self._m3wm_context_evidence_stream_enabled = False
        self._m3wm_terminal_evidence_contract_enabled = False
        stream_capacity = max(
            512, int(self.cfg.get("m3wm_context_evidence_stream_capacity", 4096))
        )
        self._m3wm_context_evidence_stream = None
        self._m3wm_context_evidence_episode_id = 0
        self._m3wm_context_evidence_generation = 0
        self._m3wm_context_evidence_generation_counter = 0
        self._m3wm_last_terminal_evidence_contract = {
            "valid": False,
            "action": "UNKNOWN_PENDING",
            "reason": "not_evaluated",
        }
        self._m3wm_last_terminal_evidence_diagnostics = {
            "schema": "m3wm_terminal_evidence_diagnostics",
            "version": 1,
            "contract": deepcopy(self._m3wm_last_terminal_evidence_contract),
        }
        self._m3wm_context_evidence_stream_last_error = None
        self._m3wm_context_evidence_trial_pending = False
        self._m3wm_context_evidence_hot_cache_token = None
        self._m3wm_context_evidence_hot_cache = {}
        self._m3wm_context_evidence_chunk_summary_cache = {}
        self._m3wm_context_evidence_chunk_feature_cache = {}
        self._m3wm_online_theta_writeback_not_before_runtime_step = 0
        self._m3wm_online_theta_writeback_release_armed = False
        self._m3wm_online_theta_clean_episode_updates = 0
        self._m3wm_online_theta_transaction_updates = 0
        self._m3wm_online_theta_transaction_proposal_step = None
        self._m3wm_online_theta_terminal_context_hold = False
        self._m3wm_online_theta_last_update_owner = "not_called"
        self._m3wm_online_theta_last_adapt_eligible = False
        self._m3wm_episodic_theta_reward_guard_slots = {}
        self._m3wm_episodic_theta_reward_guard_episode = None
        self._m3wm_episodic_theta_reward_guard_accept_count = 0
        self._m3wm_episodic_theta_reward_guard_reject_count = 0
        self._m3wm_episodic_theta_reward_guard_last_decision = "none"
        self._m3wm_episodic_theta_reward_guard_last_reason = "none"
        self._m3wm_episodic_theta_reward_guard_last_score = float("nan")
        self._m3wm_episodic_theta_baseline_first = {}
        self._m3wm_episodic_theta_baseline_first_last_event = "none"
        self._m3wm_theta_version = -1
        self._m3wm_train_updates = 0
        self._m3wm_schedule_update_offset = 0
        self._m3wm_inherited_specialization_eligible = False
        self._m3wm_current_temperature = float(
            self.cfg.get("m3wm_theta_temperature_start", 1.0)
        )
        self._m3wm_current_base_lr_scale = 1.0
        self._m3wm_current_residual_scale = float(
            self.cfg.get("m3wm_residual_scale", 0.1)
        )
        raw_reward_residual_scale = self.cfg.get("m3wm_reward_residual_scale", None)
        self._m3wm_current_reward_residual_scale = float(
            self._m3wm_current_residual_scale
            if raw_reward_residual_scale is None
            else raw_reward_residual_scale
        )
        self._m3wm_current_value_residual_scale = float(
            self.cfg.get("m3wm_value_residual_scale", self._m3wm_current_residual_scale)
        )
        self._m3wm_specialization_active = False
        self._m3wm_context_memory = None
        self._m3wm_change_detector = None
        self._m3wm_live_bocpd_edge_gate = None
        self._m3wm_active_context_id = None
        self._m3wm_last_change_triggered = False
        self._m3wm_last_change_statistic = float("nan")
        self._m3wm_last_fast_dynamics_shift = float("nan")
        self._m3wm_last_fast_abs_reward_shift = float("nan")
        self._m3wm_last_slow_signed_reward_shift = float("nan")
        self._m3wm_last_existing_log_posterior = float("nan")
        self._m3wm_last_new_log_posterior = float("nan")
        self._m3wm_last_trigger_source = "none"
        self._m3wm_pending_change_tracker = M3wmPendingChangeTracker()
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_proposal_local_objective_enabled must be boolean")
        self._m3wm_objective_enabled = False
        self._m3wm_objective_evaluator = None
        self._m3wm_objective_proposal_step = None
        self._m3wm_objective_proposal_available = False
        self._m3wm_last_objective_decision = None
        self._m3wm_objective_proposal_count = 0
        self._m3wm_objective_confirm_count = 0
        self._m3wm_objective_reject_count = 0
        self._m3wm_objective_unavailable_count = 0
        self._m3wm_identity_quarantine = M3wmOneShotQuarantine()
        self._m3wm_identity_proposal_rollback = None
        configured_memory_feature_mode = str(
            self.cfg.get("m3wm_memory_feature_mode", "full")
        ).lower()
        self._m3wm_reward_identity_enabled = (
            configured_memory_feature_mode == "router_embedding_long_term_reward"
        )
        self._m3wm_reward_identity_residual_index = (
            _context_reward_residual_index(self.model)
            if self._m3wm_reward_identity_enabled
            else None
        )
        self._m3wm_reward_identity_history = deque(maxlen=_M3WM_REWARD_IDENTITY_WINDOW)
        self._m3wm_reward_identity_null_history = []
        self._m3wm_reward_identity_null_median = float("nan")
        self._m3wm_reward_identity_null_mad = float("nan")
        self._m3wm_reward_identity_null_frozen = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_reward_only_quarantine_enabled must be boolean")
        self._m3wm_reward_only_disagreement_enabled = False
        self._m3wm_reward_only_min_abs_signed_score = (
            _validated_reward_only_score_threshold(
                self.cfg.get("m3wm_reward_only_min_abs_signed_score", 2.5),
                "m3wm_reward_only_min_abs_signed_score",
            )
        )
        self._m3wm_reward_only_min_energy_score = (
            _validated_reward_only_score_threshold(
                self.cfg.get("m3wm_reward_only_min_energy_score", 2.5),
                "m3wm_reward_only_min_energy_score",
            )
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_group_recall_conflict_quarantine_enabled must be boolean"
            )
        self._m3wm_group_recall_conflict_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_group_recall_conflict_dual_novelty_retry_enabled must be boolean"
            )
        self._m3wm_group_recall_conflict_dual_novelty_retry_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_group_recall_conflict_dual_ambiguous_retry_enabled must be boolean"
            )
        self._m3wm_group_recall_conflict_dual_ambiguous_retry_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_group_recall_conflict_absolute_novelty_retry_enabled must be boolean"
            )
        self._m3wm_group_recall_conflict_absolute_novelty_retry_enabled = False
        self._m3wm_group_recall_conflict_min_log_bayes = (
            _validated_reward_only_score_threshold(
                self.cfg.get(
                    "m3wm_group_recall_conflict_min_log_bayes", 2.302585092994046
                ),
                "m3wm_group_recall_conflict_min_log_bayes",
            )
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_p2_recall_consensus_enabled must be boolean")
        self._m3wm_p2_recall_consensus_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_unstable_boundary_bridge_enabled must be boolean")
        self._m3wm_unstable_boundary_bridge_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_unstable_boundary_bridge_initial_masked_active_near_energy_quarantine_enabled must be boolean"
            )
        self._m3wm_unstable_boundary_bridge_initial_masked_active_near_energy_enabled = (
            False
        )
        self._m3wm_unstable_boundary_bridge_initial_masked_active_min_energy_ratio = _validated_reward_only_score_threshold(
            self.cfg.get(
                "m3wm_unstable_boundary_bridge_initial_masked_active_min_energy_ratio",
                1.0,
            ),
            "m3wm_unstable_boundary_bridge_initial_masked_active_min_energy_ratio",
        )
        if (
            self._m3wm_unstable_boundary_bridge_initial_masked_active_min_energy_ratio
            > 1.0
        ):
            raise ValueError(
                "m3wm_unstable_boundary_bridge_initial_masked_active_min_energy_ratio must be at most 1.0"
            )
        self._m3wm_unstable_boundary_bridge_initial_masked_active_min_signed_surplus_ratio = _validated_reward_only_score_threshold(
            self.cfg.get(
                "m3wm_unstable_boundary_bridge_initial_masked_active_min_signed_surplus_ratio",
                1.0,
            ),
            "m3wm_unstable_boundary_bridge_initial_masked_active_min_signed_surplus_ratio",
        )
        if (
            self._m3wm_unstable_boundary_bridge_initial_masked_active_min_signed_surplus_ratio
            < 1.0
        ):
            raise ValueError(
                "m3wm_unstable_boundary_bridge_initial_masked_active_min_signed_surplus_ratio must be at least 1.0"
            )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_unstable_boundary_bridge_retry_active_consensus_enabled must be boolean"
            )
        self._m3wm_unstable_boundary_bridge_retry_active_consensus_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_unstable_boundary_bridge_initial_strong_active_consensus_quarantine_enabled must be boolean"
            )
        self._m3wm_unstable_boundary_bridge_initial_strong_active_consensus_enabled = (
            False
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_active_consensus_ambiguous_direct_create_enabled must be boolean"
            )
        self._m3wm_active_consensus_ambiguous_enabled = False
        self._m3wm_active_consensus_ambiguous_max_shadow_log_gap = (
            _validated_reward_only_score_threshold(
                self.cfg.get("m3wm_active_consensus_ambiguous_max_shadow_log_gap", 1.0),
                "m3wm_active_consensus_ambiguous_max_shadow_log_gap",
            )
        )
        self._m3wm_unstable_boundary_bridge_min_abs_signed_score = (
            _validated_reward_only_score_threshold(
                self.cfg.get(
                    "m3wm_unstable_boundary_bridge_min_abs_signed_score", 1.25
                ),
                "m3wm_unstable_boundary_bridge_min_abs_signed_score",
            )
        )
        self._m3wm_unstable_boundary_bridge_min_energy_score = (
            _validated_reward_only_score_threshold(
                self.cfg.get("m3wm_unstable_boundary_bridge_min_energy_score", 1.65),
                "m3wm_unstable_boundary_bridge_min_energy_score",
            )
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_unstable_boundary_bridge_initial_dual_novel_regime_failure_quarantine_enabled must be boolean"
            )
        self._m3wm_unstable_boundary_bridge_initial_dual_novel_regime_failure_enabled = (
            False
        )
        self._m3wm_unstable_boundary_bridge_initial_dual_novel_min_abs_signed_score = _validated_reward_only_score_threshold(
            self.cfg.get(
                "m3wm_unstable_boundary_bridge_initial_dual_novel_min_abs_signed_score",
                0.75,
            ),
            "m3wm_unstable_boundary_bridge_initial_dual_novel_min_abs_signed_score",
        )
        self._m3wm_unstable_boundary_bridge_initial_dual_novel_min_energy_score = (
            _validated_reward_only_score_threshold(
                self.cfg.get(
                    "m3wm_unstable_boundary_bridge_initial_dual_novel_min_energy_score",
                    1.25,
                ),
                "m3wm_unstable_boundary_bridge_initial_dual_novel_min_energy_score",
            )
        )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_p2_rollback_theta_on_terminal_dismiss must be boolean"
            )
        self._m3wm_rollback_theta_on_terminal_dismiss = False
        if not isinstance(True, (bool, np.bool_)):
            raise ValueError("m3wm_dp_confirmed_departure_enabled must be boolean")
        self._m3wm_dp_confirmed_departure_enabled = True
        raw_departure_confidence = self.cfg.get("m3wm_dp_departure_confidence", 0.95)
        if isinstance(raw_departure_confidence, (str, bool, np.bool_)):
            raise ValueError("m3wm_dp_departure_confidence must be in (0.5, 1)")
        self._m3wm_dp_departure_confidence = float(raw_departure_confidence)
        if (
            not np.isfinite(self._m3wm_dp_departure_confidence)
            or not 0.5 < self._m3wm_dp_departure_confidence < 1.0
        ):
            raise ValueError("m3wm_dp_departure_confidence must be in (0.5, 1)")
        raw_departure_dz_threshold = self.cfg.get("m3wm_dp_departure_dz_threshold", 2.0)
        if isinstance(raw_departure_dz_threshold, (str, bool, np.bool_)):
            raise ValueError("m3wm_dp_departure_dz_threshold must be positive")
        self._m3wm_dp_departure_dz_threshold = float(raw_departure_dz_threshold)
        if (
            not np.isfinite(self._m3wm_dp_departure_dz_threshold)
            or self._m3wm_dp_departure_dz_threshold <= 0.0
        ):
            raise ValueError("m3wm_dp_departure_dz_threshold must be positive")
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_dp_strong_dynamics_departure_enabled must be boolean"
            )
        self._m3wm_dp_strong_dynamics_departure_enabled = False
        raw_strong_dynamics_departure_multiplier = self.cfg.get(
            "m3wm_dp_strong_dynamics_departure_multiplier", 1.25
        )
        if isinstance(raw_strong_dynamics_departure_multiplier, (str, bool, np.bool_)):
            raise ValueError(
                "m3wm_dp_strong_dynamics_departure_multiplier must be finite and greater than 1"
            )
        self._m3wm_dp_strong_dynamics_departure_multiplier = float(
            raw_strong_dynamics_departure_multiplier
        )
        if (
            not np.isfinite(self._m3wm_dp_strong_dynamics_departure_multiplier)
            or self._m3wm_dp_strong_dynamics_departure_multiplier <= 1.0
        ):
            raise ValueError(
                "m3wm_dp_strong_dynamics_departure_multiplier must be finite and greater than 1"
            )
        self._m3wm_dp_pending_persistence_required = (
            _validated_pending_persistence_runtime_contract(self.cfg, True)
        )
        if (
            str(self.cfg.get("m3wm_context_assignment_mode", "regime_gate")).lower()
            != "dp_identity"
            or configured_memory_feature_mode != _M3WM_LIVE_MEMORY_FEATURE_MODE
            or str(self.cfg.get("m3wm_change_statistic", "")).lower()
            not in _M3WM_FIXED_LIVE_IDENTITY_MODES
        ):
            raise ValueError(
                "m3wm_dp_confirmed_departure_enabled requires fixed-live change detection, architecture_live_identity DP memory, dp_identity assignment, and regime confirmation"
            )
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_dp_identity_direct_create_require_regime_pass must be boolean"
            )
        self._m3wm_dp_identity_direct_create_require_regime_pass = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError(
                "m3wm_dp_identity_direct_recall_require_regime_pass must be boolean"
            )
        self._m3wm_dp_identity_direct_recall_require_regime_pass = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_p2_fixed_proposal_anchor must be boolean")
        self._m3wm_fixed_proposal_anchor_enabled = False
        raw_multiscale_episode_warmup = self.cfg.get(
            "m3wm_multiscale_episode_warmup_steps", 0
        )
        if (
            isinstance(raw_multiscale_episode_warmup, (bool, np.bool_))
            or not isinstance(raw_multiscale_episode_warmup, (int, np.integer))
            or int(raw_multiscale_episode_warmup) < 0
        ):
            raise ValueError(
                "m3wm_multiscale_episode_warmup_steps must be a non-negative integer"
            )
        self._m3wm_multiscale_episode_warmup_steps = int(raw_multiscale_episode_warmup)
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_multiscale_deferred_warmup_pulse must be boolean")
        self._m3wm_multiscale_deferred_warmup_pulse = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("m3wm_episode_identity_audit_enabled must be boolean")
        self._m3wm_episode_identity_audit_enabled = False
        self._m3wm_identity_audit_pending_evidence = None
        self._m3wm_episode_identity_audit_proposal_count = 0
        self._m3wm_last_identity_audit_step = -1
        self._m3wm_last_identity_audit_score = float("nan")
        self._m3wm_last_identity_audit_reason = "none"
        self._m3wm_reward_only_frozen_theta = _validated_reward_only_frozen_theta(
            self.cfg.get("m3wm_reward_only_frozen_theta", None), self.model
        )
        if self._m3wm_reward_only_frozen_theta is not None:
            raise ValueError(
                "m3wm_reward_only_frozen_theta requires m3wm_reward_only_quarantine_enabled=true"
            )
        if self._m3wm_rollback_theta_on_terminal_dismiss and (
            not self._m3wm_p2_identity_enabled()
        ):
            raise ValueError(
                "m3wm_p2_rollback_theta_on_terminal_dismiss requires router_embedding_long_term_reward memory features and dp_identity assignment"
            )
        if self._m3wm_dp_identity_direct_create_require_regime_pass and (
            not self._m3wm_p2_identity_enabled()
            or int(self.cfg.get("m3wm_regime_confirmation_delay", 160)) != 160
            or (not self._m3wm_rollback_theta_on_terminal_dismiss)
        ):
            raise ValueError(
                "m3wm_dp_identity_direct_create_require_regime_pass requires the P2 identity path, regime confirmation, and terminal-dismiss rollback"
            )
        if self._m3wm_dp_identity_direct_recall_require_regime_pass:
            raise ValueError(
                "m3wm_dp_identity_direct_recall_require_regime_pass requires the P2 identity path, dual-view recall consensus, regime confirmation, and terminal-dismiss rollback"
            )
        if self._m3wm_branchwise_theta_trust_region_enabled:
            raise ValueError(
                "m3wm_branchwise_theta_trust_region_enabled requires the P2 identity path, terminal-dismiss rollback, and a fixed proposal anchor"
            )
        if self._m3wm_reward_identity_enabled:
            if int(self.model.m3wm_context_embedding_dim) != 16:
                raise ValueError(
                    "router_embedding_long_term_reward is the frozen 19-D schema and requires m3wm_context_embedding_dim=16"
                )
            if self._m3wm_reward_identity_residual_index is None:
                raise ValueError(
                    "router_embedding_long_term_reward requires an existing signed reward-residual context coordinate"
                )
            _validated_reward_identity_weight(
                self.cfg.get("m3wm_dp_reward_feature_weight", 0.5)
            )
        self._m3wm_diagnostic_recorder = M3wmDiagnosticRecorder(
            enabled=True, reward_window=128
        )
        self._m3wm_diagnostic_reward_residual_history = deque(maxlen=128)
        self._m3wm_diagnostic_reward_residual_index = (
            _diagnostic_reward_residual_index(self.model)
            if self._m3wm_diagnostic_recorder.enabled
            else None
        )
        self._m3wm_diagnostic_suppressed_pending = False
        self._m3wm_last_confirmed_event = None
        self._m3wm_context_creations = 0
        self._m3wm_context_recalls = 0
        self._m3wm_confirmed_change_count = 0
        self._m3wm_rejected_change_count = 0
        self._m3wm_bocpd_a1_rejected_count = 0
        self._m3wm_bocpd_a2_rejected_count = 0
        self._m3wm_regime_ut_history = deque(maxlen=192)
        self._m3wm_regime_pre_window = None
        self._m3wm_regime_post_window = None
        self._m3wm_regime_gap_remaining = 0
        self._m3wm_regime_event_step = -1
        self._m3wm_regime_rejected_count = 0
        self._m3wm_dp_identity_dismissed = 0
        self._m3wm_dp_identity_created = 0
        self._m3wm_dp_identity_switched = 0
        self._m3wm_last_dp_best_existing_context = -1
        self._m3wm_last_dp_active_log_posterior = float("nan")
        self._m3wm_last_dp_recall_advantage = float("nan")
        self._m3wm_last_dp_recall_emission_advantage = float("nan")
        self._m3wm_last_dp_recall_emission_pass = False
        self._m3wm_last_dp_recall_candidate_context = -1
        self._m3wm_last_dp_recall_candidate_rms = float("nan")
        self._m3wm_last_dp_recall_candidate_rms_threshold = float("nan")
        self._m3wm_last_dp_recall_candidate_support_pass = False
        self._m3wm_last_dp_identity_decision = "none"
        self._m3wm_last_dp_departure_reason = "none"
        self._m3wm_last_dp_departure_active_standardized_rms = float("nan")
        self._m3wm_last_dp_departure_rms_threshold = float("nan")
        self._m3wm_last_dp_departure_regime_d_z = float("nan")
        self._m3wm_last_dp_proposal_fast_dynamics_shift = float("nan")
        self._m3wm_last_dp_proposal_fast_abs_reward_shift = float("nan")
        self._m3wm_last_dp_proposal_slow_signed_reward_shift = float("nan")
        self._m3wm_last_dp_same_version_witness_score = float("nan")
        self._m3wm_last_dp_same_version_witness_threshold = float("nan")
        self._m3wm_last_dp_same_version_witness_dynamics = float("nan")
        self._m3wm_last_dp_same_version_witness_reward = float("nan")
        self._m3wm_last_dp_pending_persistence_score = float("nan")
        self._m3wm_last_dp_pending_persistence_threshold = float("nan")
        self._m3wm_last_dp_pending_persistence_dynamics = float("nan")
        self._m3wm_last_dp_pending_persistence_reward = float("nan")
        self._m3wm_last_source_theta_shadow = self._m3wm_empty_source_theta_shadow(
            enabled=True, reason="not_evaluated"
        )
        self._m3wm_last_dp_physical_witness_valid = False
        self._m3wm_last_dp_physical_witness_pass = False
        self._m3wm_last_dp_physical_witness_score = float("nan")
        self._m3wm_last_dp_physical_witness_threshold = float("nan")
        self._m3wm_last_dp_physical_witness_support = 0
        self._m3wm_last_dp_physical_witness_required = 0
        self._m3wm_last_dp_physical_witness_mutual_score = float("nan")
        self._m3wm_last_dp_physical_witness_mutual_support = 0
        self._m3wm_last_dp_physical_witness_min_rank = 0
        self._m3wm_last_dp_physical_departure_used = False
        self._m3wm_dp_prototype_updates = 0
        self._m3wm_regime_last_dz = float("nan")
        self._m3wm_regime_last_s_post = float("nan")
        self._m3wm_last_context_changed = False
        self._m3wm_pending_trigger_steps = 0
        self._m3wm_active_context_origin = "unassigned"
        self._m3wm_pure_online_runtime = None
        self._m3wm_pure_online_scheduler = None
        self._m3wm_pure_online_last_result = None
        self._m3wm_pure_online_creation_count = 0
        self._m3wm_pure_online_recall_count = 0
        self._m3wm_pure_online_train_step = 0
        self._m3wm_pure_online_last_update_decision = None
        self._m3wm_pure_online_pending_probe_metadata = None
        self._m3wm_three_time_scale_lr_enabled = False
        if self.cfg.get("enable_m3wm", False):
            self._apply_m3wm_training_schedule()

    def _apply_m3wm_training_schedule(self):
        if not self.cfg.get("enable_m3wm", False):
            return
        warmup = max(0, int(self.cfg.get("m3wm_warmup_steps", 5000)))
        base_reduction_steps = max(
            warmup, int(self.cfg.get("m3wm_base_lr_reduction_steps", warmup))
        )
        anneal_steps = max(1, int(self.cfg.get("m3wm_temperature_anneal_steps", 10000)))
        start_temp = float(self.cfg.get("m3wm_theta_temperature_start", 1.0))
        end_temp = float(self.cfg.get("m3wm_theta_temperature_end", 0.5))
        schedule_updates = int(self._m3wm_train_updates) + int(
            getattr(self, "_m3wm_schedule_update_offset", 0)
        )
        was_active = bool(self._m3wm_specialization_active)
        active = (
            bool(getattr(self, "_m3wm_inherited_specialization_eligible", False))
            or schedule_updates >= warmup
        )
        became_active = active and (not was_active)
        if (
            became_active
            and self.model.m3wm_composition_mode == "hard_switching"
            and (warmup > 0)
        ):
            self.model.synchronize_hard_switch_experts()
            print(
                f"[ReCo-WM HardSwitch] copied warm-started expert 0 to all experts at update {self._m3wm_train_updates}"
            )
        if became_active and self.model.m3wm_use_context_anchor:
            self.model.refresh_context_anchor()
            self.reset_context(clear_memory=True, preserve_runtime_step=True)
            print(
                f"[ReCo-WM ContextAnchor] frozen encoder/base snapshot created at update {self._m3wm_train_updates}"
            )
        elif (
            became_active
            and str(self.cfg.get("m3wm_context_runtime_mode", "joint")).lower()
            == "online_after_specialization"
        ):
            self.reset_context(clear_memory=True, preserve_runtime_step=True)
            print(
                f"[ReCo-WM FigureLive] cleared pre-specialization high-level history at update {self._m3wm_train_updates}"
            )
        if active:
            progress = min(1.0, (schedule_updates - warmup) / anneal_steps)
            temperature = start_temp + progress * (end_temp - start_temp)
            base_scale = (
                float(self.cfg.get("m3wm_base_lr_scale_after_warmup", 0.1))
                if self.model.m3wm_composition_mode in {"modular", "hybrid_modular"}
                and schedule_updates >= base_reduction_steps
                else 1.0
            )
        else:
            temperature = start_temp
            base_scale = 1.0
        for group in self.optim.param_groups:
            name = group.get("name", "")
            if name in {"m3wm_dynamics_base", "m3wm_reward_base", "m3wm_value_base"}:
                group["lr"] = self.cfg.lr * base_scale
            elif name in {
                "m3wm_dynamics_primitives",
                "m3wm_reward_primitives",
                "m3wm_value_primitives",
            }:
                group["lr"] = self.cfg.lr if active else 0.0
            elif name == "m3wm_context":
                group["lr"] = self.cfg.lr if active else 0.0
        residual_end = float(self.cfg.get("m3wm_residual_scale", 0.1))
        raw_reward_residual_end = self.cfg.get("m3wm_reward_residual_scale", None)
        reward_residual_end = float(
            residual_end if raw_reward_residual_end is None else raw_reward_residual_end
        )
        value_residual_end = float(
            self.cfg.get("m3wm_value_residual_scale", residual_end)
        )
        residual_start = float(self.cfg.get("m3wm_residual_scale_start", residual_end))
        raw_reward_residual_start = self.cfg.get(
            "m3wm_reward_residual_scale_start", None
        )
        reward_residual_start = float(
            residual_start
            if raw_reward_residual_start is None
            else raw_reward_residual_start
        )
        value_residual_start = float(
            self.cfg.get("m3wm_value_residual_scale_start", residual_start)
        )
        residual_ramp_steps = max(
            0, int(self.cfg.get("m3wm_residual_scale_ramp_steps", 0))
        )
        if (
            self.model.m3wm_composition_mode == "hybrid_modular"
            and residual_ramp_steps > 0
        ):
            residual_progress = min(
                1.0, max(0, schedule_updates - warmup) / residual_ramp_steps
            )
            current_residual_scale = residual_start + residual_progress * (
                residual_end - residual_start
            )
            current_reward_residual_scale = (
                reward_residual_start
                + residual_progress * (reward_residual_end - reward_residual_start)
            )
            current_value_residual_scale = value_residual_start + residual_progress * (
                value_residual_end - value_residual_start
            )
        else:
            current_residual_scale = residual_end
            current_reward_residual_scale = reward_residual_end
            current_value_residual_scale = value_residual_end
        self.model.set_modular_residual_scale(
            current_residual_scale,
            current_value_residual_scale,
            reward_scale=current_reward_residual_scale,
        )
        self._m3wm_current_temperature = float(temperature)
        self._m3wm_current_base_lr_scale = float(base_scale)
        self._m3wm_current_residual_scale = float(current_residual_scale)
        self._m3wm_current_reward_residual_scale = float(current_reward_residual_scale)
        self._m3wm_current_value_residual_scale = float(current_value_residual_scale)
        self._m3wm_specialization_active = bool(active)
        self.model.set_context_temperature(temperature)

    def reset_context_activation_transients(self):
        self._prev_mean.zero_()
        self._guarded_control_frame = None
        for name in (
            "_m3wm_online_adaptation_history",
            "_m3wm_context_history",
            "_m3wm_context_raw_history",
            "_m3wm_change_history",
            "_m3wm_local_system_id_history",
            "_m3wm_detector_raw_history",
        ):
            value = getattr(self, name, None)
            if hasattr(value, "clear"):
                value.clear()
        self._m3wm_last_context_feature = None

    def _m3wm_diagnostics_enabled(self):
        recorder = getattr(self, "_m3wm_diagnostic_recorder", None)
        return bool(recorder is not None and recorder.enabled)

    def _m3wm_diag_increment_guard(self, name, amount=1):
        if self._m3wm_diagnostics_enabled():
            self._m3wm_diagnostic_recorder.increment_guard(name, amount)

    def _m3wm_diag_reward_summary(self):
        return signed_reward_residual_summary(
            getattr(self, "_m3wm_diagnostic_reward_residual_history", ()), window=128
        )

    def _m3wm_diag_observe_context_feature(self, feature):
        if not self._m3wm_diagnostics_enabled():
            return
        reward_index = getattr(self, "_m3wm_diagnostic_reward_residual_index", None)
        value = None
        if reward_index is not None and int(reward_index) < feature.numel():
            value = float(feature[reward_index].detach().cpu())
        self._m3wm_diagnostic_reward_residual_history.append(value)

    def _m3wm_diag_slot_digests(self):
        if not self._m3wm_diagnostics_enabled():
            return None
        return canonical_context_slot_digests(None)

    def _m3wm_diag_theta_snapshot(
        self, pre_adapt_provisional=None, pre_adapt_control=None, post_adapt_final=None
    ):

        def branch_view(theta):
            if theta is None:
                return None
            branches = int(getattr(self.model, "m3wm_num_composition_branches", 1))
            primitives = int(
                getattr(
                    self.model, "m3wm_num_primitives", torch.as_tensor(theta).numel()
                )
            )
            if (
                bool(getattr(self.model, "m3wm_separate_composition_heads", False))
                and torch.as_tensor(theta).numel() == branches * primitives
            ):
                return torch.as_tensor(theta).reshape(branches, primitives)
            return theta

        return {
            "router_proposal": tensor_snapshot(
                branch_view(self._m3wm_last_router_proposal)
            ),
            "memory_blend_input": tensor_snapshot(
                branch_view(self._m3wm_last_memory_theta)
            ),
            "pre_adapt_provisional": tensor_snapshot(
                branch_view(
                    self._m3wm_raw_theta
                    if pre_adapt_provisional is None
                    else pre_adapt_provisional
                )
            ),
            "pre_adapt_control": tensor_snapshot(
                branch_view(
                    self._m3wm_last_theta
                    if pre_adapt_control is None
                    else pre_adapt_control
                )
            ),
            "post_adapt_final_control": tensor_snapshot(
                branch_view(
                    self._m3wm_last_theta
                    if post_adapt_final is None
                    else post_adapt_final
                )
            ),
            "active_slot_theta": tensor_snapshot(branch_view(None)),
        }

    def _m3wm_identity_rollback_digest_state(self, snapshot):
        cloned = self._m3wm_clone_identity_proposal_rollback(snapshot)

        def branch_view(theta):
            if theta is None:
                return None
            branches = int(getattr(self.model, "m3wm_num_composition_branches", 1))
            primitives = int(
                getattr(
                    self.model, "m3wm_num_primitives", torch.as_tensor(theta).numel()
                )
            )
            if (
                bool(getattr(self.model, "m3wm_separate_composition_heads", False))
                and torch.as_tensor(theta).numel() == branches * primitives
            ):
                return torch.as_tensor(theta).reshape(branches, primitives)
            return theta

        history_samples = [
            {
                "step": int(sample[0]),
                "obs": tensor_snapshot(sample[1]),
                "action": tensor_snapshot(sample[2]),
                "reward": tensor_snapshot(sample[3]),
                "next_obs": tensor_snapshot(sample[4]),
                "task": None if sample[5] is None else tensor_snapshot(sample[5]),
            }
            for sample in cloned["online_adaptation_history"]
        ]
        tensor_fields = ("obs", "action", "reward", "next_obs")
        if any(
            (
                not sample[name].get("valid", False)
                for sample in history_samples
                for name in tensor_fields
            )
        ) or any(
            (
                sample["task"] is not None and (not sample["task"].get("valid", False))
                for sample in history_samples
            )
        ):
            raise ValueError("Invalid rollback adaptation-history snapshot")
        history_carrier = {
            "schema": "m3wm_rollback_adaptation_history",
            "version": 2,
            "samples": history_samples,
        }
        state = {
            "schema": "m3wm_terminal_dismiss_theta_rollback_state",
            "version": 2,
            "proposal_step": int(cloned["proposal_step"]),
            "active_context_id": cloned["active_context_id"],
            "active_context_origin": cloned["active_context_origin"],
            "theta": {
                name: tensor_snapshot(branch_view(cloned[name]))
                for name in (
                    "raw_theta",
                    "smoothed_theta",
                    "last_theta",
                    "last_memory_theta",
                    "last_router_proposal",
                )
            },
            "runtime_theta": tensor_snapshot(branch_view(cloned["last_theta"])),
            "last_router_blend": cloned["last_router_blend"],
            "theta_version": int(cloned["theta_version"]),
            "online_adaptation_history": {
                "length": len(cloned["online_adaptation_history"]),
                "sha256": canonical_json_sha256(history_carrier),
            },
            "online_theta_transaction_state": deepcopy(
                cloned.get(
                    "online_theta_transaction_state",
                    self._m3wm_validated_online_theta_transaction_state(None),
                )
            ),
            "online_theta_clean_episode_updates": int(
                cloned.get("online_theta_clean_episode_updates", 0)
            ),
        }
        if "proposal_context_state" in cloned:
            state["proposal_context_state"] = deepcopy(cloned["proposal_context_state"])
        state["control_state_sha256"] = canonical_json_sha256(
            {
                key: value
                for (key, value) in state.items()
                if key not in {"schema", "version", "theta_version"}
            }
        )
        state["state_sha256"] = canonical_json_sha256(state)
        return state

    def _m3wm_diag_identity_rollback_state(self, snapshot):
        return self._m3wm_identity_rollback_digest_state(snapshot)

    def _m3wm_diag_current_identity_rollback_state(self, proposal_step):
        snapshot = {
            "proposal_step": int(proposal_step),
            "active_context_id": None,
            "active_context_origin": str(self._m3wm_active_context_origin),
            "raw_theta": self._clone_optional_tensor(self._m3wm_raw_theta),
            "smoothed_theta": self._clone_optional_tensor(self._m3wm_smoothed_theta),
            "last_theta": self._clone_optional_tensor(self._m3wm_last_theta),
            "last_memory_theta": self._clone_optional_tensor(
                self._m3wm_last_memory_theta
            ),
            "last_router_proposal": self._clone_optional_tensor(
                self._m3wm_last_router_proposal
            ),
            "last_router_blend": finite_or_none(self._m3wm_last_router_blend),
            "theta_version": int(self._m3wm_theta_version),
            "online_adaptation_history": self._m3wm_clone_adaptation_history(
                self._m3wm_online_adaptation_history
            ),
            "online_adaptation_history_schema": {
                "schema": "m3wm_raw_transition_history",
                "version": 1,
            },
            "online_theta_transaction_state": self._m3wm_online_theta_transaction_state_dict(),
            "online_theta_clean_episode_updates": self._m3wm_online_theta_clean_episode_update_count_value(),
        }
        return self._m3wm_diag_identity_rollback_state(snapshot)

    def _m3wm_diag_finalize_transaction(
        self,
        outcome,
        reason,
        pre_adapt_provisional,
        pre_adapt_control,
        terminal_dismiss_rollback_requested=False,
        terminal_dismiss_rollback_applied=False,
        terminal_dismiss_rollback_proposal_step=None,
    ):
        if not self._m3wm_diagnostics_enabled():
            return
        if getattr(self, "_m3wm_diagnostic_suppressed_pending", False):
            return
        requested = bool(terminal_dismiss_rollback_requested)
        applied = bool(terminal_dismiss_rollback_applied)
        if applied and (not requested):
            raise RuntimeError(
                "Terminal-dismiss rollback cannot be applied unless requested"
            )
        final = {
            **runtime_step_coordinates(self._m3wm_runtime_step),
            "active_context_id_after": None,
            "context_count_after": 0,
            "slot_digests_after": self._m3wm_diag_slot_digests(),
            "theta": self._m3wm_diag_theta_snapshot(
                pre_adapt_provisional=pre_adapt_provisional,
                pre_adapt_control=pre_adapt_control,
                post_adapt_final=self._m3wm_last_theta,
            ),
            "reward_residual": self._m3wm_diag_reward_summary(),
        }
        if getattr(self, "_m3wm_rollback_theta_on_terminal_dismiss", False):
            if requested and terminal_dismiss_rollback_proposal_step is None:
                raise RuntimeError(
                    "Terminal-dismiss rollback request has no proposal binding"
                )
            current_state = (
                self._m3wm_diag_current_identity_rollback_state(
                    terminal_dismiss_rollback_proposal_step
                )
                if applied
                else None
            )
            runtime_theta = getattr(
                self.model,
                "_m3wm_runtime_theta",
                getattr(self.model, "runtime_theta", None),
            )
            final["terminal_dismiss_theta_rollback"] = {
                "enabled": True,
                "requested": requested,
                "applied": applied,
                "proposal_step": (
                    None
                    if terminal_dismiss_rollback_proposal_step is None
                    else int(terminal_dismiss_rollback_proposal_step)
                ),
                "restored_state": current_state,
                "actual_runtime_theta": tensor_snapshot(runtime_theta),
            }
        self._m3wm_diagnostic_recorder.finalize(str(outcome), str(reason), final)

    def _m3wm_diag_orphan_transaction(self, reason):
        if (
            not self._m3wm_diagnostics_enabled()
            or not self._m3wm_diagnostic_recorder.active
        ):
            return
        identity = self._m3wm_diagnostic_recorder.active_identity()
        final_runtime_step = max(
            1, int(self._m3wm_runtime_step), int(identity["proposal_step"])
        )
        self._m3wm_diagnostic_recorder.orphan(
            final_runtime_step,
            str(reason),
            {
                "active_context_id_after": None,
                "context_count_after": 0,
                "slot_digests_after": self._m3wm_diag_slot_digests(),
                "theta": self._m3wm_diag_theta_snapshot(),
                "reward_residual": self._m3wm_diag_reward_summary(),
            },
        )

    def _m3wm_context_runtime_enabled(self):
        mode = str(self.cfg.get("m3wm_context_runtime_mode", "joint")).lower()
        if mode == "joint":
            return True
        if mode == "deployment_only":
            return not self.training
        if mode == "online_after_specialization":
            live_semantic = (
                str(self.cfg.get("m3wm_change_statistic", "")).lower()
                == "bocpd_live_three_stream"
            )
            anchor_required = False
            anchor_ready = bool(getattr(self.model, "context_anchor_ready", False))
            return bool(self._m3wm_specialization_active)
        raise ValueError(
            f"m3wm_context_runtime_mode must be 'joint', 'deployment_only', or 'online_after_specialization', received {mode!r}"
        )

    def _m3wm_close_multiscale_terminal_decision(self, *, accepted, confirmation_delay):
        return False

    def _m3wm_has_pending_change(self):
        return bool(self._m3wm_pending_change_tracker.active)

    def _m3wm_observe_episodic_theta_baseline_first(self, reward):
        return False

    def _m3wm_reset_episodic_theta_baseline_first_episode(self):
        entries = getattr(self, "_m3wm_episodic_theta_baseline_first", {})
        for entry in entries.values():
            if isinstance(entry, dict):
                entry["partial_rewards"] = []
        return False

    def _m3wm_reset_episodic_theta_guard_episode(self):
        self._m3wm_episodic_theta_reward_guard_episode = {
            "context_id": None,
            "sample_count": 0,
            "tail_rewards": [],
            "candidate_theta": None,
            "eligible": True,
            "ineligible_reason": "none",
            "resolved": False,
            "resolution": "none",
        }

    def _m3wm_invalidate_episodic_theta_guard_episode(self, reason):
        episode = getattr(self, "_m3wm_episodic_theta_reward_guard_episode", None)
        if (
            not isinstance(episode, dict)
            or episode.get("resolved", False)
            or (not episode.get("eligible", False))
        ):
            return
        episode["eligible"] = False
        episode["ineligible_reason"] = str(reason)

    def _m3wm_install_episodic_theta_guard_value(self, context_id, theta):
        return False

    def _m3wm_episodic_theta_champion_score_is_current(self, accepted):
        if not isinstance(accepted, dict):
            return False
        try:
            score_updates = int(accepted.get("score_train_updates", -1))
            current_updates = int(getattr(self, "_m3wm_train_updates", 0))
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(score_updates >= 0 and current_updates == score_updates)

    def _m3wm_restore_episodic_theta_best_for_clean_episode(self):
        True
        return False

    def _m3wm_finalize_episodic_theta_reward_guard(self):
        episode = getattr(self, "_m3wm_episodic_theta_reward_guard_episode", None)
        if not isinstance(episode, dict) or episode.get("sample_count", 0) == 0:
            return "empty"
        if bool(episode.get("resolved", False)):
            return str(episode.get("resolution", "ineligible"))
        reason = str(episode.get("ineligible_reason", "invalid_episode"))
        context_id = episode.get("context_id")
        tail_rewards = episode.get("tail_rewards", [])
        candidate = episode.get("candidate_theta")
        block_size = _M3WM_EPISODIC_THETA_REWARD_BLOCK
        required_block_count = (
            self._m3wm_episodic_theta_reward_guard_min_tail_samples // block_size
        )
        complete_block_count = len(tail_rewards) // block_size
        eligible = bool(
            episode.get("eligible", False)
            and context_id is not None
            and (not self._m3wm_has_pending_change())
            and (complete_block_count >= required_block_count)
            and torch.is_tensor(candidate)
        )
        if not eligible:
            False
            self._m3wm_episodic_theta_reward_guard_last_decision = "ineligible"
            self._m3wm_episodic_theta_reward_guard_last_reason = (
                reason if not episode.get("eligible", False) else "short_tail"
            )
            episode["resolved"] = True
            episode["resolution"] = "ineligible"
            return "ineligible"
        block_values = (
            np.asarray(
                tail_rewards[: complete_block_count * block_size], dtype=np.float64
            )
            .reshape(complete_block_count, block_size)
            .mean(axis=1)
        )
        score = float(np.median(block_values))
        if not np.isfinite(score):
            self._m3wm_episodic_theta_reward_guard_last_decision = "ineligible"
            self._m3wm_episodic_theta_reward_guard_last_reason = "nonfinite_score"
            episode["resolved"] = True
            episode["resolution"] = "ineligible"
            return "ineligible"
        context_id = int(context_id)
        slots = self._m3wm_episodic_theta_reward_guard_slots
        accepted = slots.get(context_id)
        score_coordinate_current = bool(
            accepted is None
            or self._m3wm_episodic_theta_champion_score_is_current(accepted)
        )
        improved = bool(
            accepted is None
            or not score_coordinate_current
            or score
            >= float(accepted["best_score"])
            + self._m3wm_episodic_theta_reward_guard_improvement_margin
        )
        if improved:
            bootstrap_score = (
                score
                if accepted is None or not score_coordinate_current
                else float(accepted["bootstrap_score"])
            )
            new_accepted = {
                "best_theta": candidate.detach().clone().cpu(),
                "best_score": score,
                "bootstrap_score": bootstrap_score,
                "score_train_updates": int(getattr(self, "_m3wm_train_updates", 0)),
                "score_context_version": int(self._m3wm_context_version),
                "accepted_episodes": (
                    1 if accepted is None else int(accepted["accepted_episodes"]) + 1
                ),
                "locked": False,
            }
            if not self._m3wm_install_episodic_theta_guard_value(context_id, candidate):
                self._m3wm_episodic_theta_reward_guard_last_decision = "ineligible"
                self._m3wm_episodic_theta_reward_guard_last_reason = "invalid_context"
                episode["resolved"] = True
                episode["resolution"] = "ineligible"
                return "ineligible"
            slots[context_id] = new_accepted
            self._m3wm_episodic_theta_reward_guard_accept_count += 1
            decision = (
                "bootstrap"
                if accepted is None
                else "stale_rebootstrap" if not score_coordinate_current else "accept"
            )
        else:
            self._m3wm_install_episodic_theta_guard_value(
                context_id, accepted["best_theta"]
            )
            self._m3wm_episodic_theta_reward_guard_reject_count += 1
            decision = "reject_restore"
        self._m3wm_episodic_theta_reward_guard_last_decision = decision
        self._m3wm_episodic_theta_reward_guard_last_reason = "real_reward_tail"
        self._m3wm_episodic_theta_reward_guard_last_score = score
        episode["resolved"] = True
        episode["resolution"] = decision
        return decision

    def _m3wm_observe_episodic_theta_reward_guard(self, reward):
        episode = self._m3wm_episodic_theta_reward_guard_episode
        if not isinstance(episode, dict):
            self._m3wm_reset_episodic_theta_guard_episode()
            episode = self._m3wm_episodic_theta_reward_guard_episode
        episode["sample_count"] += 1
        if bool(episode.get("resolved", False)):
            return
        self._m3wm_invalidate_episodic_theta_guard_episode("unassigned_context")
        if self._m3wm_episode_is_initial_null_calibration:
            self._m3wm_invalidate_episodic_theta_guard_episode("initial_null")
        if self._m3wm_last_context_changed:
            self._m3wm_invalidate_episodic_theta_guard_episode("mixed_context")
        transient_guard = bool(
            self._m3wm_has_pending_change()
            or self._m3wm_online_theta_last_update_owner == "transaction_gradient"
        )
        post_terminal_physical_guard = bool(
            self._m3wm_post_terminal_writeback_guard_active()
        )
        if transient_guard or post_terminal_physical_guard:
            episode["ineligible_reason"] = (
                "transaction_suffix_restart"
                if transient_guard
                else "post_terminal_physical_state_guard"
            )
            episode["tail_rewards"] = []
            episode["candidate_theta"] = None
        stable_tail_ready = bool(
            self._m3wm_online_theta_last_update_owner == "clean_budget_hold"
            and self._m3wm_online_theta_clean_episode_budget_exhausted()
        )
        if (
            episode.get("eligible", False)
            and (not transient_guard)
            and (not post_terminal_physical_guard)
            and stable_tail_ready
            and (self._m3wm_last_theta is not None)
        ):
            value = float(torch.as_tensor(reward).detach().reshape(-1)[0].cpu())
            if not np.isfinite(value):
                self._m3wm_invalidate_episodic_theta_guard_episode("nonfinite_reward")
                return
            episode["tail_rewards"].append(value)
            episode["candidate_theta"] = self._m3wm_last_theta.detach().clone()
            if len(episode["tail_rewards"]) >= int(
                self._m3wm_episodic_theta_reward_guard_min_tail_samples
            ):
                self._m3wm_finalize_episodic_theta_reward_guard()

    def _m3wm_post_terminal_writeback_clean_steps_value(self):
        value = getattr(
            self,
            "_m3wm_post_terminal_writeback_clean_steps",
            self.cfg.get("m3wm_post_terminal_writeback_clean_steps", 0),
        )
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(
                "m3wm_post_terminal_writeback_clean_steps must be a non-negative integer"
            )
        return int(value)

    def _m3wm_maybe_arm_post_terminal_writeback_guard(
        self, transaction_owned_this_step
    ):
        if not bool(transaction_owned_this_step) or self._m3wm_has_pending_change():
            return False
        clean_steps = self._m3wm_post_terminal_writeback_clean_steps_value()
        self._m3wm_online_theta_writeback_not_before_runtime_step = (
            int(self._m3wm_runtime_step) + clean_steps
        )
        self._m3wm_online_theta_writeback_release_armed = True
        return True

    def _m3wm_post_terminal_writeback_guard_active(self):
        deadline = getattr(
            self, "_m3wm_online_theta_writeback_not_before_runtime_step", 0
        )
        if (
            isinstance(deadline, (bool, np.bool_))
            or not isinstance(deadline, (int, np.integer))
            or int(deadline) < 0
        ):
            return True
        return int(self._m3wm_runtime_step) < int(deadline)

    def _m3wm_post_terminal_writeback_release_is_armed(self):
        value = getattr(self, "_m3wm_online_theta_writeback_release_armed", False)
        return bool(value) if isinstance(value, (bool, np.bool_)) else False

    def _m3wm_post_terminal_latch_release_active(self):
        return bool(
            self._m3wm_post_terminal_writeback_clean_steps_value() > 0
            and self._m3wm_post_terminal_writeback_release_is_armed()
            and (not self._m3wm_post_terminal_writeback_guard_active())
        )

    def _m3wm_online_theta_clean_episode_update_limit_value(self):
        value = getattr(
            self,
            "_m3wm_online_theta_clean_episode_update_limit",
            self.cfg.get("m3wm_online_theta_clean_episode_update_limit", 0),
        )
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(
                "m3wm_online_theta_clean_episode_update_limit must be a non-negative integer"
            )
        return int(value)

    def _m3wm_online_theta_clean_episode_update_count_value(self):
        limit = self._m3wm_online_theta_clean_episode_update_limit_value()
        value = getattr(self, "_m3wm_online_theta_clean_episode_updates", 0)
        valid = bool(
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer))
            and (int(value) >= 0)
            and (limit == 0 and int(value) == 0 or (limit > 0 and int(value) <= limit))
        )
        if not valid:
            raise ValueError("Invalid m3wm online-theta clean episode update count")
        return int(value)

    def _m3wm_online_theta_clean_episode_budget_exhausted(self):
        limit = self._m3wm_online_theta_clean_episode_update_limit_value()
        return bool(
            limit > 0
            and self._m3wm_online_theta_clean_episode_update_count_value() >= limit
        )

    def _m3wm_online_theta_transaction_update_limit_value(self):
        value = getattr(
            self,
            "_m3wm_online_theta_transaction_update_limit",
            self.cfg.get("m3wm_online_theta_transaction_update_limit", 0),
        )
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(
                "m3wm_online_theta_transaction_update_limit must be a non-negative integer"
            )
        return int(value)

    def _m3wm_online_theta_transaction_update_count_value(self):
        limit = self._m3wm_online_theta_transaction_update_limit_value()
        value = getattr(self, "_m3wm_online_theta_transaction_updates", 0)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            or (limit == 0 and int(value) != 0)
            or (limit > 0 and int(value) > limit)
        ):
            raise ValueError("Invalid ReCo-WM transaction theta-update count")
        return int(value)

    def _m3wm_online_theta_transaction_budget_exhausted(self):
        limit = self._m3wm_online_theta_transaction_update_limit_value()
        return bool(
            limit > 0
            and self._m3wm_online_theta_transaction_update_count_value() >= limit
        )

    def _m3wm_online_theta_transaction_state_dict(self):
        limit = self._m3wm_online_theta_transaction_update_limit_value()
        updates = self._m3wm_online_theta_transaction_update_count_value()
        proposal_step = getattr(
            self, "_m3wm_online_theta_transaction_proposal_step", None
        )
        if proposal_step is not None and (
            isinstance(proposal_step, (bool, np.bool_))
            or not isinstance(proposal_step, (int, np.integer))
            or int(proposal_step) < 1
        ):
            raise ValueError("Invalid ReCo-WM transaction theta proposal binding")
        terminal_hold = getattr(self, "_m3wm_online_theta_terminal_context_hold", False)
        if not isinstance(terminal_hold, (bool, np.bool_)):
            raise ValueError("Invalid ReCo-WM terminal Context theta hold")
        if (
            limit == 0
            and (updates != 0 or proposal_step is not None or terminal_hold)
            or (terminal_hold and proposal_step is None)
        ):
            raise ValueError("Incoherent ReCo-WM transaction theta state")
        return {
            "schema": "m3wm_online_theta_transaction_budget",
            "version": 1,
            "configured_limit": limit,
            "updates": updates,
            "proposal_step": None if proposal_step is None else int(proposal_step),
            "terminal_context_hold": bool(terminal_hold),
        }

    def _m3wm_validated_online_theta_transaction_state(self, state):
        limit = self._m3wm_online_theta_transaction_update_limit_value()
        if state is None:
            return {
                "schema": "m3wm_online_theta_transaction_budget",
                "version": 1,
                "configured_limit": limit,
                "updates": 0,
                "proposal_step": None,
                "terminal_context_hold": False,
            }
        required = {
            "schema",
            "version",
            "configured_limit",
            "updates",
            "proposal_step",
            "terminal_context_hold",
        }
        if (
            not isinstance(state, dict)
            or set(state) != required
            or state.get("schema") != "m3wm_online_theta_transaction_budget"
            or (state.get("version") != 1)
        ):
            raise ValueError("Invalid ReCo-WM transaction theta-state schema")
        raw_limit = state["configured_limit"]
        raw_updates = state["updates"]
        proposal_step = state["proposal_step"]
        terminal_hold = state["terminal_context_hold"]
        if (
            isinstance(raw_limit, (bool, np.bool_))
            or not isinstance(raw_limit, (int, np.integer))
            or int(raw_limit) != limit
            or isinstance(raw_updates, (bool, np.bool_))
            or (not isinstance(raw_updates, (int, np.integer)))
            or (int(raw_updates) < 0)
            or (limit == 0 and int(raw_updates) != 0)
            or (limit > 0 and int(raw_updates) > limit)
            or (not isinstance(terminal_hold, (bool, np.bool_)))
        ):
            raise ValueError(
                "Invalid or configuration-mismatched ReCo-WM transaction theta state"
            )
        if proposal_step is not None and (
            isinstance(proposal_step, (bool, np.bool_))
            or not isinstance(proposal_step, (int, np.integer))
            or int(proposal_step) < 1
        ):
            raise ValueError("Invalid ReCo-WM transaction theta proposal step")
        if (
            limit == 0
            and (proposal_step is not None or bool(terminal_hold))
            or (bool(terminal_hold) and proposal_step is None)
        ):
            raise ValueError("Incoherent ReCo-WM transaction theta checkpoint")
        return {
            "schema": "m3wm_online_theta_transaction_budget",
            "version": 1,
            "configured_limit": limit,
            "updates": int(raw_updates),
            "proposal_step": None if proposal_step is None else int(proposal_step),
            "terminal_context_hold": bool(terminal_hold),
        }

    def _m3wm_install_online_theta_transaction_state(self, state):
        validated = self._m3wm_validated_online_theta_transaction_state(state)
        self._m3wm_online_theta_transaction_updates = validated["updates"]
        self._m3wm_online_theta_transaction_proposal_step = validated["proposal_step"]
        self._m3wm_online_theta_terminal_context_hold = validated[
            "terminal_context_hold"
        ]
        return validated

    def _m3wm_end_online_theta_transaction(self, *, context_committed):
        self._m3wm_online_theta_transaction_updates = 0
        self._m3wm_online_theta_transaction_proposal_step = None
        self._m3wm_online_theta_terminal_context_hold = False

    def _m3wm_pending_steps_remaining(self):
        tracker_remaining = 0
        pending = self._m3wm_pending_change_tracker.get()
        if self._m3wm_pending_change_tracker.active and pending is not None:
            tracker_remaining = max(
                0, int(pending["confirmation_due_step"]) - int(self._m3wm_runtime_step)
            )
        return max(0, tracker_remaining)

    def _m3wm_pending_theta_inheritance_enabled(self):
        value = getattr(self, "_m3wm_new_context_inherit_pending_runtime_theta", False)
        return bool(value) if isinstance(value, (bool, np.bool_)) else False

    def _m3wm_capture_pending_runtime_theta(self, runtime_step):
        if not self._m3wm_pending_theta_inheritance_enabled():
            return False
        theta = getattr(self, "_m3wm_last_theta", None)
        window = getattr(
            self,
            "_m3wm_new_context_inherit_theta_window",
            self.cfg.get("m3wm_new_context_inherit_theta_window", 128),
        )
        if not torch.is_tensor(theta):
            return False
        return self._m3wm_pending_change_tracker.append_runtime_theta(
            runtime_step, theta, window
        )

    def _m3wm_close_pending_transaction(
        self, consume_confirmed=False, preserve_pending=False
    ):
        had_transaction = self._m3wm_pending_change_tracker.get() is not None
        if consume_confirmed:
            event = self._m3wm_pending_change_tracker.consume_confirmed()
        elif preserve_pending:
            event = None
        else:
            self._m3wm_pending_change_tracker.cancel_pending()
            event = None
        self._m3wm_regime_pre_window = None
        self._m3wm_regime_post_window = None
        self._m3wm_regime_gap_remaining = 0
        self._m3wm_regime_event_step = -1
        self._m3wm_bocpd_window_retriggers = 0
        self._m3wm_identity_audit_pending_evidence = None
        quarantine = getattr(self, "_m3wm_identity_quarantine", None)
        if quarantine is not None:
            quarantine.reset()
        self._m3wm_discard_identity_proposal_rollback()
        if (
            had_transaction
            and str(self.cfg.get("m3wm_change_statistic", "")).lower()
            == "bocpd_live_three_stream"
        ):
            self._m3wm_live_bocpd_edge_gate_for_mode().close_transaction()
        self._m3wm_objective_abort_if_incomplete()
        return event

    def _m3wm_objective_abort_if_incomplete(self):
        evaluator = getattr(self, "_m3wm_objective_evaluator", None)
        owner = getattr(self, "_m3wm_objective_proposal_step", None)
        if evaluator is None or owner is None:
            return False
        if evaluator.phase == "complete":
            return False
        if evaluator.phase == "proposal":
            evaluator.cancel_proposal()
        elif evaluator.phase != "building_null":
            raise RuntimeError("Unknown proposal-local objective phase")
        self._m3wm_objective_proposal_step = None
        self._m3wm_objective_proposal_available = False
        self._m3wm_last_objective_decision = None
        self._m3wm_objective_reject_count += 1
        return True

    def _m3wm_objective_finalize(self, *, context_committed):
        evaluator = getattr(self, "_m3wm_objective_evaluator", None)
        owner = getattr(self, "_m3wm_objective_proposal_step", None)
        if evaluator is None or owner is None:
            return False
        if evaluator.phase == "building_null":
            self._m3wm_objective_proposal_step = None
            self._m3wm_objective_proposal_available = False
            self._m3wm_objective_reject_count += 1
            return True
        if evaluator.phase != "complete" or evaluator.decision is None:
            raise RuntimeError(
                "Terminal Context outcome has incomplete objective evidence"
            )
        adopt = bool(context_committed and evaluator.decision.changed)
        evaluator.resolve_proposal(
            adopt_alternative=adopt, reset_stable_statistics=bool(context_committed)
        )
        if adopt:
            self._m3wm_objective_confirm_count += 1
        else:
            self._m3wm_objective_reject_count += 1
        self._m3wm_objective_proposal_step = None
        self._m3wm_objective_proposal_available = False
        return True

    def _m3wm_abort_pending_at_episode_reset(self):
        pending = self._m3wm_pending_change_tracker.get()
        if pending is None:
            return False
        if pending is not None:
            rollback_plan = self._m3wm_prepare_terminal_dismiss_theta_rollback(
                pending, decision_requires_rollback=True
            )
            if rollback_plan is None:
                raise RuntimeError("Episode-reset pending abort has no rollback plan")
            proposal_step, rollback_snapshot = rollback_plan
            if not self._m3wm_restore_identity_proposal_rollback(
                proposal_step, rollback_snapshot
            ):
                raise RuntimeError(
                    "Episode-reset pending abort could not restore its proposal-edge control state"
                )
        self._m3wm_pending_trigger_steps = 0
        self._m3wm_close_pending_transaction()
        self._m3wm_end_online_theta_transaction(context_committed=False)
        self._m3wm_rejected_change_count += 1
        self._m3wm_regime_rejected_count += 1
        self._m3wm_diag_orphan_transaction("episode_reset_pending_abort")
        self._m3wm_episode_change_latched = True
        return True

    def _m3wm_detector_raw_history_maxlen(self):
        long_window = max(
            int(self.cfg.get("m3wm_change_long_window", 32)),
            (
                2 * int(getattr(self, "_m3wm_signed_reward_slow_window", 32))
                if bool(getattr(self, "_m3wm_signed_reward_slow_enabled", False))
                else 1
            ),
        )
        base_capacity = max(
            1,
            int(self.cfg.get("m3wm_regime_confirmation_delay", 160)) + long_window + 1,
        )
        return max(
            base_capacity,
            (
                2 * 192
                if bool(getattr(self, "_m3wm_v65_fast2o_authority_enabled", False))
                or bool(
                    getattr(self, "_m3wm_v65_replay_carrier_capacity_enabled", False)
                )
                else 1
            ),
        )

    def _m3wm_detector_raw_history_for_runtime(self):
        expected_maxlen = self._m3wm_detector_raw_history_maxlen()
        history = getattr(self, "_m3wm_detector_raw_history", None)
        if history is None:
            history = deque(maxlen=expected_maxlen)
            self._m3wm_detector_raw_history = history
        elif history.maxlen != expected_maxlen:
            history = deque(history, maxlen=expected_maxlen)
            self._m3wm_detector_raw_history = history
        return history

    def _m3wm_context_raw_history_maxlen(self):
        return max(
            1,
            int(self.cfg.get("m3wm_context_window", 32)),
            int(self.cfg.get("m3wm_memory_min_history", 1)),
            int(
                self.cfg.get(
                    "m3wm_memory_window", self.cfg.get("m3wm_context_window", 32)
                )
            ),
            int(
                self.cfg.get(
                    "m3wm_memory_trigger_window",
                    self.cfg.get("m3wm_context_short_window", 8),
                )
            ),
            int(self.cfg.get("m3wm_context_long_window", 32)),
        )

    def _m3wm_context_raw_history_for_runtime(self):
        expected_maxlen = self._m3wm_context_raw_history_maxlen()
        history = getattr(self, "_m3wm_context_raw_history", None)
        if history is None:
            history = deque(maxlen=expected_maxlen)
            self._m3wm_context_raw_history = history
        elif history.maxlen != expected_maxlen:
            history = deque(history, maxlen=expected_maxlen)
            self._m3wm_context_raw_history = history
        return history

    def _m3wm_clear_context_evidence_hot_cache(self, *, clear_summaries=False):
        self._m3wm_context_evidence_hot_cache_token = None
        self._m3wm_context_evidence_hot_cache = {}
        if clear_summaries:
            self._m3wm_context_evidence_chunk_summary_cache = {}
            self._m3wm_context_evidence_chunk_feature_cache = {}

    def _m3wm_append_context_evidence_stream(
        self, raw_sample, feature, *, terminal=False
    ):
        return False

    def _m3wm_context_evidence_mark_reset(self, *, clear_memory=False):
        stream = None
        if stream is None:
            return False
        if clear_memory:
            stream = ContextEvidenceStream(
                capacity=stream.capacity, episode_id=0, context_generation=0
            )
            self._m3wm_context_evidence_stream = stream
            self._m3wm_context_evidence_episode_id = 0
            self._m3wm_context_evidence_generation = 0
            self._m3wm_context_evidence_generation_counter = 0
            self._m3wm_clear_context_evidence_hot_cache(clear_summaries=True)
            return True
        next_episode_id = max(
            int(getattr(self, "_m3wm_context_evidence_episode_id", 0)) + 1,
            int(stream.episode_id) + 1,
        )
        try:
            stream.mark_reset(int(next_episode_id))
        except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
            return False
        self._m3wm_context_evidence_episode_id = int(next_episode_id)
        self._m3wm_clear_context_evidence_hot_cache(clear_summaries=True)
        return True

    def _m3wm_pack_raw_transition(self, step, obs, action, reward, next_obs, task=None):
        try:
            if (
                isinstance(step, (bool, np.bool_))
                or not isinstance(step, (int, np.integer))
                or int(step) < 1
            ):
                return None
            obs_value = torch.as_tensor(obs).detach().reshape(-1).cpu().clone()
            action_value = torch.as_tensor(action).detach().reshape(-1).cpu().clone()
            reward_value = torch.as_tensor(reward).detach().reshape(-1).cpu().clone()
            next_obs_value = (
                torch.as_tensor(next_obs).detach().reshape(-1).cpu().clone()
            )
            task_value = (
                None
                if task is None
                else torch.as_tensor(task).detach().reshape(-1).cpu().clone()
            )
            if (
                obs_value.numel() < 1
                or next_obs_value.shape != obs_value.shape
                or action_value.numel() != int(self.cfg.action_dim)
                or (reward_value.numel() != 1)
                or (not torch.is_floating_point(obs_value))
                or (not torch.is_floating_point(action_value))
                or (not torch.is_floating_point(reward_value))
                or (not torch.is_floating_point(next_obs_value))
                or (not bool(torch.isfinite(obs_value).all()))
                or (not bool(torch.isfinite(action_value).all()))
                or (not bool(torch.isfinite(reward_value).all()))
                or (not bool(torch.isfinite(next_obs_value).all()))
                or (
                    task_value is not None
                    and (
                        task_value.numel() != 1
                        or not bool(torch.isfinite(task_value.float()).all())
                    )
                )
            ):
                return None
        except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
            return None
        return (
            int(step),
            obs_value,
            action_value,
            reward_value,
            next_obs_value,
            task_value,
        )

    def _m3wm_validate_raw_transition_history(
        self, samples, *, maxlen=None, upper_step=None, require_contiguous=True
    ):
        if not isinstance(samples, (list, tuple, deque)):
            raise ValueError("Invalid raw-transition history container")
        samples = list(samples)
        if maxlen is not None and len(samples) > int(maxlen):
            raise ValueError("Raw-transition history exceeds buffer capacity")
        validated = []
        previous_step = None
        for raw_sample in samples:
            if not isinstance(raw_sample, (list, tuple)) or len(raw_sample) != 6:
                raise ValueError("Invalid raw-transition history sample")
            sample = self._m3wm_pack_raw_transition(*raw_sample)
            if sample is None:
                raise ValueError("Invalid raw-transition history tensor")
            step = int(sample[0])
            if previous_step is not None and (
                step <= previous_step
                or (require_contiguous and step != previous_step + 1)
            ):
                raise ValueError("Raw-transition history step order is invalid")
            if upper_step is not None and step > int(upper_step):
                raise ValueError("Raw-transition history is ahead of runtime")
            validated.append(sample)
            previous_step = step
        return validated

    def _m3wm_raw_transition_batch(self, samples, *, require_contiguous=False):
        try:
            validated = self._m3wm_validate_raw_transition_history(
                samples, require_contiguous=require_contiguous
            )
            if not validated:
                return None
            observations = torch.stack([sample[1] for sample in validated], dim=0).to(
                self.device
            )
            actions = torch.stack([sample[2] for sample in validated], dim=0).to(
                device=self.device, dtype=observations.dtype
            )
            rewards = torch.stack([sample[3] for sample in validated], dim=0).to(
                device=self.device, dtype=observations.dtype
            )
            next_observations = torch.stack(
                [sample[4] for sample in validated], dim=0
            ).to(device=self.device, dtype=observations.dtype)
            task_values = [sample[5] for sample in validated]
            if all((value is None for value in task_values)):
                task = None
            elif any((value is None for value in task_values)):
                return None
            else:
                task = torch.stack(
                    [value.detach().reshape(()) for value in task_values], dim=0
                ).to(self.device)
        except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
            return None
        return {
            "validated": validated,
            "steps": [int(sample[0]) for sample in validated],
            "obs": observations,
            "action": actions,
            "reward": rewards,
            "next_obs": next_observations,
            "task": task,
        }

    def _m3wm_current_version_latent_batch(self, samples):
        batch = self._m3wm_raw_transition_batch(samples, require_contiguous=False)
        if batch is None:
            return None
        count = int(batch["obs"].shape[0])
        joint_obs = torch.cat([batch["obs"], batch["next_obs"]], dim=0)
        joint_task = (
            None
            if batch["task"] is None
            else torch.cat([batch["task"], batch["task"]], dim=0)
        )
        with torch.no_grad():
            joint_z = self.model.encode(joint_obs, joint_task).detach()
        if (
            joint_z.ndim < 2
            or int(joint_z.shape[0]) != 2 * count
            or (not bool(torch.isfinite(joint_z).all()))
        ):
            return None
        return (
            joint_z[:count],
            batch["action"].to(dtype=joint_z.dtype),
            batch["reward"].to(dtype=joint_z.dtype),
            joint_z[count:],
            batch["task"],
        )

    def _m3wm_append_context_raw_transition(
        self, obs, action, reward, next_obs, task=None, prepacked_sample=None
    ):
        history = self._m3wm_context_raw_history_for_runtime()
        step = int(self._m3wm_runtime_step) + 1
        sample = prepacked_sample
        if sample is None:
            sample = self._m3wm_pack_raw_transition(
                step, obs, action, reward, next_obs, task
            )
        elif (
            not isinstance(sample, (tuple, list))
            or len(sample) != 6
            or int(sample[0]) != step
        ):
            sample = None
        if sample is None:
            history.clear()
            return None
        if history and int(history[-1][0]) + 1 != step:
            history.clear()
        history.append(sample)
        return sample

    def _m3wm_detector_episode_prefix_limit(self):
        long_window = int(
            self.cfg.get(
                "m3wm_change_long_window", self.cfg.get("m3wm_context_long_window", 32)
            )
        )
        signed_window = (
            2 * int(getattr(self, "_m3wm_signed_reward_slow_window", 32))
            if bool(getattr(self, "_m3wm_signed_reward_slow_enabled", False))
            else 1
        )
        return max(1, long_window, signed_window)

    @staticmethod
    def _m3wm_clone_detector_raw_sample(sample):
        return (
            int(sample[0]),
            sample[1].detach().clone().cpu(),
            sample[2].detach().clone().cpu(),
            sample[3].detach().clone().cpu(),
            sample[4].detach().clone().cpu(),
            None if sample[5] is None else sample[5].detach().clone().cpu(),
        )

    def _m3wm_ensure_detector_episode_alignment(self):
        if not hasattr(self, "_m3wm_detector_episode_segment_id"):
            runtime_step = int(getattr(self, "_m3wm_runtime_step", 0))
            self._m3wm_detector_episode_segment_id = 0
            self._m3wm_detector_episode_start_step = 1
            self._m3wm_detector_episode_prefix_valid = bool(runtime_step == 0)
            self._m3wm_detector_current_episode_prefix = []
            self._m3wm_detector_previous_episode_start_step = None
            self._m3wm_detector_previous_episode_end_step = None
            self._m3wm_detector_previous_episode_prefix_valid = False
            self._m3wm_detector_previous_episode_prefix = []

    def _m3wm_clear_detector_episode_alignment(self, *, physical_start):
        if not isinstance(physical_start, (bool, np.bool_)):
            raise ValueError("physical_start must be boolean")
        self._m3wm_detector_episode_segment_id = 0
        self._m3wm_detector_episode_start_step = (
            int(getattr(self, "_m3wm_runtime_step", 0)) + 1
        )
        self._m3wm_detector_episode_prefix_valid = bool(physical_start)
        self._m3wm_detector_current_episode_prefix = []
        self._m3wm_detector_previous_episode_start_step = None
        self._m3wm_detector_previous_episode_end_step = None
        self._m3wm_detector_previous_episode_prefix_valid = False
        self._m3wm_detector_previous_episode_prefix = []

    def _m3wm_rotate_detector_episode_alignment(self):
        if (
            str(self.cfg.get("m3wm_change_statistic", "")).lower()
            != "multiscale_live_error_shift"
        ):
            return False
        self._m3wm_ensure_detector_episode_alignment()
        limit = self._m3wm_detector_episode_prefix_limit()
        start_step = int(self._m3wm_detector_episode_start_step)
        end_step = int(getattr(self, "_m3wm_runtime_step", 0))
        prefix = list(self._m3wm_detector_current_episode_prefix)
        complete = bool(
            self._m3wm_detector_episode_prefix_valid
            and end_step >= start_step + limit - 1
            and (len(prefix) == limit)
            and all(
                (
                    int(sample[0]) == start_step + index
                    for (index, sample) in enumerate(prefix)
                )
            )
        )
        self._m3wm_detector_previous_episode_start_step = (
            start_step if end_step >= start_step else None
        )
        self._m3wm_detector_previous_episode_end_step = (
            end_step if end_step >= start_step else None
        )
        self._m3wm_detector_previous_episode_prefix_valid = complete
        self._m3wm_detector_previous_episode_prefix = (
            [self._m3wm_clone_detector_raw_sample(sample) for sample in prefix]
            if complete
            else []
        )
        self._m3wm_detector_episode_segment_id = (
            int(self._m3wm_detector_episode_segment_id) + 1
        )
        self._m3wm_detector_episode_start_step = end_step + 1
        self._m3wm_detector_episode_prefix_valid = True
        self._m3wm_detector_current_episode_prefix = []
        return complete

    def _m3wm_append_detector_episode_prefix(self, sample):
        self._m3wm_ensure_detector_episode_alignment()
        if not self._m3wm_detector_episode_prefix_valid:
            return False
        limit = self._m3wm_detector_episode_prefix_limit()
        prefix = self._m3wm_detector_current_episode_prefix
        if len(prefix) >= limit:
            return True
        expected_step = int(self._m3wm_detector_episode_start_step) + len(prefix)
        if int(sample[0]) != expected_step:
            self._m3wm_detector_episode_prefix_valid = False
            prefix.clear()
            return False
        prefix.append(self._m3wm_clone_detector_raw_sample(sample))
        return True

    def _m3wm_append_detector_raw_transition(
        self, obs, action, reward, next_obs, task=None, prepacked_sample=None
    ):
        if (
            str(self.cfg.get("m3wm_change_statistic", "")).lower()
            != "multiscale_live_error_shift"
        ):
            return False
        history = self._m3wm_detector_raw_history_for_runtime()
        step = int(self._m3wm_runtime_step) + 1
        sample = prepacked_sample
        if sample is None:
            sample = self._m3wm_pack_raw_transition(
                step, obs, action, reward, next_obs, task
            )
        elif (
            not isinstance(sample, (tuple, list))
            or len(sample) != 6
            or int(sample[0]) != step
        ):
            sample = None
        if sample is None or (history and int(history[-1][0]) + 1 != step):
            history.clear()
            self._m3wm_ensure_detector_episode_alignment()
            self._m3wm_detector_episode_prefix_valid = False
            self._m3wm_detector_current_episode_prefix.clear()
            return False
        history.append(sample)
        self._m3wm_append_detector_episode_prefix(sample)
        return True

    def _m3wm_recompute_detector_raw_features(self, samples, *, theta=None):
        try:
            samples = list(samples)
            batch = self._m3wm_raw_transition_batch(samples)
            if batch is None:
                return None
            kwargs = {"task": batch["task"]}
            if theta is not None:
                kwargs["theta"] = theta
            with torch.no_grad():
                features = self.model.build_context_features_from_observations(
                    batch["obs"],
                    batch["action"],
                    batch["reward"],
                    batch["next_obs"],
                    **kwargs,
                ).detach()
        except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
            return None
        if (
            features.ndim != 2
            or int(features.shape[0]) != len(samples)
            or (not torch.is_floating_point(features))
            or (not bool(torch.isfinite(features).all()))
        ):
            return None
        return features

    def pop_context_identity_event_artifact(self):
        value = getattr(self, "_m3wm_last_identity_event_artifact", None)
        self._m3wm_last_identity_event_artifact = None
        return value

    def _m3wm_source_theta_shadow_is_enabled(self):
        return bool(
            str(self.cfg.get("m3wm_change_statistic", "")).lower()
            == "multiscale_live_error_shift"
        )

    @staticmethod
    def _m3wm_empty_source_theta_shadow(*, enabled, reason):

        def empty_components():
            return {
                "valid": False,
                "score": float("nan"),
                "fast_dynamics_score": float("nan"),
                "fast_abs_reward_score": float("nan"),
                "slow_signed_reward_score": float("nan"),
                "threshold": float("nan"),
            }

        def empty_persistence():
            value = empty_components()
            value.update(
                {"support_count": 0, "required_support_count": 0, "blocks": []}
            )
            return value

        return {
            "enabled": bool(enabled),
            "valid": False,
            "binding_valid": False,
            "reason": str(reason),
            "proposal_step": None,
            "due_step": None,
            "theta_l1": float("nan"),
            "theta_l2": float("nan"),
            "theta_linf": float("nan"),
            "live_same_version": empty_components(),
            "source_same_version": empty_components(),
            "live_persistence": empty_persistence(),
            "source_persistence": empty_persistence(),
            "reward_absorption_rescue": {
                "enabled": False,
                "evaluated": False,
                "valid": False,
                "authorized": False,
                "reason": "not_evaluated",
            },
        }

    def _m3wm_source_theta_shadow_metrics(self):
        shadow = getattr(self, "_m3wm_last_source_theta_shadow", None)
        if not isinstance(shadow, dict):
            shadow = self._m3wm_empty_source_theta_shadow(
                enabled=self._m3wm_source_theta_shadow_is_enabled(),
                reason="not_evaluated",
            )

        def number(value):
            try:
                if value is None:
                    return float("nan")
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()
                return float(value)
            except (TypeError, ValueError, RuntimeError, OverflowError):
                return float("nan")

        def metric(value):
            return torch.tensor(number(value))

        reason_code = {
            "disabled": 0,
            "not_evaluated": 1,
            "ok": 2,
            "missing_transaction_binding": 3,
            "transaction_step_binding_failed": 4,
            "active_context_binding_failed": 5,
            "proposal_theta_binding_failed": 6,
            "invalid_transaction_binding": 7,
            "source_same_version_invalid": 8,
            "source_persistence_invalid": 9,
            "source_evaluation_failed": 10,
        }.get(str(shadow.get("reason", "not_evaluated")), -1)
        metrics = {
            "source_theta_shadow_enabled": torch.tensor(
                float(bool(shadow.get("enabled", False)))
            ),
            "source_theta_shadow_valid": torch.tensor(
                float(bool(shadow.get("valid", False)))
            ),
            "source_theta_shadow_binding_valid": torch.tensor(
                float(bool(shadow.get("binding_valid", False)))
            ),
            "source_theta_shadow_reason_code": torch.tensor(float(reason_code)),
            "source_theta_shadow_proposal_step": metric(shadow.get("proposal_step")),
            "source_theta_shadow_due_step": metric(shadow.get("due_step")),
            "source_theta_shadow_theta_l1": metric(shadow.get("theta_l1")),
            "source_theta_shadow_theta_l2": metric(shadow.get("theta_l2")),
            "source_theta_shadow_theta_linf": metric(shadow.get("theta_linf")),
        }
        rescue = shadow.get("reward_absorption_rescue", {})
        if not isinstance(rescue, dict):
            rescue = {}
        rescue_reason_code = {
            "disabled": 0,
            "not_evaluated": 1,
            "invalid_contract": 2,
            "transaction_binding_failed": 3,
            "source_theta_shadow_invalid": 4,
            "witness_invalid": 5,
            "persistence_support_invalid": 6,
            "nonfinite_evidence": 7,
            "regime_d_z_gate_failed": 8,
            "proposal_not_reward_only": 9,
            "live_witness_not_faded": 10,
            "source_same_version_gate_failed": 11,
            "persistence_support_failed": 12,
            "persistence_gate_failed": 13,
            "reward_sign_mismatch": 14,
            "authorized": 15,
        }.get(str(rescue.get("reason", "not_evaluated")), -1)
        metrics.update(
            {
                "source_theta_shadow_reward_rescue_enabled": torch.tensor(
                    float(bool(rescue.get("enabled", False)))
                ),
                "source_theta_shadow_reward_rescue_evaluated": torch.tensor(
                    float(bool(rescue.get("evaluated", False)))
                ),
                "source_theta_shadow_reward_rescue_valid": torch.tensor(
                    float(bool(rescue.get("valid", False)))
                ),
                "source_theta_shadow_reward_rescue_authorized": torch.tensor(
                    float(bool(rescue.get("authorized", False)))
                ),
                "source_theta_shadow_reward_rescue_reason_code": torch.tensor(
                    float(rescue_reason_code)
                ),
            }
        )
        for prefix, key, persistence in (
            ("live_same_version", "live_same_version", False),
            ("source_same_version", "source_same_version", False),
            ("live_persistence", "live_persistence", True),
            ("source_persistence", "source_persistence", True),
        ):
            values = shadow.get(key, {})
            if not isinstance(values, dict):
                values = {}
            metrics.update(
                {
                    f"source_theta_shadow_{prefix}_valid": torch.tensor(
                        float(bool(values.get("valid", False)))
                    ),
                    f"source_theta_shadow_{prefix}_score": metric(values.get("score")),
                    f"source_theta_shadow_{prefix}_dynamics": metric(
                        values.get("fast_dynamics_score")
                    ),
                    f"source_theta_shadow_{prefix}_abs_reward": metric(
                        values.get("fast_abs_reward_score")
                    ),
                    f"source_theta_shadow_{prefix}_signed_reward": metric(
                        values.get("slow_signed_reward_score")
                    ),
                    f"source_theta_shadow_{prefix}_threshold": metric(
                        values.get("threshold")
                    ),
                }
            )
            if not persistence:
                continue
            metrics.update(
                {
                    f"source_theta_shadow_{prefix}_support": metric(
                        values.get("support_count", 0)
                    ),
                    f"source_theta_shadow_{prefix}_required": metric(
                        values.get("required_support_count", 0)
                    ),
                }
            )
            blocks = values.get("blocks", [])
            if not isinstance(blocks, (list, tuple)):
                blocks = []
            for index in range(4):
                block = blocks[index] if index < len(blocks) else {}
                if not isinstance(block, dict):
                    block = {}
                metrics.update(
                    {
                        f"source_theta_shadow_{prefix}_block_{index}_score": metric(
                            block.get("score")
                        ),
                        f"source_theta_shadow_{prefix}_block_{index}_dynamics": metric(
                            block.get("fast_dynamics_score")
                        ),
                        f"source_theta_shadow_{prefix}_block_{index}_abs_reward": metric(
                            block.get("fast_abs_reward_score")
                        ),
                        f"source_theta_shadow_{prefix}_block_{index}_signed_reward": metric(
                            block.get("slow_signed_reward_score")
                        ),
                    }
                )
        return metrics

    def _m3wm_active_probe_metrics(self):
        receipt = getattr(self, "_m3wm_last_active_probe_receipt", None)
        if not isinstance(receipt, dict):
            receipt = {}
        selection = receipt.get("selection")
        if not isinstance(selection, dict):
            selection = {}
        reason_code = {
            "disabled": 0,
            "not_evaluated": 1,
            "no_pending_proposal": 2,
            "outside_probe_interval": 3,
            "insufficient_context_hypotheses": 4,
            "proposal_theta_binding_failed": 5,
            "planner_tensor_contract_failed": 6,
            "context_theta_contract_failed": 7,
            "identifiability_unavailable": 8,
            "maximin_elite_selected": 9,
            "probe_evaluation_exception": 10,
        }.get(str(receipt.get("reason", "not_evaluated")), -1)

        def number(value, default=float("nan")):
            try:
                if value is None:
                    return float(default)
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()
                return float(value)
            except (TypeError, ValueError, RuntimeError, OverflowError):
                return float(default)

        return {
            "context_active_probe_enabled": torch.tensor(
                float(bool(receipt.get("enabled", False)))
            ),
            "context_active_probe_applied": torch.tensor(
                float(bool(receipt.get("override_applied", False)))
            ),
            "context_active_probe_reason_code": torch.tensor(float(reason_code)),
            "context_active_probe_proposal_step": torch.tensor(
                number(receipt.get("proposal_step"))
            ),
            "context_active_probe_due_step": torch.tensor(
                number(receipt.get("due_step"))
            ),
            "context_active_probe_hypothesis_count": torch.tensor(
                number(
                    selection.get(
                        "hypothesis_count", len(receipt.get("hypothesis_ids", []))
                    ),
                    default=0.0,
                )
            ),
            "context_active_probe_elite_count": torch.tensor(
                number(selection.get("elite_count"), default=0.0)
            ),
            "context_active_probe_selected_index": torch.tensor(
                number(selection.get("selected_index"))
            ),
            "context_active_probe_selected_score": torch.tensor(
                number(selection.get("selected_score"))
            ),
        }

    def _m3wm_initial_context_bootstrap_due(self, initial_step_calibration):
        mode = str(self.cfg.get("m3wm_change_statistic", "")).lower()
        return (False, False)[1]

    def _m3wm_live_bocpd_edge_gate_for_mode(self):
        if (
            str(self.cfg.get("m3wm_change_statistic", "")).lower()
            != "bocpd_live_three_stream"
        ):
            return None
        raise RuntimeError(
            "bocpd_live_three_stream has no Bayesian detector for edge gating"
        )

    def _m3wm_empirical_change_null_enabled(self):
        return (False, False)[1]

    def _m3wm_change_null_active_context_id(self):
        context_id = None
        if context_id is None:
            return None
        try:
            context_id = int(context_id)
        except (TypeError, ValueError, OverflowError):
            return None
        return None

    def _m3wm_change_null_reset(self):
        calibrator = getattr(self, "_m3wm_change_null_calibrator", None)
        if calibrator is not None:
            calibrator.reset()
        self._m3wm_initial_null_dynamics_statistics = []
        self._m3wm_initial_null_reward_statistics = []
        self._m3wm_change_null_std_floor_hits = 0
        self._m3wm_change_null_statistic_count = 0
        if hasattr(self, "_m3wm_last_change_null_metadata"):
            self._m3wm_last_change_null_metadata = {
                "mode": getattr(
                    self, "_m3wm_change_null_calibration_mode", "fixed_floor"
                ),
                "alpha": float(calibrator.alpha if calibrator is not None else 0.01),
                "effective_threshold": float(
                    calibrator.fallback_threshold
                    if calibrator is not None
                    else self.cfg.get("m3wm_change_shift_threshold", 1.8)
                ),
                "source": "fallback",
                "ready": False,
                "sample_count": 0,
                "context_sample_count": 0,
                "robot_sample_count": 0,
                "percentile": float("nan"),
                "tail_probability": float("nan"),
                "dynamics_percentile": float("nan"),
                "dynamics_tail_probability": float("nan"),
                "reward_percentile": float("nan"),
                "reward_tail_probability": float("nan"),
                "frozen": False,
                "clean_observation_count": 0,
                "rejected_observation_count": 0,
            }

    def _m3wm_change_null_clean_transition_eligible(
        self,
        *,
        score,
        initial_step_calibration,
        pending_before,
        candidate_step,
        transaction_outcome,
        confirmed_trigger,
        terminal,
        trial_started,
        bootstrap_committed,
    ):
        if not self._m3wm_empirical_change_null_enabled():
            return False
        if bool(initial_step_calibration) or bool(pending_before):
            return False
        if bool(candidate_step) or bool(confirmed_trigger):
            return False
        if bool(transaction_outcome) or bool(trial_started):
            return False
        if bool(getattr(self, "_m3wm_soft_reuse_applied_this_step", False)):
            return False
        if bool(terminal) or bool(bootstrap_committed):
            return False
        if self._m3wm_has_pending_change():
            return False
        if bool(getattr(self, "_m3wm_context_evidence_trial_pending", False)):
            return False
        if bool(getattr(self, "_m3wm_last_context_changed", False)):
            return False
        if bool(getattr(self, "_m3wm_episode_change_latched", False)):
            return False
        try:
            value = float(score)
        except (TypeError, ValueError, OverflowError):
            return False
        if not np.isfinite(value):
            return False
        return False

    def _m3wm_update_empirical_change_null(
        self,
        *,
        score,
        components,
        initial_step_calibration,
        pending_before,
        candidate_step,
        transaction_outcome,
        confirmed_trigger,
        terminal,
        trial_started,
        bootstrap_committed,
    ):
        if not self._m3wm_empirical_change_null_enabled():
            return
        calibrator = getattr(self, "_m3wm_change_null_calibrator", None)
        if calibrator is None:
            return
        context_id = self._m3wm_change_null_active_context_id()
        dynamics = None
        reward = None
        if isinstance(components, dict):
            dynamics = components.get("fast_dynamics_score")
            reward = components.get("slow_signed_reward_score")
            if reward is None:
                reward = components.get("fast_abs_reward_score")
            try:
                self._m3wm_change_null_std_floor_hits += int(
                    components.get("feature_std_floor_hits", 0)
                )
            except (TypeError, ValueError, OverflowError):
                pass
        try:
            if np.isfinite(float(score)):
                self._m3wm_change_null_statistic_count += 1
        except (TypeError, ValueError, OverflowError):
            pass
        metadata = calibrator.metadata(
            score, context_id=context_id, dynamics=dynamics, reward=reward
        )
        metadata["std_floor_hits"] = int(
            getattr(self, "_m3wm_change_null_std_floor_hits", 0)
        )
        metadata["statistic_count"] = int(
            getattr(self, "_m3wm_change_null_statistic_count", 0)
        )
        self._m3wm_last_change_null_metadata = metadata
        pending_after = self._m3wm_has_pending_change()
        if pending_after:
            calibrator.freeze()
            metadata["frozen"] = True
            self._m3wm_last_change_null_metadata = metadata
            return
        calibrator.unfreeze()
        metadata["frozen"] = False
        if not self._m3wm_change_null_clean_transition_eligible(
            score=score,
            initial_step_calibration=initial_step_calibration,
            pending_before=pending_before,
            candidate_step=candidate_step,
            transaction_outcome=transaction_outcome,
            confirmed_trigger=confirmed_trigger,
            terminal=terminal,
            trial_started=trial_started,
            bootstrap_committed=bootstrap_committed,
        ):
            return
        calibrator.observe(
            score, context_id=context_id, dynamics=dynamics, reward=reward
        )
        metadata["clean_observation_count"] = int(calibrator.clean_observation_count)
        metadata["rejected_observation_count"] = int(
            calibrator.rejected_observation_count
        )
        self._m3wm_last_change_null_metadata = metadata

    @torch.no_grad()
    def pure_online_finalize_terminal(self):
        return None

    def pure_online_artifact_snapshot(self):
        return None

    def reset_context(self, clear_memory=False, preserve_runtime_step=False):
        if not isinstance(clear_memory, (bool, np.bool_)):
            raise ValueError("clear_memory must be boolean")
        if not isinstance(preserve_runtime_step, (bool, np.bool_)):
            raise ValueError("preserve_runtime_step must be boolean")
        clear_memory = bool(clear_memory)
        preserve_runtime_step = bool(preserve_runtime_step)
        if preserve_runtime_step and (not clear_memory):
            raise ValueError("preserve_runtime_step requires clear_memory=True")
        pending_before_reset = self._m3wm_pending_change_tracker.get()
        v9_pending_across_reset = (False, False)[1]
        self._m3wm_context_evidence_mark_reset(clear_memory=clear_memory)
        exemplar_bank = getattr(self, "_m3wm_context_exemplar_bank", None)
        if exemplar_bank is not None:
            if clear_memory:
                exemplar_bank.reset()
                self._m3wm_v57_shadow_registration = _m3wm_v57_shadow_receipt(
                    "feature_flag_disabled", state="DISABLED"
                )
                self._m3wm_v57_shadow_owner_ledger = None
                self._m3wm_v57_shadow_owner_evaluation = _m3wm_v57_shadow_receipt(
                    "feature_flag_disabled", state="DISABLED"
                )
                self._m3wm_v57_shadow_candidate_self_consistency = None
            else:
                stable_exemplar_episode = bool(not self._m3wm_has_pending_change())
                exemplar_bank.finish_episode(
                    stable=stable_exemplar_episode,
                    active_context_id=None,
                    context_count=0,
                )
        if not clear_memory:
            self._m3wm_rotate_detector_episode_alignment()
        if clear_memory:
            self._m3wm_episodic_theta_reward_guard_slots = {}
            self._m3wm_episodic_theta_baseline_first = {}
            self._m3wm_episodic_theta_baseline_first_last_event = "cleared"
            self._m3wm_episodic_theta_reward_guard_accept_count = 0
            self._m3wm_episodic_theta_reward_guard_reject_count = 0
            self._m3wm_episodic_theta_reward_guard_last_decision = "none"
            self._m3wm_episodic_theta_reward_guard_last_reason = "none"
            self._m3wm_episodic_theta_reward_guard_last_score = float("nan")
        else:
            self._m3wm_finalize_episodic_theta_reward_guard()
            self._m3wm_restore_episodic_theta_best_for_clean_episode()
            self._m3wm_reset_episodic_theta_baseline_first_episode()
        self._m3wm_reset_episodic_theta_guard_episode()
        pending_theta_reset_guarded = False
        if not clear_memory:
            pending_state = self._m3wm_pending_change_tracker.get()
            preserve_sequential_pending = bool(
                v9_pending_across_reset
                and isinstance(pending_state, dict)
                and (
                    int(pending_state.get("confirmation_due_step", -1))
                    > int(self._m3wm_runtime_step)
                )
            )
            if preserve_sequential_pending:
                pending_theta_reset_guarded = True
            else:
                pending_aborted_at_reset = self._m3wm_abort_pending_at_episode_reset()
        self._m3wm_online_theta_clean_episode_updates = 0
        if clear_memory or not self._m3wm_has_pending_change():
            self._m3wm_end_online_theta_transaction(context_committed=False)
        else:
            transaction_state = self._m3wm_online_theta_transaction_state_dict()
            if transaction_state["terminal_context_hold"]:
                raise RuntimeError(
                    "A pending transaction cannot carry a terminal theta hold"
                )
        self._m3wm_online_theta_writeback_release_armed = False
        context_runtime_enabled = self._m3wm_context_runtime_enabled()
        episode_reset_from_memory_theta = getattr(
            self, "_m3wm_episode_reset_from_memory_theta", False
        )
        if not isinstance(episode_reset_from_memory_theta, (bool, np.bool_)):
            raise ValueError("m3wm_episode_reset_from_memory_theta must be boolean")
        episode_reset_from_memory_theta = bool(episode_reset_from_memory_theta)
        if (
            not clear_memory
            and context_runtime_enabled
            and self._m3wm_current_episode_change_history
        ):
            stable_episode = (
                self._m3wm_episode_is_null_calibration
                or not self._m3wm_episode_change_latched
            )
            stable_episode_observe_due = False
            stable_episode_observe_due and self._m3wm_has_pending_change()
            self._m3wm_previous_episode_change_history = (
                self._m3wm_current_episode_change_history
            )
            self._m3wm_current_episode_change_history = []
            self._m3wm_previous_episode_system_id = self._m3wm_current_episode_system_id
            self._m3wm_current_episode_system_id = []
        self._m3wm_episode_change_latched = False
        self._m3wm_episode_is_null_calibration = False
        self._m3wm_episode_is_initial_null_calibration = False
        self._m3wm_episode_null_max_statistic = float("-inf")
        self._m3wm_episode_null_statistics = []
        if (
            not clear_memory
            and context_runtime_enabled
            and (self._m3wm_initial_null_episodes_remaining > 0)
        ):
            self._m3wm_episode_change_latched = True
            self._m3wm_episode_is_null_calibration = True
            self._m3wm_episode_is_initial_null_calibration = True
            self._m3wm_initial_null_episodes_remaining -= 1
        elif (
            not clear_memory
            and context_runtime_enabled
            and (self._m3wm_change_skip_episodes_remaining > 0)
        ):
            self._m3wm_episode_change_latched = True
            self._m3wm_episode_is_null_calibration = True
            self._m3wm_change_skip_episodes_remaining -= 1
        self._m3wm_context_history.clear()
        self._m3wm_context_raw_history_for_runtime().clear()
        if pending_theta_reset_guarded:
            self._m3wm_diag_increment_guard("episode_reset_theta")
        else:
            self._m3wm_online_adaptation_history.clear()
        self._m3wm_last_context_feature = None
        if not pending_theta_reset_guarded:
            self._m3wm_context_version = 0
        self._m3wm_last_change_triggered = False
        self._m3wm_last_change_statistic = float("nan")
        self._m3wm_last_fast_dynamics_shift = float("nan")
        self._m3wm_last_fast_abs_reward_shift = float("nan")
        self._m3wm_last_slow_signed_reward_shift = float("nan")
        self._m3wm_last_existing_log_posterior = float("nan")
        self._m3wm_last_new_log_posterior = float("nan")
        if clear_memory:
            self._m3wm_diag_orphan_transaction("full_context_reset")
            self._m3wm_reset_reward_identity_state()
            diagnostic_residuals = getattr(
                self, "_m3wm_diagnostic_reward_residual_history", None
            )
            if diagnostic_residuals is not None:
                diagnostic_residuals.clear()
            if not preserve_runtime_step:
                self._m3wm_runtime_step = 0
            self._m3wm_online_theta_writeback_not_before_runtime_step = 0
            self._m3wm_change_history.clear()
            self._m3wm_regime_ut_history.clear()
            bocpd_ut_history = getattr(self, "_m3wm_bocpd_ut_history", None)
            if bocpd_ut_history is not None:
                bocpd_ut_history.clear()
            self._m3wm_bocpd_fixed_null_samples = []
            self._m3wm_bocpd_fixed_location = None
            self._m3wm_bocpd_fixed_scale = None
            self._m3wm_bocpd_fixed_calibrated = False
            self._m3wm_current_episode_change_history = []
            self._m3wm_previous_episode_change_history = []
            self._m3wm_current_episode_system_id = []
            self._m3wm_previous_episode_system_id = []
            self._m3wm_local_system_id_history.clear()
            self._m3wm_detector_raw_history_for_runtime().clear()
            self._m3wm_clear_detector_episode_alignment(
                physical_start=not preserve_runtime_step
            )
            self._m3wm_change_skip_episodes_remaining = 0
            self._m3wm_initial_null_episodes_remaining = max(
                0, int(self.cfg.get("m3wm_change_initial_null_episodes", 0))
            )
            self._m3wm_initial_null_steps_remaining = max(
                0, int(self.cfg.get("m3wm_change_initial_null_steps", 0))
            )
            self._m3wm_episode_is_null_calibration = False
            self._m3wm_episode_is_initial_null_calibration = False
            self._m3wm_episode_null_max_statistic = float("-inf")
            self._m3wm_episode_null_statistics = []
            self._m3wm_initial_null_statistics = []
            self._m3wm_change_null_reset()
            self._m3wm_last_null_reference_statistic = float("nan")
            self._m3wm_adaptive_change_threshold = float(
                self.cfg.get("m3wm_change_shift_threshold", 2.5)
            )
            self._m3wm_raw_theta = None
            self._m3wm_last_router_proposal = None
            self._m3wm_last_memory_theta = None
            self._m3wm_last_router_blend = float("nan")
            self._m3wm_smoothed_theta = None
            self._m3wm_last_theta = None
            self._m3wm_theta_version = -1
            self._m3wm_active_context_id = None
            live_bocpd_gate = getattr(self, "_m3wm_live_bocpd_edge_gate", None)
            if live_bocpd_gate is not None:
                live_bocpd_gate.reset()
            self._m3wm_context_creations = 0
            self._m3wm_context_recalls = 0
            self._m3wm_confirmed_change_count = 0
            self._m3wm_rejected_change_count = 0
            self._m3wm_bocpd_a1_rejected_count = 0
            self._m3wm_bocpd_a2_rejected_count = 0
            self._m3wm_last_confirmed_event = None
            if self._m3wm_stable_context_selector is not None:
                self._m3wm_stable_context_selector.reset()
            if self._m3wm_soft_context_reuse is not None:
                self._m3wm_soft_context_reuse.reset()
            if getattr(self, "_m3wm_reversible_reuse_trial", None) is not None:
                self._m3wm_reversible_reuse_trial.reset()
            self._m3wm_reversible_trial_nomination_count = 0
            self._m3wm_reversible_trial_commit_count = 0
            self._m3wm_reversible_trial_abort_count = 0
            self._m3wm_last_reversible_trial_receipt = {
                "schema": "m3wm_reversible_reuse_trial",
                "version": M3WM_REVERSIBLE_REUSE_TRIAL_VERSION,
                "state": "INACTIVE",
                "active": False,
                "applied": False,
                "committed_target_context_id": None,
                "target_memory_writes": 0,
                "official_active_context_changed": False,
                "reason": "context_reset",
            }
            self._m3wm_soft_reuse_applied_this_step = False
            self._m3wm_last_trigger_source = "none"
            self._m3wm_regime_pre_window = None
            self._m3wm_regime_post_window = None
            self._m3wm_regime_gap_remaining = 0
            self._m3wm_regime_event_step = -1
            self._m3wm_regime_rejected_count = 0
            self._m3wm_bocpd_window_retriggers = 0
            self._m3wm_regime_last_dz = float("nan")
            self._m3wm_regime_last_s_post = float("nan")
            self._m3wm_dp_identity_dismissed = 0
            self._m3wm_dp_identity_created = 0
            self._m3wm_dp_identity_switched = 0
            self._m3wm_last_dp_best_existing_context = -1
            self._m3wm_last_dp_active_log_posterior = float("nan")
            self._m3wm_last_dp_recall_advantage = float("nan")
            self._m3wm_last_dp_recall_emission_advantage = float("nan")
            self._m3wm_last_dp_recall_emission_pass = False
            self._m3wm_last_dp_recall_candidate_context = -1
            self._m3wm_last_dp_recall_candidate_rms = float("nan")
            self._m3wm_last_dp_recall_candidate_rms_threshold = float("nan")
            self._m3wm_last_dp_recall_candidate_support_pass = False
            self._m3wm_last_dp_identity_decision = "none"
            self._m3wm_last_dp_departure_reason = "none"
            self._m3wm_last_dp_departure_active_standardized_rms = float("nan")
            self._m3wm_last_dp_departure_rms_threshold = float("nan")
            self._m3wm_last_dp_departure_regime_d_z = float("nan")
            self._m3wm_last_dp_proposal_fast_dynamics_shift = float("nan")
            self._m3wm_last_dp_proposal_fast_abs_reward_shift = float("nan")
            self._m3wm_last_dp_proposal_slow_signed_reward_shift = float("nan")
            self._m3wm_last_dp_same_version_witness_score = float("nan")
            self._m3wm_last_dp_same_version_witness_threshold = float("nan")
            self._m3wm_last_dp_same_version_witness_dynamics = float("nan")
            self._m3wm_last_dp_same_version_witness_reward = float("nan")
            self._m3wm_last_dp_pending_persistence_score = float("nan")
            self._m3wm_last_dp_pending_persistence_threshold = float("nan")
            self._m3wm_last_dp_pending_persistence_dynamics = float("nan")
            self._m3wm_last_dp_pending_persistence_reward = float("nan")
            self._m3wm_last_source_theta_shadow = self._m3wm_empty_source_theta_shadow(
                enabled=True, reason="not_evaluated"
            )
            self._m3wm_last_dp_physical_witness_valid = False
            self._m3wm_last_dp_physical_witness_pass = False
            self._m3wm_last_dp_physical_witness_score = float("nan")
            self._m3wm_last_dp_physical_witness_threshold = float("nan")
            self._m3wm_last_dp_physical_witness_support = 0
            self._m3wm_last_dp_physical_witness_required = 0
            self._m3wm_last_dp_physical_witness_mutual_score = float("nan")
            self._m3wm_last_dp_physical_witness_mutual_support = 0
            self._m3wm_last_dp_physical_witness_min_rank = 0
            self._m3wm_last_dp_physical_departure_used = False
            self._m3wm_dp_prototype_updates = 0
            self._m3wm_identity_audit_pending_evidence = None
            self._m3wm_episode_identity_audit_proposal_count = 0
            self._m3wm_last_identity_audit_step = -1
            self._m3wm_last_identity_audit_score = float("nan")
            self._m3wm_last_identity_audit_reason = "none"
            self._m3wm_last_context_changed = False
            self._m3wm_pending_trigger_steps = 0
            objective_evaluator = getattr(self, "_m3wm_objective_evaluator", None)
            if objective_evaluator is not None:
                if objective_evaluator.phase in {"proposal", "complete"}:
                    objective_evaluator.cancel_proposal()
                elif objective_evaluator.phase != "building_null":
                    raise RuntimeError(
                        "Unknown proposal-local objective phase at full reset"
                    )
                if not preserve_runtime_step:
                    self._m3wm_objective_evaluator = (
                        ProposalLocalObjectiveChangeEvaluator(
                            objective_evaluator.config
                        )
                    )
            self._m3wm_objective_proposal_step = None
            self._m3wm_objective_proposal_available = False
            self._m3wm_last_objective_decision = None
            if not preserve_runtime_step:
                self._m3wm_objective_proposal_count = 0
                self._m3wm_objective_confirm_count = 0
                self._m3wm_objective_reject_count = 0
                self._m3wm_objective_unavailable_count = 0
                self._m3wm_pending_change_tracker.clear()
                self._m3wm_context_evidence_trial_pending = False
                if not hasattr(self, "_m3wm_identity_quarantine"):
                    self._m3wm_identity_quarantine = M3wmOneShotQuarantine()
            self._m3wm_identity_quarantine.reset()
            self._m3wm_discard_identity_proposal_rollback()
            self._m3wm_diagnostic_suppressed_pending = False
            self._m3wm_active_context_origin = "unassigned"
        elif not pending_theta_reset_guarded:
            self._m3wm_theta_version = 0
        if self.cfg.get("enable_m3wm", False) and (not pending_theta_reset_guarded):
            self.model.set_runtime_theta(self._m3wm_last_theta)

    def _m3wm_freeze_reward_identity_null(self):
        if (
            not getattr(self, "_m3wm_reward_identity_enabled", False)
            or self._m3wm_reward_identity_null_frozen
        ):
            return self._m3wm_reward_identity_null_frozen
        calibration = _reward_identity_null_location_scale(
            self._m3wm_reward_identity_null_history,
            min_samples=_M3WM_REWARD_IDENTITY_WINDOW,
        )
        if calibration is None:
            return False
        self._m3wm_reward_identity_null_median, self._m3wm_reward_identity_null_mad = (
            calibration
        )
        self._m3wm_reward_identity_null_frozen = True
        return True

    def _m3wm_reset_reward_identity_state(self):
        history = getattr(self, "_m3wm_reward_identity_history", None)
        if history is not None:
            history.clear()
        null_history = getattr(self, "_m3wm_reward_identity_null_history", None)
        if null_history is not None:
            null_history.clear()
        self._m3wm_reward_identity_null_median = float("nan")
        self._m3wm_reward_identity_null_mad = float("nan")
        self._m3wm_reward_identity_null_frozen = False

    def _m3wm_reward_identity_observe_context_feature(
        self, feature, initial_null_step=False
    ):
        if not getattr(self, "_m3wm_reward_identity_enabled", False):
            return
        index = self._m3wm_reward_identity_residual_index
        value = float("nan")
        if index is not None and int(index) < int(feature.numel()):
            value = float(feature[int(index)].detach().cpu())
        self._m3wm_reward_identity_history.append(value)
        if initial_null_step and (not self._m3wm_reward_identity_null_frozen):
            self._m3wm_reward_identity_null_history.append(value)
            if int(self._m3wm_initial_null_steps_remaining) == 1:
                self._m3wm_freeze_reward_identity_null()

    def _m3wm_p2_identity_enabled(self):
        return bool(
            getattr(self, "_m3wm_reward_identity_enabled", False)
            and str(self.cfg.get("m3wm_context_assignment_mode", "regime_gate")).lower()
            == "dp_identity"
        )

    def _m3wm_clone_adaptation_history(self, history, device=None):
        del device
        return self._m3wm_validate_raw_transition_history(
            history,
            maxlen=getattr(
                getattr(self, "_m3wm_online_adaptation_history", None), "maxlen", None
            ),
            require_contiguous=False,
        )

    def _m3wm_clone_active_consensus_context_transaction_state(
        self, value, *, proposal_step=None
    ):
        required = {
            "schema",
            "version",
            "proposal_step",
            "active_context_id",
            "active_context_origin",
            "context_count",
            "slot_digests",
            "state_sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value.get("schema") != "m3wm_active_consensus_proposal_context_state"
            or (value.get("version") != 1)
        ):
            raise ValueError("Invalid R16 proposal Context-state keys")
        raw_step = value["proposal_step"]
        raw_active = value["active_context_id"]
        active_origin = value["active_context_origin"]
        raw_count = value["context_count"]
        if (
            isinstance(raw_step, (bool, np.bool_))
            or not isinstance(raw_step, (int, np.integer))
            or int(raw_step) < 1
            or isinstance(raw_active, (bool, np.bool_))
            or (not isinstance(raw_active, (int, np.integer)))
            or (int(raw_active) < 0)
            or (not isinstance(active_origin, str))
            or (not active_origin)
            or isinstance(raw_count, (bool, np.bool_))
            or (not isinstance(raw_count, (int, np.integer)))
            or (int(raw_count) < 1)
        ):
            raise ValueError("Invalid R16 proposal Context-state coordinates")
        step = int(raw_step)
        active_context_id = int(raw_active)
        context_count = int(raw_count)
        if active_context_id >= context_count:
            raise ValueError("R16 proposal active Context is outside memory")
        if proposal_step is not None and step != int(proposal_step):
            raise ValueError("R16 proposal Context state is bound elsewhere")
        digests = value["slot_digests"]
        digest_keys = {"schema", "version", "num_contexts", "memory_sha256", "slots"}
        if (
            not isinstance(digests, dict)
            or set(digests) != digest_keys
            or digests.get("schema") != "m3wm_context_slot_digests"
            or (digests.get("version") != 1)
            or isinstance(digests.get("num_contexts"), (bool, np.bool_))
            or (not isinstance(digests.get("num_contexts"), (int, np.integer)))
            or (int(digests["num_contexts"]) != context_count)
            or (not isinstance(digests.get("slots"), list))
            or (len(digests["slots"]) != context_count)
        ):
            raise ValueError("Invalid R16 proposal Context slot digests")

        def valid_sha256(item):
            return bool(
                isinstance(item, str)
                and len(item) == 64
                and all((character in "0123456789abcdef" for character in item))
            )

        if not valid_sha256(digests.get("memory_sha256")):
            raise ValueError("Invalid R16 proposal Context memory digest")
        slot_keys = {
            "context_id",
            "count",
            "prototype_sha256",
            "theta_sha256",
            "full_slot_sha256",
        }
        for expected_context_id, slot in enumerate(digests["slots"]):
            if (
                not isinstance(slot, dict)
                or set(slot) != slot_keys
                or isinstance(slot.get("context_id"), (bool, np.bool_))
                or (not isinstance(slot.get("context_id"), (int, np.integer)))
                or (int(slot["context_id"]) != expected_context_id)
                or isinstance(slot.get("count"), (bool, np.bool_))
                or (not isinstance(slot.get("count"), (int, np.integer)))
                or (int(slot["count"]) < 0)
                or (not valid_sha256(slot.get("prototype_sha256")))
                or (not valid_sha256(slot.get("theta_sha256")))
                or (not valid_sha256(slot.get("full_slot_sha256")))
            ):
                raise ValueError("Invalid R16 proposal Context slot digest")
        if not valid_sha256(value.get("state_sha256")):
            raise ValueError("Invalid R16 proposal Context-state digest")
        unsigned = {key: value[key] for key in value if key != "state_sha256"}
        if canonical_json_sha256(unsigned) != value["state_sha256"]:
            raise ValueError("R16 proposal Context-state digest mismatch")
        return deepcopy(value)

    def _m3wm_clone_identity_proposal_rollback(self, snapshot, device=None):
        required = {
            "proposal_step",
            "active_context_id",
            "active_context_origin",
            "raw_theta",
            "smoothed_theta",
            "last_theta",
            "last_memory_theta",
            "last_router_proposal",
            "last_router_blend",
            "theta_version",
            "online_adaptation_history",
            "online_adaptation_history_schema",
        }
        optional = {
            "proposal_context_state",
            "online_theta_transaction_state",
            "online_theta_clean_episode_updates",
        }
        if (
            not isinstance(snapshot, dict)
            or not required.issubset(snapshot)
            or (not set(snapshot).issubset(required | optional))
        ):
            raise ValueError("Invalid P2 proposal rollback keys")
        if snapshot["online_adaptation_history_schema"] != {
            "schema": "m3wm_raw_transition_history",
            "version": 1,
        }:
            raise ValueError("Invalid P2 rollback adaptation-history schema")
        raw_proposal_step = snapshot["proposal_step"]
        if not isinstance(raw_proposal_step, (int, np.integer)) or isinstance(
            raw_proposal_step, (bool, np.bool_)
        ):
            raise ValueError("Invalid P2 rollback proposal step")
        proposal_step = int(raw_proposal_step)
        if proposal_step < 1:
            raise ValueError("Invalid P2 rollback proposal step")
        active_context_id = snapshot["active_context_id"]
        if active_context_id is not None:
            if not isinstance(active_context_id, (int, np.integer)) or isinstance(
                active_context_id, (bool, np.bool_)
            ):
                raise ValueError("Invalid P2 rollback active Context")
            active_context_id = int(active_context_id)
            if active_context_id < 0:
                raise ValueError("Invalid P2 rollback active Context")
        if (
            not isinstance(snapshot["active_context_origin"], str)
            or not snapshot["active_context_origin"]
        ):
            raise ValueError("Invalid P2 rollback Context origin")
        expected_theta_numel = int(self.model.m3wm_num_primitives) * (
            int(self.model.m3wm_num_composition_branches)
            if bool(self.model.m3wm_separate_composition_heads)
            else 1
        )
        expected_theta_shape = (
            (
                int(self.model.m3wm_num_composition_branches),
                int(self.model.m3wm_num_primitives),
            )
            if bool(self.model.m3wm_separate_composition_heads)
            else (int(self.model.m3wm_num_primitives),)
        )

        def clone_tensor(name):
            value = snapshot[name]
            if value is None:
                return None
            tensor = torch.as_tensor(value).detach().clone()
            if not torch.is_floating_point(tensor):
                raise ValueError(f"Invalid P2 rollback theta dtype: {name}")
            if tensor.numel() and (not bool(torch.isfinite(tensor).all())):
                raise ValueError(f"Non-finite P2 rollback tensor: {name}")
            if (
                tensor.numel() != expected_theta_numel
                or tuple(tensor.shape) != expected_theta_shape
            ):
                raise ValueError(f"Invalid P2 rollback theta shape: {name}")
            return tensor if device is None else tensor.to(device)

        blend = snapshot["last_router_blend"]
        if blend is not None:
            if isinstance(blend, (bool, np.bool_)):
                raise ValueError("Invalid P2 rollback Router blend")
            blend = float(blend)
            if not np.isfinite(blend):
                raise ValueError("Invalid P2 rollback Router blend")
        raw_theta_version = snapshot["theta_version"]
        if not isinstance(raw_theta_version, (int, np.integer)) or isinstance(
            raw_theta_version, (bool, np.bool_)
        ):
            raise ValueError("Invalid P2 rollback theta version")
        theta_version = int(raw_theta_version)
        adaptation_history = self._m3wm_clone_adaptation_history(
            snapshot["online_adaptation_history"], device=device
        )
        if adaptation_history and int(adaptation_history[-1][0]) >= proposal_step:
            raise ValueError("P2 rollback raw history is not proposal-edge causal")

        def configured_dimension(name):
            try:
                raw = self.cfg.get(name, None)
            except (AttributeError, KeyError, TypeError, ValueError):
                raw = None
            if raw is None or str(raw) == "???":
                return None
            if (
                isinstance(raw, (bool, np.bool_))
                or not isinstance(raw, (int, np.integer))
                or int(raw) < 1
            ):
                raise ValueError(f"Invalid configured rollback dimension: {name}")
            return int(raw)

        expected_obs_dim = None
        try:
            obs_shape = self.cfg.get("obs_shape", None)
            if isinstance(obs_shape, dict) and obs_shape:
                shape = next(iter(obs_shape.values()))
                expected_obs_dim = int(np.prod(shape))
        except (TypeError, ValueError, OverflowError):
            expected_obs_dim = None
        expected_action_dim = configured_dimension("action_dim")
        strict_history_schema = bool(
            getattr(self, "_m3wm_rollback_theta_on_terminal_dismiss", False)
        )
        if (
            strict_history_schema
            and adaptation_history
            and (expected_obs_dim is None or expected_action_dim is None)
        ):
            raise ValueError("R8 rollback raw-history dimensions are not configured")
        for _step, obs, action, reward, next_obs, _task in (
            adaptation_history if strict_history_schema else ()
        ):
            if (
                obs.ndim != 1
                or action.ndim != 1
                or reward.shape != (1,)
                or (next_obs.ndim != 1)
            ):
                raise ValueError("Invalid P2 rollback raw-history tensor schema")
            if expected_obs_dim is not None and (
                obs.numel() != expected_obs_dim or next_obs.numel() != expected_obs_dim
            ):
                raise ValueError("Invalid P2 rollback raw-history observation width")
            if (
                expected_action_dim is not None
                and action.numel() != expected_action_dim
            ):
                raise ValueError("Invalid P2 rollback adaptation-history action width")
        history_maxlen = getattr(
            getattr(self, "_m3wm_online_adaptation_history", None), "maxlen", None
        )
        if history_maxlen is not None and len(adaptation_history) > history_maxlen:
            raise ValueError("P2 rollback adaptation history exceeds buffer capacity")
        cloned = {
            "proposal_step": proposal_step,
            "active_context_id": active_context_id,
            "active_context_origin": snapshot["active_context_origin"],
            "raw_theta": clone_tensor("raw_theta"),
            "smoothed_theta": clone_tensor("smoothed_theta"),
            "last_theta": clone_tensor("last_theta"),
            "last_memory_theta": clone_tensor("last_memory_theta"),
            "last_router_proposal": clone_tensor("last_router_proposal"),
            "last_router_blend": blend,
            "theta_version": theta_version,
            "online_adaptation_history": adaptation_history,
            "online_adaptation_history_schema": {
                "schema": "m3wm_raw_transition_history",
                "version": 1,
            },
        }
        if "online_theta_transaction_state" in snapshot:
            cloned["online_theta_transaction_state"] = (
                self._m3wm_validated_online_theta_transaction_state(
                    snapshot["online_theta_transaction_state"]
                )
            )
        if "online_theta_clean_episode_updates" in snapshot:
            raw_clean_updates = snapshot["online_theta_clean_episode_updates"]
            clean_limit = self._m3wm_online_theta_clean_episode_update_limit_value()
            if (
                isinstance(raw_clean_updates, (bool, np.bool_))
                or not isinstance(raw_clean_updates, (int, np.integer))
                or int(raw_clean_updates) < 0
                or (clean_limit == 0 and int(raw_clean_updates) != 0)
                or (clean_limit > 0 and int(raw_clean_updates) > clean_limit)
            ):
                raise ValueError("Invalid rollback clean theta-update count")
            cloned["online_theta_clean_episode_updates"] = int(raw_clean_updates)
        if "proposal_context_state" in snapshot:
            try:
                cloned["proposal_context_state"] = (
                    self._m3wm_clone_active_consensus_context_transaction_state(
                        snapshot["proposal_context_state"], proposal_step=proposal_step
                    )
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        return cloned

    def _m3wm_restore_identity_proposal_rollback(self, proposal_step, snapshot=None):
        if snapshot is None:
            snapshot = self._m3wm_identity_proposal_rollback
        if snapshot is None:
            return False
        try:
            restored = self._m3wm_clone_identity_proposal_rollback(
                snapshot, device=self.device
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
            return False
        if (
            not isinstance(proposal_step, (int, np.integer))
            or isinstance(proposal_step, (bool, np.bool_))
            or int(restored["proposal_step"]) != int(proposal_step)
        ):
            return False
        active_context_id = restored["active_context_id"]
        if active_context_id is not None:
            return False
        self._m3wm_active_context_id = active_context_id
        self._m3wm_active_context_origin = restored["active_context_origin"]
        self._m3wm_raw_theta = restored["raw_theta"]
        self._m3wm_smoothed_theta = restored["smoothed_theta"]
        self._m3wm_last_theta = restored["last_theta"]
        self._m3wm_last_memory_theta = restored["last_memory_theta"]
        self._m3wm_last_router_proposal = restored["last_router_proposal"]
        self._m3wm_last_router_blend = (
            float("nan")
            if restored["last_router_blend"] is None
            else float(restored["last_router_blend"])
        )
        self._m3wm_online_adaptation_history.clear()
        self._m3wm_online_adaptation_history.extend(
            restored["online_adaptation_history"]
        )
        self._m3wm_install_online_theta_transaction_state(
            restored.get("online_theta_transaction_state")
        )
        if "online_theta_clean_episode_updates" in restored:
            self._m3wm_online_theta_clean_episode_updates = int(
                restored["online_theta_clean_episode_updates"]
            )
        self._m3wm_theta_version = self._m3wm_context_version
        self.model.set_runtime_theta(self._m3wm_last_theta)
        return True

    def _m3wm_discard_identity_proposal_rollback(self):
        self._m3wm_identity_proposal_rollback = None
        self._m3wm_context_identity_model_snapshot = None

    def _m3wm_v65_overlay_owns_terminal_dismiss(self, pending):
        try:
            overlay = self._m3wm_v65_provisional_overlay
            bank = self._m3wm_context_exemplar_bank
            candidate = bank.read_provisional_novel_candidate()
            audit = bank.provisional_novel_candidate_audit()
            proposal_step = int(pending["proposal_step"])
            return bool(
                self._m3wm_v65_fast2o_authority_enabled
                and isinstance(overlay, dict)
                and (overlay.get("valid") is True)
                and (overlay.get("state") == "PROVISIONAL_ACTIVE")
                and (overlay.get("proposal_step") == proposal_step)
                and (overlay.get("generation_id") == audit.get("generation_id"))
                and isinstance(candidate, dict)
                and (
                    overlay.get("candidate_payload_sha256")
                    == candidate.get("payload_sha256")
                )
                and (
                    overlay.get("source_context_id")
                    == candidate.get("active_context_id")
                )
                and (int(None) == int(overlay["source_context_id"]))
                and (self._m3wm_active_context_origin == "v65_provisional_overlay")
                and torch.is_tensor(overlay.get("theta"))
                and torch.is_tensor(overlay.get("source_theta"))
            )
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return False

    def _m3wm_prepare_terminal_dismiss_theta_rollback(
        self, pending, *, decision_requires_rollback=False
    ):
        transaction_budget_restore = bool(
            self._m3wm_online_theta_transaction_update_limit_value() > 0
            and getattr(self, "_m3wm_online_theta_transaction_proposal_step", None)
            is not None
        )
        should_restore = bool(
            decision_requires_rollback
            or getattr(self, "_m3wm_rollback_theta_on_terminal_dismiss", False)
            or transaction_budget_restore
        )
        if not should_restore:
            return None
        False
        if not isinstance(pending, dict):
            raise RuntimeError(
                "P2 terminal-dismiss rollback has no bound pending transaction"
            )
        if self._m3wm_v65_overlay_owns_terminal_dismiss(pending):
            return None
        try:
            raw_proposal_step = pending["proposal_step"]
            if not isinstance(raw_proposal_step, (int, np.integer)) or isinstance(
                raw_proposal_step, (bool, np.bool_)
            ):
                raise ValueError("invalid pending proposal step")
            proposal_step = int(raw_proposal_step)
            if proposal_step < 1:
                raise ValueError("invalid pending proposal step")
            snapshot = self._m3wm_clone_identity_proposal_rollback(
                self._m3wm_identity_proposal_rollback, device=self.device
            )
            current_history = self._m3wm_clone_adaptation_history(
                self._m3wm_online_adaptation_history, device=self.device
            )
            if (
                snapshot["online_adaptation_history"]
                and current_history
                and (
                    tuple(
                        (
                            tuple(value.shape)
                            for value in snapshot["online_adaptation_history"][0][1:5]
                        )
                    )
                    != tuple((tuple(value.shape) for value in current_history[0][1:5]))
                    or (snapshot["online_adaptation_history"][0][5] is None)
                    != (current_history[0][5] is None)
                )
            ):
                raise ValueError(
                    "rollback adaptation-history schema changed while pending"
                )
        except (KeyError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
            raise RuntimeError(
                "P2 terminal-dismiss rollback snapshot is malformed"
            ) from exc
        if getattr(self, "_m3wm_rollback_theta_on_terminal_dismiss", False) and any(
            (
                snapshot[name] is None
                for name in (
                    "raw_theta",
                    "smoothed_theta",
                    "last_theta",
                    "last_memory_theta",
                )
            )
        ):
            raise RuntimeError(
                "P2 terminal-dismiss rollback snapshot has missing control theta"
            )
        if int(snapshot["proposal_step"]) != proposal_step:
            raise RuntimeError(
                "P2 terminal-dismiss rollback is bound to another proposal"
            )
        if (
            snapshot.get("active_context_id") != None
            or snapshot.get("active_context_origin") != self._m3wm_active_context_origin
        ):
            raise RuntimeError(
                "P2 terminal-dismiss rollback active Context drifted while pending"
            )
        return (proposal_step, snapshot)

    def _system_id_norm_feature(self, samples, window):
        window = max(2, int(window))
        if len(samples) < window:
            return None
        selected = samples[-window:]
        conditioned = len(selected[0]) == 3
        if conditioned:
            states = torch.stack(
                [state for (state, _action, _delta) in selected], dim=0
            ).float()
            actions = torch.stack(
                [action for (_state, action, _delta) in selected], dim=0
            ).float()
            deltas = torch.stack(
                [delta for (_state, _action, delta) in selected], dim=0
            ).float()
        else:
            actions = torch.stack([sample[-2] for sample in selected], dim=0).float()
            deltas = torch.stack([sample[-1] for sample in selected], dim=0).float()
        if actions.ndim != 2 or deltas.ndim != 2:
            raise ValueError("system_id_norm requires vector observations and actions")
        ridge = float(self.cfg.get("m3wm_system_id_ridge", 0.01))
        if conditioned:
            return conditioned_action_response_log_norm(
                states,
                actions,
                deltas,
                ridge=ridge,
                min_std=float(self.cfg.get("m3wm_system_id_min_std", 0.001)),
            ).reshape(1)
        return linear_action_response_log_norm(actions, deltas, ridge=ridge).reshape(1)

    @staticmethod
    def _clone_optional_tensor(value):
        return None if value is None else value.detach().clone()

    @torch.no_grad()
    def observe_transition(
        self, obs, action, reward, next_obs, task=None, terminal=False
    ):
        if not self.cfg.get("enable_m3wm", False):
            return
        self._m3wm_online_theta_last_update_owner = "not_called"
        self._m3wm_online_theta_last_adapt_eligible = False
        self._m3wm_capture_pending_runtime_theta(int(self._m3wm_runtime_step) + 1)
        self._m3wm_last_confirmed_event = None
        context_runtime_enabled = self._m3wm_context_runtime_enabled()
        pending_transaction_this_step = self._m3wm_has_pending_change()
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        next_obs = next_obs.to(self.device, non_blocking=True).unsqueeze(0)
        action = torch.as_tensor(action, device=self.device, dtype=obs.dtype).reshape(
            1, -1
        )
        reward = torch.as_tensor(reward, device=self.device, dtype=obs.dtype).reshape(
            1, 1
        )
        task_tensor = task
        if task is not None and (not torch.is_tensor(task)):
            task_tensor = torch.tensor([task], device=self.device)
        prepacked_raw_sample = self._m3wm_pack_raw_transition(
            int(self._m3wm_runtime_step) + 1, obs, action, reward, next_obs, task_tensor
        )
        self._m3wm_append_detector_raw_transition(
            obs,
            action,
            reward,
            next_obs,
            task=task_tensor,
            prepacked_sample=prepacked_raw_sample,
        )
        raw_sample = self._m3wm_append_context_raw_transition(
            obs,
            action,
            reward,
            next_obs,
            task=task_tensor,
            prepacked_sample=prepacked_raw_sample,
        )
        if raw_sample is None:
            raise RuntimeError(
                "ReCo-WM rejected a non-finite or discontinuous physical transition"
            )
        exemplar_bank = getattr(self, "_m3wm_context_exemplar_bank", None)
        if exemplar_bank is not None:
            if not self._m3wm_has_pending_change():
                exemplar_bank.restart_after_resolution_if_needed(
                    guard_steps=max(1, int(self.cfg.get("m3wm_change_long_window", 64)))
                )
            exemplar_bank.observe(
                raw_sample,
                context_id=None,
                episode_start_step=self._m3wm_detector_episode_start_step,
            )
        with torch.no_grad():
            feature = (
                self.model.build_context_features_from_observations(
                    obs, action, reward, next_obs, task=task_tensor
                )
                .squeeze(0)
                .detach()
            )
        self._m3wm_append_context_evidence_stream(
            raw_sample, feature, terminal=bool(terminal)
        )
        self._m3wm_context_history.append(feature)
        self._m3wm_change_history.append(feature)
        self._m3wm_current_episode_change_history.append(feature)
        self._m3wm_diag_observe_context_feature(feature)
        system_id_sample = (
            obs.squeeze(0).detach(),
            action.squeeze(0).detach(),
            (next_obs - obs).squeeze(0).detach(),
        )
        self._m3wm_current_episode_system_id.append(system_id_sample)
        self._m3wm_local_system_id_history.append(system_id_sample)
        self._m3wm_last_context_feature = feature
        self._m3wm_context_version += 1
        self._m3wm_runtime_step += 1
        initial_step_calibration = (
            context_runtime_enabled and self._m3wm_initial_null_steps_remaining > 0
        )
        self._m3wm_reward_identity_observe_context_feature(
            feature, initial_null_step=initial_step_calibration
        )
        if initial_step_calibration:
            self._m3wm_initial_null_steps_remaining -= 1
        self._m3wm_soft_reuse_applied_this_step = False
        self._m3wm_last_soft_reuse_receipt = {
            "active": False,
            "applied": False,
            "reason": "not_evaluated",
        }
        diagnostic_transaction_outcome = None
        diagnostic_transaction_reason = None
        prebootstrap_unbound_proposal = (False, False)[1]
        False
        confirmation_delay = max(
            0, int(self.cfg.get("m3wm_trigger_confirmation_delay", 0))
        )
        confirmation_delay = max(
            0, int(self.cfg.get("m3wm_trigger_confirmation_delay", 0))
        )
        initial_context_bootstrap_due = self._m3wm_initial_context_bootstrap_due(
            initial_step_calibration
        )
        continuous_inference_interval = max(
            1,
            int(
                self.cfg.get(
                    "m3wm_continuous_inference_interval",
                    self.cfg.get("m3wm_memory_window", 32),
                )
            ),
        )
        continuous_memory_inference = False
        pending_transaction_this_step = bool(
            pending_transaction_this_step or self._m3wm_has_pending_change()
        )
        diagnostic_finalize_due = bool(
            self._m3wm_diagnostics_enabled()
            and (not getattr(self, "_m3wm_diagnostic_suppressed_pending", False))
            and (diagnostic_transaction_outcome is not None)
        )
        identity_mutation_intended = bool(
            diagnostic_transaction_outcome in {"create", "recall"}
        )
        diagnostic_active_before_commit = (
            None if diagnostic_finalize_due or identity_mutation_intended else None
        )
        diagnostic_context_count_before_commit = (
            0 if diagnostic_finalize_due or identity_mutation_intended else 0
        )
        soft_reuse_active_for_step = bool(
            self._m3wm_soft_context_reuse is not None
            and self._m3wm_soft_context_reuse.active
        )
        reuse_execution_active_for_step = bool(soft_reuse_active_for_step)
        self._activate_runtime_theta(
            force=continuous_memory_inference or initial_context_bootstrap_due,
            re_infer=(continuous_memory_inference or initial_context_bootstrap_due)
            and (not reuse_execution_active_for_step),
            suppress_background_memory_updates=pending_transaction_this_step,
        )
        self._m3wm_soft_reuse_applied_this_step = bool(reuse_execution_active_for_step)
        self._m3wm_last_soft_reuse_receipt = {
            "active": bool(reuse_execution_active_for_step),
            "applied": False,
            "reason": (
                "pending_post_adapt_blend"
                if reuse_execution_active_for_step
                else "inactive"
            ),
        }
        initial_context_bootstrap_committed = (False, False)[1]
        context_changed_after_activation = bool(
            diagnostic_active_before_commit is not None
            and None != diagnostic_active_before_commit
        )
        identity_intent_not_committed = bool(
            identity_mutation_intended
            and (not reuse_execution_active_for_step)
            and (0 <= diagnostic_context_count_before_commit)
            and (not context_changed_after_activation)
        )
        memory_confirmation_failed = (False, False)[1]
        terminal_identity_commit_failed = bool(
            identity_intent_not_committed
            and getattr(self, "_m3wm_rollback_theta_on_terminal_dismiss", False)
        )
        if memory_confirmation_failed or terminal_identity_commit_failed:
            if terminal_identity_commit_failed:
                raise RuntimeError("Uncommitted identity intent lost its rollback plan")
            confirmed_trigger = False
            self._m3wm_rejected_change_count += 1
            self._m3wm_close_multiscale_terminal_decision(
                accepted=False, confirmation_delay=confirmation_delay
            )
        elif diagnostic_transaction_outcome == "dismiss":
            self._m3wm_close_multiscale_terminal_decision(
                accepted=False, confirmation_delay=confirmation_delay
            )
        stable_update_interval = max(
            0, int(self.cfg.get("m3wm_memory_stable_update_interval", 0))
        )
        stable_step_observe_due = False
        stable_step_observe_due and (
            pending_transaction_this_step or self._m3wm_has_pending_change()
        )
        self._m3wm_last_change_triggered = False
        self._m3wm_maybe_arm_post_terminal_writeback_guard(
            pending_transaction_this_step
        )
        diagnostic_theta_capture_due = bool(
            self._m3wm_diagnostics_enabled()
            and (not getattr(self, "_m3wm_diagnostic_suppressed_pending", False))
            and (diagnostic_transaction_outcome is not None)
        )
        pre_adapt_provisional = (
            self._clone_optional_tensor(self._m3wm_raw_theta)
            if diagnostic_theta_capture_due
            else None
        )
        pre_adapt_control = (
            self._clone_optional_tensor(self._m3wm_last_theta)
            if diagnostic_theta_capture_due
            else None
        )
        self._online_adapt_runtime_theta(
            obs,
            action,
            reward,
            next_obs,
            task_tensor,
            suppress_memory_writeback=pending_transaction_this_step,
        )
        self._m3wm_last_soft_reuse_receipt = self._m3wm_apply_soft_context_reuse_theta()
        if (
            isinstance(self._m3wm_last_soft_reuse_receipt, dict)
            and self._m3wm_last_soft_reuse_receipt.get("committed_target_context_id")
            is not None
        ):
            self._m3wm_context_recalls += 1
        False
        last_exemplar_assignment = getattr(
            self, "_m3wm_last_context_exemplar_assignment", None
        )
        v66_authority = (
            last_exemplar_assignment.get("v66_faci_authority")
            if isinstance(last_exemplar_assignment, dict)
            else None
        )
        transaction_terminal_due = bool(
            pending_transaction_this_step
            and diagnostic_transaction_outcome is not None
            and (not self._m3wm_has_pending_change())
        )
        if transaction_terminal_due:
            transaction_context_committed = False
            self._m3wm_end_online_theta_transaction(context_committed=False)
            self._m3wm_objective_finalize(context_committed=False)
        self._m3wm_observe_episodic_theta_baseline_first(reward)
        self._m3wm_observe_episodic_theta_reward_guard(reward)
        if diagnostic_finalize_due:
            context_count_after_commit = 0
            context_changed = (
                diagnostic_active_before_commit is not None
                and None != diagnostic_active_before_commit
            )
            intended_outcome = diagnostic_transaction_outcome
            if intended_outcome in {"create", "recall", "confirmed"}:
                if 0 > diagnostic_context_count_before_commit:
                    diagnostic_transaction_outcome = "create"
                elif context_changed:
                    diagnostic_transaction_outcome = "recall"
                else:
                    diagnostic_transaction_outcome = "dismiss"
                    diagnostic_transaction_reason = (
                        f"{diagnostic_transaction_reason}_not_committed"
                    )
            self._m3wm_diag_finalize_transaction(
                diagnostic_transaction_outcome,
                diagnostic_transaction_reason,
                pre_adapt_provisional,
                pre_adapt_control,
                terminal_dismiss_rollback_requested=False,
                terminal_dismiss_rollback_applied=False,
                terminal_dismiss_rollback_proposal_step=None,
            )
        self._m3wm_update_empirical_change_null(
            score=self._m3wm_last_change_statistic,
            components=None,
            initial_step_calibration=initial_step_calibration,
            pending_before=pending_transaction_this_step,
            candidate_step=False,
            transaction_outcome=diagnostic_transaction_outcome,
            confirmed_trigger=False,
            terminal=bool(terminal),
            trial_started=False,
            bootstrap_committed=initial_context_bootstrap_committed,
        )
        if getattr(self, "_m3wm_diagnostic_suppressed_pending", False) and (
            not self._m3wm_pending_change_tracker.active
        ):
            self._m3wm_diagnostic_suppressed_pending = False

    def _online_theta_loss_coefficients(self):
        reward_coef = float(self.cfg.get("m3wm_online_reward_coef", 0.1))
        value_coef = float(self.cfg.get("m3wm_online_value_coef", 0.1))
        policy = str(self.cfg.get("m3wm_online_branch_policy", "fixed")).lower()
        if policy == "fixed":
            return (reward_coef, value_coef)
        if policy != "novel_full_recall_gated":
            raise ValueError(f"Unknown m3wm_online_branch_policy: {policy!r}")
        if self._m3wm_active_context_origin == "recalled":
            reward_coef = float(self.cfg.get("m3wm_recall_reward_coef", 0.0))
            value_coef = float(self.cfg.get("m3wm_recall_value_coef", 0.0))
        return (reward_coef, value_coef)

    def _m3wm_maybe_writeback_online_theta(
        self, theta, *, suppress_memory_writeback=False
    ):
        writeback_interval = max(
            0, int(self.cfg.get("m3wm_online_theta_memory_writeback_interval", 32))
        )
        writeback_require_unlatched = getattr(
            self, "_m3wm_online_theta_writeback_require_unlatched", True
        )
        if not isinstance(writeback_require_unlatched, (bool, np.bool_)):
            raise ValueError(
                "m3wm_online_theta_writeback_require_unlatched must be boolean"
            )
        post_terminal_latch_release = self._m3wm_post_terminal_latch_release_active()
        return "not_due"

    def _online_adapt_runtime_theta(
        self, obs, action, reward, next_obs, task, suppress_memory_writeback=False
    ):
        self._m3wm_online_theta_last_update_owner = "ineligible"
        base_eligible = bool(
            self._m3wm_context_runtime_enabled()
            and self._m3wm_specialization_active
            and (self._m3wm_last_theta is not None)
        )
        reward_only_frozen = (False, False)[1]
        self._m3wm_online_theta_last_adapt_eligible = bool(
            base_eligible and (not reward_only_frozen)
        )
        if not base_eligible:
            return self._m3wm_last_theta
        clean_episode_limit = self._m3wm_online_theta_clean_episode_update_limit_value()
        transaction_owned = bool(
            suppress_memory_writeback
            or self._m3wm_has_pending_change()
            or isinstance(getattr(self, "_m3wm_v65_provisional_overlay", None), dict)
            or getattr(self, "_m3wm_last_context_changed", False)
        )
        transaction_limit = self._m3wm_online_theta_transaction_update_limit_value()
        transaction_updates = (
            self._m3wm_online_theta_transaction_update_count_value()
            if transaction_limit > 0
            else 0
        )
        transaction_proposal_step = getattr(
            self, "_m3wm_online_theta_transaction_proposal_step", None
        )
        transaction_budget_owned = bool(
            transaction_owned
            and transaction_limit > 0
            and (transaction_proposal_step is not None)
        )
        if transaction_budget_owned and self._m3wm_has_pending_change():
            pending = self._m3wm_pending_change_tracker.get()
            if not isinstance(pending, dict) or int(
                pending.get("proposal_step", -1)
            ) != int(transaction_proposal_step):
                raise RuntimeError(
                    "Online-theta transaction budget lost its proposal binding"
                )

        def append_current_raw_sample():
            step = int(self._m3wm_runtime_step)
            sample = self._m3wm_pack_raw_transition(
                step, obs, action, reward, next_obs, task
            )
            if sample is None:
                raise RuntimeError(
                    "Online theta received an invalid physical transition"
                )
            history = self._m3wm_online_adaptation_history
            if history and int(history[-1][0]) >= step:
                if int(history[-1][0]) == step:
                    history[-1] = sample
                    return sample
                history.clear()
            history.append(sample)
            return sample

        terminal_context_hold = getattr(
            self, "_m3wm_online_theta_terminal_context_hold", False
        )
        if not isinstance(terminal_context_hold, (bool, np.bool_)):
            raise ValueError("Invalid ReCo-WM terminal Context theta hold")
        budget_hold_owner = None
        if bool(terminal_context_hold) and (not transaction_owned):
            budget_hold_owner = "terminal_context_hold"
        elif transaction_budget_owned and transaction_updates >= transaction_limit:
            budget_hold_owner = "transaction_budget_hold"
        if budget_hold_owner is not None:
            append_current_raw_sample()
            adapted = self._m3wm_last_theta.detach().clone()
            self._m3wm_raw_theta = adapted
            self._m3wm_smoothed_theta = adapted
            self._m3wm_last_theta = adapted
            self.model.set_runtime_theta(adapted)
            self._m3wm_maybe_writeback_online_theta(
                adapted,
                suppress_memory_writeback=suppress_memory_writeback
                or budget_hold_owner == "terminal_context_hold",
            )
            self._m3wm_online_theta_last_update_owner = budget_hold_owner
            return adapted
        baseline_first_entry = None if transaction_owned else None
        if baseline_first_entry is not None:
            baseline_theta = baseline_first_entry["baseline_theta"].to(self.device)
            if self._m3wm_last_theta is None or not torch.equal(
                self._m3wm_last_theta, baseline_theta
            ):
                if not self._m3wm_install_episodic_theta_guard_value(
                    int(None), baseline_theta
                ):
                    raise RuntimeError("Cannot enforce active baseline-first theta")
            else:
                self.model.set_runtime_theta(self._m3wm_last_theta)
            append_current_raw_sample()
            self._m3wm_online_theta_last_update_owner = "baseline_first_hold"
            return self._m3wm_last_theta
        clean_episode_updates = (
            self._m3wm_online_theta_clean_episode_update_count_value()
            if clean_episode_limit > 0
            else 0
        )
        if (
            clean_episode_limit > 0
            and (not transaction_owned)
            and (clean_episode_updates >= clean_episode_limit)
        ):
            self._m3wm_online_theta_last_update_owner = "clean_budget_hold"
            append_current_raw_sample()
            adapted = self._m3wm_last_theta.detach().clone()
            self._m3wm_raw_theta = adapted
            self._m3wm_smoothed_theta = adapted
            self._m3wm_last_theta = adapted
            self.model.set_runtime_theta(adapted)
            self._m3wm_maybe_writeback_online_theta(
                adapted, suppress_memory_writeback=suppress_memory_writeback
            )
            return adapted
        steps = max(1, int(self.cfg.get("m3wm_online_theta_steps", 1)))
        lr = float(self.cfg.get("m3wm_online_theta_lr", 0.05))
        reward_coef, value_coef = self._online_theta_loss_coefficients()
        if lr <= 0 or reward_coef < 0 or value_coef < 0:
            raise ValueError("Invalid online theta adaptation coefficients")
        theta_trust_anchor = None
        trust_enabled, _trust_radius = (
            self._m3wm_branchwise_theta_trust_region_settings()
        )
        if trust_enabled:
            pending_live = self._m3wm_has_pending_change()
            terminal_transaction_exempt = bool(
                suppress_memory_writeback and (not pending_live)
            )
            identity_commit_exempt = bool(
                getattr(self, "_m3wm_last_context_changed", False)
            )
            if not terminal_transaction_exempt and (not identity_commit_exempt):
                if pending_live:
                    theta_trust_anchor = self._m3wm_pending_theta_trust_region_anchor(
                        self._m3wm_last_theta
                    )
                else:
                    theta_trust_anchor = self._m3wm_last_theta.detach().clone()
        append_current_raw_sample()
        current_batch = self._m3wm_current_version_latent_batch(
            list(self._m3wm_online_adaptation_history)
        )
        if current_batch is None:
            raise RuntimeError(
                "Online theta could not rebuild its raw adaptation window"
            )
        z, action, reward, next_z, task = current_batch
        with torch.enable_grad():
            logits = self._m3wm_last_theta.clamp_min(1e-08).log()
            logits = logits.detach().requires_grad_(True)
            projected_candidate = None
            for _ in range(steps):
                theta = torch.softmax(logits, dim=-1)
                predicted_next = self.model.next(z, action, task, theta=theta)
                reward_logits = self.model.reward(z, action, task, theta=theta)
                predicted_reward = math.two_hot_inv(reward_logits, self.cfg)
                if predicted_reward.ndim == reward.ndim - 1:
                    predicted_reward = predicted_reward.unsqueeze(-1)
                loss = F.mse_loss(predicted_next, next_z)
                loss = loss + reward_coef * F.mse_loss(predicted_reward, reward)
                if value_coef > 0 and self.model.m3wm_value_uses_theta:
                    value_logits = self.model.Q(
                        z, action, task, return_type="all", theta=theta
                    )
                    predicted_value = math.two_hot_inv(value_logits, self.cfg).mean(
                        dim=0
                    )
                    with torch.no_grad():
                        next_action, _ = self.model.pi(next_z, task)
                        discount = self.discount
                        target_value = reward + discount * self.model.Q(
                            next_z,
                            next_action,
                            task,
                            return_type="min",
                            target=True,
                            theta=theta.detach(),
                        )
                    loss = loss + value_coef * F.mse_loss(predicted_value, target_value)
                gradient = torch.autograd.grad(loss, logits)[0]
                candidate_logits = logits - lr * gradient
                if theta_trust_anchor is None:
                    logits = candidate_logits.detach().requires_grad_(True)
                else:
                    projected_candidate = (
                        self._m3wm_apply_branchwise_theta_trust_region(
                            torch.softmax(candidate_logits.detach(), dim=-1),
                            theta_trust_anchor,
                            source=(
                                "pending_online_adapt"
                                if pending_live
                                else "clean_online_adapt"
                            ),
                        )
                    )
                    logits = projected_candidate.clamp_min(1e-08).log()
                    logits = logits.detach().requires_grad_(True)
            adapted = (
                torch.softmax(logits.detach(), dim=-1)
                if theta_trust_anchor is None
                else projected_candidate
            )
        self._m3wm_raw_theta = adapted
        self._m3wm_smoothed_theta = adapted
        self._m3wm_last_theta = adapted
        self.model.set_runtime_theta(adapted)
        provisional_overlay = getattr(self, "_m3wm_v65_provisional_overlay", None)
        if (
            isinstance(provisional_overlay, dict)
            and provisional_overlay.get("state") == "PROVISIONAL_ACTIVE"
        ):
            provisional_overlay = deepcopy(provisional_overlay)
            provisional_overlay["theta"] = adapted.detach().clone()
            self._m3wm_v65_provisional_overlay = provisional_overlay
        self._m3wm_maybe_writeback_online_theta(
            adapted, suppress_memory_writeback=suppress_memory_writeback
        )
        if transaction_budget_owned and transaction_limit > 0:
            self._m3wm_online_theta_transaction_updates = transaction_updates + 1
        elif clean_episode_limit > 0 and (not transaction_owned):
            self._m3wm_online_theta_clean_episode_updates = clean_episode_updates + 1
        self._m3wm_online_theta_last_update_owner = (
            "transaction_gradient" if transaction_owned else "clean_gradient"
        )
        return adapted

    @torch.no_grad()
    def _m3wm_maybe_mature_held_post_create_quiet_prototype(self):
        post_create_quiet = bool(
            getattr(self, "_m3wm_episode_change_latched", False)
            and getattr(self, "_m3wm_episode_is_null_calibration", False)
            and (not getattr(self, "_m3wm_episode_is_initial_null_calibration", False))
            and (not self._m3wm_has_pending_change())
        )
        if not post_create_quiet:
            return False
        if (
            str(self.cfg.get("m3wm_context_assignment_mode", "regime_gate")).lower()
            != "dp_identity"
        ):
            return False
        return False

    def _m3wm_branchwise_theta_trust_region_settings(self):
        enabled = getattr(self, "_m3wm_branchwise_theta_trust_region_enabled", False)
        radius = getattr(
            self,
            "_m3wm_branchwise_theta_trust_region_radius",
            self.cfg.get("m3wm_branchwise_theta_trust_region_radius", 0.0),
        )
        return _validated_m3wm_branchwise_theta_trust_region(enabled, radius)

    def _m3wm_apply_branchwise_theta_trust_region(self, candidate, anchor, *, source):
        enabled, radius = self._m3wm_branchwise_theta_trust_region_settings()
        if not enabled:
            return candidate
        try:
            if not isinstance(source, str) or not source:
                raise ValueError("projection source is missing")
            return _m3wm_project_branchwise_centered_logit_trust_region(
                candidate, anchor, radius
            )
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise RuntimeError(
                f"Branchwise centered-logit trust region failed closed for {source}: {error}"
            ) from error

    def _m3wm_pending_theta_trust_region_anchor(self, reference_theta):
        enabled, _radius = self._m3wm_branchwise_theta_trust_region_settings()
        if not enabled:
            return None
        tracker = getattr(self, "_m3wm_pending_change_tracker", None)
        if tracker is None or not tracker.active:
            return None
        try:
            anchor = self._m3wm_pending_proposal_anchor_theta(reference_theta)
            if anchor is None:
                raise ValueError("live pending transaction has no fixed anchor")
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"Branchwise centered-logit trust region failed closed: {error}"
            ) from error
        return anchor

    def _m3wm_pending_proposal_anchor_theta(self, reference_theta):
        return None

    @torch.no_grad()
    def _m3wm_apply_soft_context_reuse_theta(self):
        controller = getattr(self, "_m3wm_soft_context_reuse", None)
        if controller is None or not controller.active:
            self._m3wm_soft_reuse_applied_this_step = False
            return {"active": False, "applied": False, "reason": "inactive"}
        controller.cancel("soft_reuse_target_unavailable")
        self._m3wm_soft_reuse_applied_this_step = False
        return {"active": False, "applied": False, "reason": "target_unavailable"}

    def _activate_runtime_theta(
        self, force=False, re_infer=False, suppress_background_memory_updates=False
    ):
        self._m3wm_last_context_changed = False
        if not self.cfg.get("enable_m3wm", False):
            return None
        context_runtime_enabled = self._m3wm_context_runtime_enabled()
        ordinary_action_activation = not bool(
            force or re_infer or suppress_background_memory_updates
        )
        provisional_overlay = getattr(self, "_m3wm_v65_provisional_overlay", None)
        if (
            ordinary_action_activation
            and isinstance(provisional_overlay, dict)
            and (provisional_overlay.get("state") == "PROVISIONAL_ACTIVE")
        ):
            overlay_theta = provisional_overlay.get("theta")
            if (
                not torch.is_tensor(overlay_theta)
                or not overlay_theta.is_floating_point()
                or (not bool(torch.isfinite(overlay_theta).all()))
            ):
                raise RuntimeError("Invalid V65 provisional runtime overlay")
            self._m3wm_raw_theta = overlay_theta.detach().clone()
            self._m3wm_smoothed_theta = overlay_theta.detach().clone()
            self._m3wm_last_theta = overlay_theta.detach().clone()
            self.model.set_runtime_theta(self._m3wm_last_theta)
            return self._m3wm_last_theta
        terminal_theta_hold = getattr(
            self, "_m3wm_online_theta_terminal_context_hold", False
        )
        if not isinstance(terminal_theta_hold, (bool, np.bool_)):
            raise ValueError("Invalid ReCo-WM terminal Context theta hold")
        if (
            bool(terminal_theta_hold)
            and context_runtime_enabled
            and ordinary_action_activation
            and (not self._m3wm_has_pending_change())
            and (self._m3wm_last_theta is not None)
        ):
            self.model.set_runtime_theta(self._m3wm_last_theta)
            return self._m3wm_last_theta
        baseline_first_entry = (
            None
            if context_runtime_enabled
            and ordinary_action_activation
            and (not self._m3wm_has_pending_change())
            else None
        )
        if baseline_first_entry is not None:
            context_id = int(None)
            baseline_theta = baseline_first_entry["baseline_theta"].to(self.device)
            if self._m3wm_last_theta is None or not torch.equal(
                self._m3wm_last_theta, baseline_theta
            ):
                if not self._m3wm_install_episodic_theta_guard_value(
                    context_id, baseline_theta
                ):
                    raise RuntimeError("Cannot hold active baseline-first theta")
            else:
                self.model.set_runtime_theta(self._m3wm_last_theta)
            self._m3wm_maybe_mature_held_post_create_quiet_prototype()
            return self._m3wm_last_theta
        False
        memory_updates_suppressed = bool(
            suppress_background_memory_updates or self._m3wm_has_pending_change()
        )
        if (
            not force
            and self._m3wm_last_theta is not None
            and (self._m3wm_theta_version == self._m3wm_context_version)
        ):
            self.model.set_runtime_theta(self._m3wm_last_theta)
            return self._m3wm_last_theta
        raw_history = list(self._m3wm_context_raw_history_for_runtime())
        context_window = max(1, int(self.cfg.get("m3wm_context_window", 32)))
        if not self._m3wm_specialization_active or len(raw_history) < context_window:
            reference = torch.zeros(1, self.cfg.latent_dim, device=self.device)
            raw_theta = self.model.uniform_theta(reference)[0]
        else:
            selected_raw = raw_history[-context_window:]
            features = self._m3wm_recompute_detector_raw_features(selected_raw)
            if features is None or int(features.shape[0]) != context_window:
                raise RuntimeError(
                    "Runtime Router could not rebuild its raw Context window"
                )
            features = features.unsqueeze(1)
            raw_theta = self.model.context_theta_from_features(features)[-1, 0]
        pending_theta_trust_anchor = None
        if self._m3wm_has_pending_change():
            pending_theta_trust_anchor = self._m3wm_pending_theta_trust_region_anchor(
                raw_theta
            )
        self._m3wm_last_router_proposal = raw_theta.detach()
        self._m3wm_last_memory_theta = None
        self._m3wm_last_router_blend = float("nan")
        reward_only_frozen_active = (False, False)[1]
        memory_feature_theta = raw_theta
        memory_feature, memory_ready = self._m3wm_build_memory_feature(
            re_infer, memory_feature_theta
        )
        if re_infer:
            self._m3wm_smoothed_theta = None
        assignment_mode = str(
            self.cfg.get("m3wm_context_assignment_mode", "regime_gate")
        ).lower()
        prototype_interval = max(
            1, int(self.cfg.get("m3wm_dp_prototype_continuous_interval", 100))
        )
        ema_rate = float(self.cfg.get("m3wm_dp_prototype_ema_rate", 0.02))
        post_create_quiet_maturation = bool(
            self._m3wm_episode_is_null_calibration
            and (not self._m3wm_episode_is_initial_null_calibration)
            and (not self._m3wm_has_pending_change())
        )
        alpha = float(self.cfg.get("m3wm_theta_ema_alpha", 0.1))
        if not 0 < alpha <= 1:
            raise ValueError("m3wm_theta_ema_alpha must be in (0, 1]")
        if self._m3wm_smoothed_theta is None:
            smoothed_theta = raw_theta
        else:
            smoothed_theta = (
                1.0 - alpha
            ) * self._m3wm_smoothed_theta + alpha * raw_theta
            smoothed_theta = smoothed_theta / smoothed_theta.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-08)
        if pending_theta_trust_anchor is not None:
            smoothed_theta = self._m3wm_apply_branchwise_theta_trust_region(
                smoothed_theta,
                pending_theta_trust_anchor,
                source="pending_router_smoothed_control",
            )
        planning_theta = smoothed_theta
        if pending_theta_trust_anchor is not None:
            planning_theta = self._m3wm_apply_branchwise_theta_trust_region(
                planning_theta,
                pending_theta_trust_anchor,
                source="pending_router_planning_control",
            )
        self._m3wm_raw_theta = raw_theta.detach()
        self._m3wm_smoothed_theta = smoothed_theta.detach()
        self._m3wm_last_theta = planning_theta.detach()
        self._m3wm_theta_version = self._m3wm_context_version
        self.model.set_runtime_theta(planning_theta)
        return planning_theta

    @staticmethod
    def _theta_metrics(theta, prefix):
        theta = theta.detach()
        branch_names = ("dyn", "rew", "val")
        branch_metrics = {}
        if theta.ndim == 2:
            for branch_index, branch_name in enumerate(branch_names):
                branch_metrics.update(
                    TDMPC2._theta_metrics(
                        theta[branch_index], f"{prefix}{branch_name}_"
                    )
                )
            theta = theta.mean(dim=0)
        theta_cpu = theta.float().cpu()
        theta_values = theta_cpu.tolist()
        entropy = -(theta_cpu * theta_cpu.clamp_min(1e-08).log()).sum().item()
        top_values = torch.topk(theta_cpu, k=min(2, theta_cpu.numel())).values
        margin = (
            float(top_values[0] - top_values[1])
            if top_values.numel() > 1
            else float(top_values[0])
        )
        metrics = {
            f"{prefix}theta_{i}": float(value) for (i, value) in enumerate(theta_values)
        }
        metrics.update(
            {
                f"{prefix}theta_entropy": float(entropy),
                f"{prefix}theta_max": float(theta_cpu.max().item()),
                f"{prefix}theta_margin": margin,
                f"{prefix}dominant_primitive": float(theta_cpu.argmax().item()),
            }
        )
        metrics.update(branch_metrics)
        return metrics

    def _m3wm_build_memory_feature(self, re_infer, raw_theta=None):
        memory_min_history = max(1, int(self.cfg.get("m3wm_memory_min_history", 1)))
        memory_trigger_window = max(
            1,
            int(
                self.cfg.get(
                    "m3wm_memory_trigger_window",
                    self.cfg.get(
                        "m3wm_change_short_window",
                        self.cfg.get("m3wm_context_short_window", 8),
                    ),
                )
            ),
        )
        memory_feature_mode = str(
            self.cfg.get("m3wm_memory_feature_mode", "full")
        ).lower()
        if memory_feature_mode == "system_id_norm":
            system_id_window = int(self.cfg.get("m3wm_system_id_window", 128))
            memory_feature = self._system_id_norm_feature(
                self._m3wm_current_episode_system_id, system_id_window
            )
            memory_ready = memory_feature is not None
        elif memory_feature_mode == "multiscale_system_id":
            system_id_feature = self._system_id_norm_feature(
                self._m3wm_current_episode_system_id,
                int(self.cfg.get("m3wm_system_id_window", 128)),
            )
            memory_feature = None
            memory_ready = system_id_feature is not None and (
                re_infer
                and len(self._m3wm_change_history) >= memory_trigger_window
                or len(self._m3wm_context_history) >= memory_min_history
            )
        elif memory_feature_mode == "router_theta":
            if raw_theta is None:
                memory_feature = None
                memory_ready = False
            else:
                memory_feature = raw_theta.reshape(-1)
                memory_ready = len(self._m3wm_context_history) >= memory_min_history
        elif memory_feature_mode in {
            "router_embedding",
            "router_embedding_long_term",
            "router_embedding_long_term_reward",
        }:
            memory_feature = None
            memory_ready = bool(self._m3wm_current_episode_system_id) and (
                re_infer
                and len(self._m3wm_change_history) >= memory_trigger_window
                or len(self._m3wm_context_history) >= memory_min_history
            )
        elif memory_feature_mode == "architecture_anchor_identity":
            memory_feature = None
            memory_ready = bool(
                getattr(self.model, "context_anchor_ready", False)
            ) and (
                re_infer
                and len(self._m3wm_change_history) >= memory_trigger_window
                or len(self._m3wm_context_history) >= memory_min_history
            )
        elif memory_feature_mode == _M3WM_LIVE_MEMORY_FEATURE_MODE:
            memory_feature = None
            live_raw_history = (
                self._m3wm_detector_raw_history_for_runtime()
                if re_infer
                else self._m3wm_context_raw_history_for_runtime()
            )
            memory_ready = self._m3wm_bocpd_fixed_calibrated and (
                re_infer
                and len(live_raw_history) >= memory_trigger_window
                or len(live_raw_history) >= memory_min_history
            )
        else:
            memory_feature = None
            memory_ready = (
                re_infer
                and len(self._m3wm_change_history) >= memory_trigger_window
                or len(self._m3wm_context_history) >= memory_min_history
            )
        if (
            memory_feature_mode == "router_embedding_long_term_reward"
            and memory_feature is None
        ):
            memory_ready = False
        return (memory_feature, memory_ready)

    def _m3wm_context_metrics_cache_key(self):
        try:
            pending = self._m3wm_pending_change_tracker.get() or {}
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pending = {}
        active_context = None
        active_context = -1 if active_context is None else int(active_context)
        return (
            str(self.cfg.get("m3wm_mode", "")).lower(),
            int(getattr(self, "_m3wm_runtime_step", 0)),
            int(getattr(self, "_m3wm_context_version", 0)),
            int(getattr(self, "_m3wm_train_updates", 0)),
            int(getattr(self, "_m3wm_theta_version", -1)),
            id(getattr(self, "_m3wm_last_theta", None)),
            id(getattr(self, "_m3wm_raw_theta", None)),
            id(getattr(self, "_m3wm_smoothed_theta", None)),
            id(getattr(self, "_m3wm_last_router_proposal", None)),
            id(getattr(self, "_m3wm_context_evidence_stream", None)),
            int(
                getattr(
                    getattr(self, "_m3wm_context_evidence_stream", None), "revision", 0
                )
            ),
            active_context,
            int(getattr(self, "_m3wm_context_creations", 0)),
            int(getattr(self, "_m3wm_context_recalls", 0)),
            int(getattr(self, "_m3wm_confirmed_change_count", 0)),
            bool(getattr(self, "_m3wm_last_change_triggered", False)),
            int(pending.get("proposal_step", -1) or -1),
            int(pending.get("confirmation_due_step", -1) or -1),
            str(pending.get("source", "none")),
            id(getattr(self, "_m3wm_last_confirmed_event", None)),
            id(getattr(self, "_m3wm_last_context_exemplar_assignment", None)),
            int(getattr(None, "trigger_count", 0)),
            int(getattr(None, "map_run_length", 0)),
            int(getattr(None, "cooldown_remaining", 0)),
            int(getattr(None, "above_threshold_count", 0)),
            int(getattr(None, "below_threshold_count", 0)),
            int(
                getattr(
                    getattr(self, "_m3wm_change_null_calibrator", None),
                    "clean_observation_count",
                    0,
                )
            ),
            bool(
                getattr(
                    getattr(self, "_m3wm_change_null_calibrator", None), "frozen", False
                )
            ),
        )

    def context_metrics(self):
        key = self._m3wm_context_metrics_cache_key()
        if key == getattr(self, "_m3wm_context_metrics_cache_key_value", None):
            cached = getattr(self, "_m3wm_context_metrics_cache_value", None)
            if cached is not None:
                return cached
        metrics = self._context_metrics_uncached()
        self._m3wm_context_metrics_cache_key_value = key
        self._m3wm_context_metrics_cache_value = metrics
        return metrics

    def _context_metrics_uncached(self):
        confirmed_event = (
            None
            if self._m3wm_last_confirmed_event is None
            else deepcopy(self._m3wm_last_confirmed_event)
        )
        pending_snapshot = self._m3wm_pending_change_tracker.get() or {}
        quarantine_snapshot = pending_snapshot.get("quarantine") or {}
        live_bocpd_gate = getattr(self, "_m3wm_live_bocpd_edge_gate", None)
        bocpd_edge_metrics = _live_bocpd_edge_gate_metrics(live_bocpd_gate)
        multiscale_fsm_metrics = _multiscale_detector_fsm_metrics(None)
        champion_score_train_updates = -1
        champion_score_context_version = -1
        champion_score_coordinate_current = bool(
            isinstance(None, dict)
            and self._m3wm_episodic_theta_champion_score_is_current(None)
        )
        if not self.cfg.get("enable_m3wm", False):
            return {
                "control_actor_update_count": torch.tensor(
                    float(self._m3wm_control_update_count)
                ),
                "control_actor_update_applied": torch.tensor(
                    float(self._m3wm_last_pi_update_applied)
                ),
                "control_actor_q_std": torch.tensor(float(self._m3wm_last_pi_q_std)),
                "control_actor_ema_penalty": torch.tensor(
                    float(self._m3wm_last_pi_ema_penalty)
                ),
            }
        exemplar_assignment = getattr(
            self, "_m3wm_last_context_exemplar_assignment", {}
        )
        if not isinstance(exemplar_assignment, dict):
            exemplar_assignment = {}
        canonical_physical_witness = exemplar_assignment.get(
            "canonical_physical_departure_witness", {}
        )
        if not isinstance(canonical_physical_witness, dict):
            canonical_physical_witness = {}
        canonical_physical_status = str(
            exemplar_assignment.get("canonical_physical_departure_status", "DISABLED")
        ).upper()
        exemplar_contexts = exemplar_assignment.get("contexts", {})
        if not isinstance(exemplar_contexts, dict):
            exemplar_contexts = {}
        exemplar_active_id = exemplar_assignment.get("active_context_id")
        exemplar_active = exemplar_contexts.get(exemplar_active_id, {})
        if not isinstance(exemplar_active, dict):
            exemplar_active = {}
        exemplar_run_views = exemplar_active.get("run_calibration", {})
        if not isinstance(exemplar_run_views, dict):
            exemplar_run_views = {}
        exemplar_coverage_view = exemplar_run_views.get("coverage", {})
        exemplar_dynamics_view = exemplar_run_views.get("dynamics", {})
        exemplar_reward_view = exemplar_run_views.get("reward", {})
        if not isinstance(exemplar_coverage_view, dict):
            exemplar_coverage_view = {}
        if not isinstance(exemplar_dynamics_view, dict):
            exemplar_dynamics_view = {}
        if not isinstance(exemplar_reward_view, dict):
            exemplar_reward_view = {}
        typed_contexts = exemplar_assignment.get("typed_context_evidence", {})
        if not isinstance(typed_contexts, dict):
            typed_contexts = {}
        exemplar_typed_active = typed_contexts.get(exemplar_active_id, {})
        if not isinstance(exemplar_typed_active, dict):
            exemplar_typed_active = {}
        exemplar_h_new = exemplar_assignment.get("temporary_h_new")
        if not isinstance(exemplar_h_new, dict):
            exemplar_h_new = {}
        terminal_contract = exemplar_assignment.get("terminal_evidence_contract")
        if not isinstance(terminal_contract, dict):
            terminal_contract = getattr(
                self, "_m3wm_last_terminal_evidence_contract", {}
            )
        if not isinstance(terminal_contract, dict):
            terminal_contract = {}
        terminal_diagnostics = exemplar_assignment.get("terminal_evidence_diagnostics")
        if not isinstance(terminal_diagnostics, dict):
            terminal_diagnostics = getattr(
                self, "_m3wm_last_terminal_evidence_diagnostics", {}
            )
        if not isinstance(terminal_diagnostics, dict):
            terminal_diagnostics = {}
        terminal_receipt = exemplar_assignment.get("terminal_evidence_receipt")
        if not isinstance(terminal_receipt, dict):
            terminal_receipt = {}
        terminal_bundle = terminal_diagnostics.get("bundle")
        if not isinstance(terminal_bundle, dict):
            terminal_bundle = terminal_receipt.get("proposal_evidence_bundle", {})
        if not isinstance(terminal_bundle, dict):
            terminal_bundle = {}
        _terminal_reason_codes = {
            "not_evaluated": 0,
            "incomplete_terminal_certificate": 1,
            "identity_confirmed_but_commit_blocked": 2,
            "identity_unresolved": 3,
            "complete_create_certificate": 4,
            "complete_recall_certificate": 5,
            "complete_active_support_certificate": 6,
        }
        _terminal_action_codes = {
            "UNKNOWN_PENDING": 0,
            "CREATE": 1,
            "RECALL": 2,
            "STAY": 3,
        }
        _terminal_stage_codes = {
            "binding": 1,
            "selector": 2,
            "panel": 3,
            "query": 4,
            "active_reference": 5,
            "target_reference": 6,
            "factorized_h_new": 7,
            "factorized_score": 8,
            "factorized_support": 9,
            "target_support": 10,
            "target_superiority": 11,
            "active_support": 12,
            "commit_safety": 13,
            "receipt": 14,
        }
        try:
            terminal_missing_count = len(terminal_contract.get("missing", []))
        except (TypeError, ValueError):
            terminal_missing_count = 0
        terminal_contract_reason = str(terminal_contract.get("reason", "not_evaluated"))
        terminal_contract_action = str(
            terminal_contract.get("action", "UNKNOWN_PENDING")
        ).upper()
        terminal_failure_stage = str(terminal_contract.get("first_failure_stage") or "")
        try:
            exemplar_receipt_proposal_step = int(
                exemplar_assignment.get("proposal_step", -1)
            )
        except (TypeError, ValueError, OverflowError):
            exemplar_receipt_proposal_step = -1
        try:
            exemplar_receipt_due_step = int(exemplar_assignment.get("due_step", -1))
        except (TypeError, ValueError, OverflowError):
            exemplar_receipt_due_step = -1
        try:
            exemplar_receipt_source_context = int(
                exemplar_assignment.get("source_context_id", -1)
            )
        except (TypeError, ValueError, OverflowError):
            exemplar_receipt_source_context = -1
        exemplar_receipt_token = exemplar_assignment.get("proposal_token")
        exemplar_receipt_proposal_bound = bool(
            isinstance(exemplar_receipt_token, (list, tuple))
            and len(exemplar_receipt_token) == 2
            and (exemplar_receipt_token[0] == "m3wm-proposal")
            and (exemplar_receipt_token[1] == exemplar_receipt_proposal_step)
            and (exemplar_receipt_proposal_step >= 0)
            and (
                exemplar_receipt_due_step < 0
                or exemplar_receipt_due_step >= exemplar_receipt_proposal_step
            )
        )
        exemplar_bank_total = {
            "context_count": 0,
            "segment_count": 0,
            "transition_count": 0,
        }
        exemplar_active_bank = {
            "context_id": -1,
            "segment_count": 0,
            "transition_count": 0,
        }
        exemplar_admission = {
            "proposal_step": -1,
            "context_id": -1,
            "episode_start_step": -1,
            "candidate_transition_count": 0,
            "retained_transition_count": 0,
            "retained_phase_min": -1,
            "retained_phase_max": -1,
            "excluded_tail_steps": 0,
            "minimum_phase_end": 0,
            "committed": False,
            "reason": "not_evaluated",
        }
        exemplar_bank = getattr(self, "_m3wm_context_exemplar_bank", None)
        v57_shadow_carrier = {
            "present": False,
            "fsm_state": "ABSENT",
            "attempt_count": 0,
            "owner_boundary_count": 0,
            "batch_prepared": False,
        }
        if exemplar_bank is not None:
            try:
                exemplar_bank_total = exemplar_bank.total_storage_audit()
                exemplar_admission = exemplar_bank.last_admission_audit()
                v57_shadow_carrier = exemplar_bank.provisional_novel_candidate_audit()
            except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
                pass
        v57_registration = getattr(self, "_m3wm_v57_shadow_registration", {})
        if not isinstance(v57_registration, dict):
            v57_registration = {}
        v57_owner_evaluation = getattr(self, "_m3wm_v57_shadow_owner_evaluation", {})
        if not isinstance(v57_owner_evaluation, dict):
            v57_owner_evaluation = {}
        v63_owner_authority = getattr(self, "_m3wm_v63_last_owner_authority", {})
        if not isinstance(v63_owner_authority, dict):
            v63_owner_authority = {}
        v65_fast_commit = getattr(self, "_m3wm_v65_last_fast_commit", {})
        if not isinstance(v65_fast_commit, dict):
            v65_fast_commit = {}
        v57_state_code = _M3WM_V57_SHADOW_STATE_CODES
        exemplar_failure_stage_code = {
            "valid": 0,
            "not_evaluated": 1,
            "identity_disabled": 2,
            "assignment_precondition": 3,
            "current_window_unavailable": 4,
            "source_theta_unavailable": 5,
            "phase_match_unavailable": 6,
            "window_precondition": 7,
            "theta_invalid": 8,
            "snapshot_binding_invalid": 9,
            "reference_batch_validation": 10,
            "query_batch_validation": 11,
            "window_task_mismatch": 12,
            "snapshot_feature_count_mismatch": 13,
            "query_slice_mismatch": 14,
            "window_reencoding_exception": 15,
            "exception": 16,
            "candidate_theta_invalid": 17,
            "insufficient_independent_segments": 18,
            "clean_window_coverage_unavailable": 19,
        }.get(str(exemplar_assignment.get("failure_stage", "not_evaluated")), -1)
        exemplar_theta_contract_code = {
            "not_evaluated": 0,
            "valid": 1,
            "precondition": 2,
            "model_shape_invalid": 3,
            "source_theta_not_tensor": 4,
            "source_theta_dtype_invalid": 5,
            "source_theta_shape_invalid": 6,
            "candidate_theta_not_tensor": 7,
            "candidate_theta_dtype_invalid": 8,
            "candidate_theta_numel_invalid": 9,
            "candidate_theta_nonfinite": 10,
            "candidate_theta_negative": 11,
            "candidate_theta_not_normalized": 12,
            "exception": 13,
        }.get(
            str(exemplar_assignment.get("theta_contract_reason", "not_evaluated")), -1
        )
        exemplar_admission_code = {
            "not_evaluated": 0,
            "committed": 1,
            "invalid_input": 2,
            "episode_invalid": 3,
            "owner_unavailable": 4,
            "owner_mismatch": 5,
            "owner_out_of_range": 6,
            "episode_empty": 7,
            "proposal_step_mismatch": 8,
            "tail_excludes_all": 9,
            "prefix_immature": 10,
        }.get(str(exemplar_admission.get("reason", "not_evaluated")), -1)
        num_primitives = int(self.cfg.get("m3wm_num_primitives", 4))
        uniform = torch.full(
            (
                (3, num_primitives)
                if self.model.m3wm_separate_composition_heads
                else (num_primitives,)
            ),
            1.0 / num_primitives,
            device=self.device,
        )
        raw_theta = uniform if self._m3wm_raw_theta is None else self._m3wm_raw_theta
        smoothed_theta = (
            raw_theta
            if self._m3wm_smoothed_theta is None
            else self._m3wm_smoothed_theta
        )
        planning_theta = (
            smoothed_theta if self._m3wm_last_theta is None else self._m3wm_last_theta
        )
        metrics = self._theta_metrics(planning_theta, "")
        metrics.update(self._theta_metrics(raw_theta, "raw_"))
        metrics.update(self._theta_metrics(smoothed_theta, "smoothed_"))
        router_proposal = (
            uniform
            if self._m3wm_last_router_proposal is None
            else self._m3wm_last_router_proposal
        )
        metrics.update(self._theta_metrics(router_proposal, "proposal_"))
        feature = self._m3wm_last_context_feature
        residual_norm = float("nan")
        residual_mean_abs = float("nan")
        residual_std_abs = float("nan")
        residual_max_abs = float("nan")
        residual_q90 = float("nan")
        action_norm = float("nan")
        action_gain = float("nan")
        action_gain_fit = float("nan")
        action_effect_energy = float("nan")
        system_id_log_norm = self._system_id_norm_feature(
            self._m3wm_current_episode_system_id,
            int(self.cfg.get("m3wm_system_id_window", 128)),
        )
        if system_id_log_norm is None:
            system_id_log_norm = float("nan")
        else:
            system_id_log_norm = system_id_log_norm.squeeze(0).cpu()
        reward_residual_signed = float("nan")
        reward_residual_abs = float("nan")
        if feature is not None:
            feature_cpu = feature.detach().cpu()
            indices = self.model.context_feature_metric_indices()
            residual_idx = indices.get("residual_rms")
            action_idx = indices.get("action_norm")
            action_gain_idx = indices.get("action_gain")
            action_gain_fit_idx = indices.get("action_gain_fit")
            action_effect_energy_idx = indices.get("action_effect_energy")
            reward_idx = indices.get("reward_residual")
            if residual_idx is not None:
                residual_norm = feature_cpu[residual_idx]
            for metric_name, local_name in (
                ("residual_mean_abs", "residual_mean_abs"),
                ("residual_std_abs", "residual_std_abs"),
                ("residual_max_abs", "residual_max_abs"),
                ("residual_q90", "residual_q90"),
            ):
                idx = indices.get(metric_name)
                if idx is None:
                    continue
                value = feature_cpu[idx]
                if local_name == "residual_mean_abs":
                    residual_mean_abs = value
                elif local_name == "residual_std_abs":
                    residual_std_abs = value
                elif local_name == "residual_max_abs":
                    residual_max_abs = value
                else:
                    residual_q90 = value
            if action_idx is not None:
                action_norm = feature_cpu[action_idx]
            if action_gain_idx is not None:
                action_gain = feature_cpu[action_gain_idx]
            if action_gain_fit_idx is not None:
                action_gain_fit = feature_cpu[action_gain_fit_idx]
            if action_effect_energy_idx is not None:
                action_effect_energy = feature_cpu[action_effect_energy_idx]
            if reward_idx is not None:
                reward_residual_signed = feature_cpu[reward_idx]
                reward_residual_abs = reward_residual_signed.abs()
        memory_theta = self._m3wm_last_memory_theta
        control_flat = planning_theta.detach().float().reshape(-1)
        router_flat = router_proposal.detach().float().reshape(-1)
        control_norm_t = torch.linalg.vector_norm(control_flat)
        router_norm_t = torch.linalg.vector_norm(router_flat)
        if memory_theta is None:
            memory_norm_t = control_flat.new_tensor(float("nan"))
            memory_router_cosine_t = control_flat.new_tensor(float("nan"))
            memory_router_l2_t = control_flat.new_tensor(float("nan"))
            memory_control_l2_t = control_flat.new_tensor(float("nan"))
        else:
            memory_flat = memory_theta.detach().float().reshape(-1)
            memory_norm_t = torch.linalg.vector_norm(memory_flat)
            memory_router_cosine_t = F.cosine_similarity(
                memory_flat.unsqueeze(0), router_flat.unsqueeze(0), dim=-1
            ).squeeze(0)
            memory_router_l2_t = torch.linalg.vector_norm(memory_flat - router_flat)
            memory_control_l2_t = torch.linalg.vector_norm(memory_flat - control_flat)
        router_control_l2_t = torch.linalg.vector_norm(router_flat - control_flat)
        norm_values = torch.stack(
            [
                control_norm_t,
                router_norm_t,
                memory_norm_t,
                memory_router_cosine_t,
                memory_router_l2_t,
                memory_control_l2_t,
                router_control_l2_t,
            ]
        ).cpu()
        (
            control_norm,
            router_norm,
            memory_norm,
            memory_router_cosine,
            memory_router_l2,
            memory_control_l2,
            router_control_l2,
        ) = norm_values.unbind(0)
        origin_code = {
            "unassigned": 0.0,
            "created": 1.0,
            "recalled": 2.0,
            "assigned": 3.0,
            "reward_only_frozen": 4.0,
            "reward_only_created": 5.0,
        }.get(self._m3wm_active_context_origin, -1.0)
        bocpd_probability = float("nan")
        bocpd_map_run_length = float("nan")
        bocpd_posterior_size = float("nan")
        null_metadata = getattr(self, "_m3wm_last_change_null_metadata", {})
        if not isinstance(null_metadata, dict):
            null_metadata = {}

        def _null_float(name, default=float("nan")):
            try:
                value = float(null_metadata.get(name, default))
            except (TypeError, ValueError, OverflowError):
                return float(default)
            return value if np.isfinite(value) else float(default)

        null_mode_code = {"fixed_floor": 0.0, "empirical_quantile": 1.0}.get(
            str(null_metadata.get("mode", "fixed_floor")).lower(), -1.0
        )
        null_source_code = {"fallback": 0.0, "robot": 1.0, "context": 2.0}.get(
            str(null_metadata.get("source", "fallback")).lower(), -1.0
        )
        metrics.update(
            {
                "theta_control_norm": control_norm,
                "theta_router_norm": router_norm,
                "theta_memory_norm": memory_norm,
                "theta_memory_router_cosine": memory_router_cosine,
                "theta_memory_router_l2": memory_router_l2,
                "theta_memory_control_l2": memory_control_l2,
                "theta_router_control_l2": router_control_l2,
                "theta_effective_router_blend": torch.tensor(
                    float(self._m3wm_last_router_blend)
                ),
                "active_context_origin_code": torch.tensor(origin_code),
            }
        )
        objective_evaluator = getattr(self, "_m3wm_objective_evaluator", None)
        objective_decision = getattr(self, "_m3wm_last_objective_decision", None)
        objective_phase = (
            "disabled"
            if objective_evaluator is None
            else str(objective_evaluator.phase)
        )
        metrics.update(
            {
                "objective_witness_enabled": torch.tensor(0.0),
                "objective_witness_phase_code": torch.tensor(
                    float(
                        {
                            "disabled": 0,
                            "building_null": 1,
                            "proposal": 2,
                            "complete": 3,
                        }.get(objective_phase, -1)
                    )
                ),
                "objective_witness_stable_samples": torch.tensor(
                    float(
                        0
                        if objective_evaluator is None
                        else objective_evaluator.stable_samples
                    )
                ),
                "objective_witness_proposal_samples": torch.tensor(
                    float(
                        0
                        if objective_evaluator is None
                        else objective_evaluator.proposal_samples
                    )
                ),
                "objective_witness_proposal_available": torch.tensor(
                    float(
                        bool(getattr(self, "_m3wm_objective_proposal_available", False))
                    )
                ),
                "objective_witness_last_changed": torch.tensor(
                    float(
                        bool(
                            objective_decision is not None
                            and objective_decision.changed
                        )
                    )
                ),
                "objective_witness_last_blocks_passed": torch.tensor(
                    float(
                        0
                        if objective_decision is None
                        else objective_decision.blocks_passed
                    )
                ),
                "objective_witness_last_total_score": torch.tensor(
                    float(
                        float("nan")
                        if objective_decision is None
                        else objective_decision.total_score
                    )
                ),
                "objective_witness_proposal_count": torch.tensor(
                    float(getattr(self, "_m3wm_objective_proposal_count", 0))
                ),
                "objective_witness_confirm_count": torch.tensor(
                    float(getattr(self, "_m3wm_objective_confirm_count", 0))
                ),
                "objective_witness_reject_count": torch.tensor(
                    float(getattr(self, "_m3wm_objective_reject_count", 0))
                ),
                "objective_witness_unavailable_count": torch.tensor(
                    float(getattr(self, "_m3wm_objective_unavailable_count", 0))
                ),
            }
        )
        metrics.update(
            {
                "context_history_length": torch.tensor(
                    float(len(self._m3wm_context_history))
                ),
                "context_dyn_residual_rms": residual_norm,
                "context_dyn_residual_mean_abs": residual_mean_abs,
                "context_dyn_residual_std_abs": residual_std_abs,
                "context_dyn_residual_max_abs": residual_max_abs,
                "context_dyn_residual_q90": residual_q90,
                "context_action_norm": action_norm,
                "context_action_gain": action_gain,
                "context_action_gain_fit": action_gain_fit,
                "context_action_effect_energy": action_effect_energy,
                "context_system_id_log_norm": system_id_log_norm,
                "context_reward_residual_signed": reward_residual_signed,
                "context_reward_residual_abs": reward_residual_abs,
                "context_anchor_ready": torch.tensor(
                    float(self.model.context_anchor_ready)
                ),
                "context_anchor_version": torch.tensor(
                    float(
                        self.model._m3wm_context_anchor_version.item()
                        if self.model.m3wm_use_context_anchor
                        else 0
                    )
                ),
                "theta_ema_alpha": torch.tensor(
                    float(self.cfg.get("m3wm_theta_ema_alpha", 0.1))
                ),
                "theta_temperature": torch.tensor(self._m3wm_current_temperature),
                "base_lr_scale": torch.tensor(self._m3wm_current_base_lr_scale),
                "specialization_active": torch.tensor(
                    float(self._m3wm_specialization_active)
                ),
                "control_actor_update_count": torch.tensor(
                    float(self._m3wm_control_update_count)
                ),
                "control_actor_update_applied": torch.tensor(
                    float(self._m3wm_last_pi_update_applied)
                ),
                "control_actor_q_std": torch.tensor(float(self._m3wm_last_pi_q_std)),
                "control_actor_ema_penalty": torch.tensor(
                    float(self._m3wm_last_pi_ema_penalty)
                ),
                "context_runtime_enabled": torch.tensor(
                    float(self._m3wm_context_runtime_enabled())
                ),
                "online_theta_clean_episode_update_limit": torch.tensor(
                    float(self._m3wm_online_theta_clean_episode_update_limit_value())
                ),
                "online_theta_clean_episode_updates": torch.tensor(
                    float(self._m3wm_online_theta_clean_episode_update_count_value())
                ),
                "online_theta_clean_episode_budget_exhausted": torch.tensor(
                    float(self._m3wm_online_theta_clean_episode_budget_exhausted())
                ),
                "online_theta_transaction_update_limit": torch.tensor(
                    float(self._m3wm_online_theta_transaction_update_limit_value())
                ),
                "online_theta_transaction_updates": torch.tensor(
                    float(self._m3wm_online_theta_transaction_update_count_value())
                ),
                "online_theta_transaction_budget_exhausted": torch.tensor(
                    float(self._m3wm_online_theta_transaction_budget_exhausted())
                ),
                "online_theta_transaction_active": torch.tensor(
                    float(
                        bool(
                            self._m3wm_has_pending_change()
                            and getattr(
                                self,
                                "_m3wm_online_theta_transaction_proposal_step",
                                None,
                            )
                            is not None
                        )
                    )
                ),
                "online_theta_transaction_proposal_step": torch.tensor(
                    float(
                        _none_to_nan(
                            getattr(
                                self,
                                "_m3wm_online_theta_transaction_proposal_step",
                                None,
                            )
                        )
                    )
                ),
                "online_theta_terminal_context_hold": torch.tensor(
                    float(
                        bool(
                            getattr(
                                self, "_m3wm_online_theta_terminal_context_hold", False
                            )
                        )
                    )
                ),
                "online_theta_update_owner_code": torch.tensor(
                    float(
                        {
                            "not_called": 0,
                            "ineligible": 1,
                            "clean_gradient": 2,
                            "clean_budget_hold": 3,
                            "transaction_gradient": 4,
                            "reward_only_frozen": 5,
                            "episodic_theta_lock": 6,
                            "baseline_first_hold": 7,
                            "transaction_budget_hold": 8,
                            "terminal_context_hold": 9,
                        }.get(
                            str(
                                getattr(
                                    self,
                                    "_m3wm_online_theta_last_update_owner",
                                    "not_called",
                                )
                            ),
                            -1,
                        )
                    )
                ),
                "online_theta_adapt_eligible": torch.tensor(
                    float(
                        bool(
                            getattr(
                                self, "_m3wm_online_theta_last_adapt_eligible", False
                            )
                        )
                    )
                ),
                "episodic_theta_guard_accept_count": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_episodic_theta_reward_guard_accept_count", 0
                        )
                    )
                ),
                "episodic_theta_guard_reject_count": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_episodic_theta_reward_guard_reject_count", 0
                        )
                    )
                ),
                "episodic_theta_guard_last_score": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_episodic_theta_reward_guard_last_score",
                            float("nan"),
                        )
                    )
                ),
                "episodic_theta_guard_last_decision_code": torch.tensor(
                    float(
                        {
                            "none": 0,
                            "bootstrap": 1,
                            "accept": 2,
                            "reject_restore": 3,
                            "ineligible": 4,
                            "baseline_bootstrap": 5,
                            "stale_rebootstrap": 6,
                        }.get(
                            getattr(
                                self,
                                "_m3wm_episodic_theta_reward_guard_last_decision",
                                "none",
                            ),
                            -1,
                        )
                    )
                ),
                "episodic_theta_guard_champion_score_train_updates": torch.tensor(
                    float(champion_score_train_updates)
                ),
                "episodic_theta_guard_champion_score_context_version": torch.tensor(
                    float(champion_score_context_version)
                ),
                "episodic_theta_guard_champion_score_coordinate_current": torch.tensor(
                    float(champion_score_coordinate_current)
                ),
                "episodic_theta_guard_champion_score_train_update_age": torch.tensor(
                    float("nan")
                    if champion_score_train_updates < 0
                    else float(
                        int(getattr(self, "_m3wm_train_updates", 0))
                        - champion_score_train_updates
                    )
                ),
                "episodic_theta_guard_champion_score_context_version_age": torch.tensor(
                    float("nan")
                    if champion_score_context_version < 0
                    else float(
                        int(self._m3wm_context_version) - champion_score_context_version
                    )
                ),
                "episodic_theta_baseline_first_enabled": torch.tensor(0.0),
                "episodic_theta_baseline_first_hold": torch.tensor(0.0),
                "episodic_theta_baseline_first_status_code": torch.tensor(
                    float({"collecting": 1, "resolved_hold": 2}.get(None, 0))
                ),
                "episodic_theta_baseline_first_complete_blocks": torch.tensor(0.0),
                "episodic_theta_baseline_first_partial_samples": torch.tensor(0.0),
                "pending_trigger_steps": torch.tensor(0.0),
                "pending_change_active": torch.tensor(
                    float(self._m3wm_has_pending_change())
                ),
                "pending_change_steps_remaining": torch.tensor(
                    float(self._m3wm_pending_steps_remaining())
                ),
                "p2_quarantine_active": torch.tensor(float(bool(quarantine_snapshot))),
                "pending_original_confirmation_due_step": torch.tensor(
                    _none_to_nan(pending_snapshot.get("original_confirmation_due_step"))
                ),
                "pending_current_confirmation_due_step": torch.tensor(
                    _none_to_nan(pending_snapshot.get("confirmation_due_step"))
                ),
                "pending_retry_confirmation_due_step": torch.tensor(
                    _none_to_nan(quarantine_snapshot.get("retry_confirmation_due_step"))
                ),
                "p2_quarantine_retry_count": torch.tensor(
                    _none_to_nan(quarantine_snapshot.get("retry_count"))
                ),
                "context_memory_size": torch.tensor(0.0),
                "active_context_id": torch.tensor(-1.0),
                "change_triggered": torch.tensor(
                    float(self._m3wm_last_change_triggered)
                ),
                "change_trigger_source": torch.tensor(
                    float(
                        {
                            "none": 0.0,
                            "bocpd": 1.0,
                            "continuous": 2.0,
                            "dp_memory": 3.0,
                            "multiscale": 4.0,
                            M3WM_EPISODE_IDENTITY_AUDIT_SOURCE: 5.0,
                        }.get(self._m3wm_last_trigger_source, 0.0)
                    )
                ),
                "episode_identity_audit_proposal_count": torch.tensor(
                    float(
                        getattr(self, "_m3wm_episode_identity_audit_proposal_count", 0)
                    )
                ),
                "last_identity_audit_step": torch.tensor(
                    float(getattr(self, "_m3wm_last_identity_audit_step", -1))
                ),
                "last_identity_audit_score": torch.tensor(
                    float(
                        getattr(self, "_m3wm_last_identity_audit_score", float("nan"))
                    )
                ),
                "last_identity_audit_reason_code": torch.tensor(
                    float(
                        {
                            "none": 0.0,
                            "dual_identity_novelty": 1.0,
                            "reward_only_disagreement": 2.0,
                            "group_recall_conflict": 3.0,
                        }.get(
                            getattr(self, "_m3wm_last_identity_audit_reason", "none"),
                            -1.0,
                        )
                    )
                ),
                "change_statistic": torch.tensor(
                    float(self._m3wm_last_change_statistic)
                ),
                "change_fast_dynamics_shift": torch.tensor(
                    float(getattr(self, "_m3wm_last_fast_dynamics_shift", float("nan")))
                ),
                "change_fast_abs_reward_shift": torch.tensor(
                    float(
                        getattr(self, "_m3wm_last_fast_abs_reward_shift", float("nan"))
                    )
                ),
                "change_slow_signed_reward_shift": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_last_slow_signed_reward_shift", float("nan")
                        )
                    )
                ),
                "change_signed_reward_slow_enabled": torch.tensor(
                    float(
                        bool(getattr(self, "_m3wm_signed_reward_slow_enabled", False))
                    )
                ),
                "bocpd_short_run_probability": torch.tensor(bocpd_probability),
                "bocpd_map_run_length": torch.tensor(bocpd_map_run_length),
                "bocpd_posterior_size": torch.tensor(bocpd_posterior_size),
                "pending_change_source": torch.tensor(
                    float(
                        {
                            "none": 0.0,
                            "bocpd": 1.0,
                            "continuous": 2.0,
                            "dp_memory": 3.0,
                            "multiscale": 4.0,
                            M3WM_EPISODE_IDENTITY_AUDIT_SOURCE: 5.0,
                        }.get(
                            (self._m3wm_pending_change_tracker.get() or {}).get(
                                "source", "none"
                            ),
                            0.0,
                        )
                    )
                ),
                "pending_drop_count": torch.tensor(
                    float(self._m3wm_pending_change_tracker.drop_count)
                ),
                "proposal_supersede_count": torch.tensor(
                    float(getattr(self, "_m3wm_proposal_supersede_count", 0))
                ),
                "proposal_superseded_from_step": torch.tensor(
                    float(getattr(self, "_m3wm_last_proposal_superseded_from_step", -1))
                ),
                "proposal_superseded_to_step": torch.tensor(
                    float(getattr(self, "_m3wm_last_proposal_superseded_to_step", -1))
                ),
                "bocpd_proposal_step": torch.tensor(
                    _none_to_nan(
                        (self._m3wm_pending_change_tracker.get() or {}).get(
                            "proposal_step"
                        )
                    )
                ),
                "bocpd_proposal_probability": torch.tensor(
                    _none_to_nan(
                        (self._m3wm_pending_change_tracker.get() or {}).get(
                            "proposal_probability"
                        )
                    )
                ),
                "bocpd_proposal_map_run_length": torch.tensor(
                    _none_to_nan(
                        (self._m3wm_pending_change_tracker.get() or {}).get(
                            "proposal_map_run_length"
                        )
                    )
                ),
                "bocpd_confirmation_step": torch.tensor(
                    _none_to_nan(
                        (self._m3wm_pending_change_tracker.get() or {}).get(
                            "confirmation_step"
                        )
                    )
                ),
                "bocpd_confirmation_probability": torch.tensor(
                    _none_to_nan(
                        (self._m3wm_pending_change_tracker.get() or {}).get(
                            "confirmation_probability"
                        )
                    )
                ),
                "last_confirmed_source": torch.tensor(
                    float(
                        {
                            "none": 0.0,
                            "bocpd": 1.0,
                            "continuous": 2.0,
                            "dp_memory": 3.0,
                            "multiscale": 4.0,
                            M3WM_EPISODE_IDENTITY_AUDIT_SOURCE: 5.0,
                        }.get((confirmed_event or {}).get("source", "none"), 0.0)
                    )
                ),
                "last_confirmed_proposal_step": torch.tensor(
                    _none_to_nan((confirmed_event or {}).get("proposal_step"))
                ),
                "last_confirmed_proposal_probability": torch.tensor(
                    _none_to_nan((confirmed_event or {}).get("proposal_probability"))
                ),
                "last_confirmed_proposal_map_run_length": torch.tensor(
                    _none_to_nan((confirmed_event or {}).get("proposal_map_run_length"))
                ),
                "last_confirmed_step": torch.tensor(
                    _none_to_nan((confirmed_event or {}).get("confirmation_step"))
                ),
                "last_confirmed_probability": torch.tensor(
                    _none_to_nan(
                        (confirmed_event or {}).get("confirmation_probability")
                    )
                ),
                "change_adaptive_threshold": torch.tensor(
                    float(self._m3wm_adaptive_change_threshold)
                ),
                "change_null_mode_code": torch.tensor(float(null_mode_code)),
                "change_null_alpha": torch.tensor(_null_float("alpha")),
                "change_null_effective_threshold": torch.tensor(
                    _null_float("effective_threshold")
                ),
                "change_null_ready": torch.tensor(
                    float(bool(null_metadata.get("ready", False)))
                ),
                "change_null_source_code": torch.tensor(float(null_source_code)),
                "change_null_sample_count": torch.tensor(
                    float(null_metadata.get("sample_count", 0))
                ),
                "change_null_context_sample_count": torch.tensor(
                    float(null_metadata.get("context_sample_count", 0))
                ),
                "change_null_robot_sample_count": torch.tensor(
                    float(null_metadata.get("robot_sample_count", 0))
                ),
                "change_null_percentile": torch.tensor(_null_float("percentile")),
                "change_null_tail_probability": torch.tensor(
                    _null_float("tail_probability")
                ),
                "change_null_dynamics_percentile": torch.tensor(
                    _null_float("dynamics_percentile")
                ),
                "change_null_dynamics_tail_probability": torch.tensor(
                    _null_float("dynamics_tail_probability")
                ),
                "change_null_reward_percentile": torch.tensor(
                    _null_float("reward_percentile")
                ),
                "change_null_reward_tail_probability": torch.tensor(
                    _null_float("reward_tail_probability")
                ),
                "change_null_frozen": torch.tensor(
                    float(bool(null_metadata.get("frozen", False)))
                ),
                "change_null_clean_observation_count": torch.tensor(
                    float(null_metadata.get("clean_observation_count", 0))
                ),
                "change_null_rejected_observation_count": torch.tensor(
                    float(null_metadata.get("rejected_observation_count", 0))
                ),
                "change_null_std_floor_hits": torch.tensor(
                    float(null_metadata.get("std_floor_hits", 0))
                ),
                "change_null_statistic_count": torch.tensor(
                    float(null_metadata.get("statistic_count", 0))
                ),
                "change_null_max_statistic": torch.tensor(
                    float(self._m3wm_episode_null_max_statistic)
                ),
                "change_null_reference_statistic": torch.tensor(
                    float(self._m3wm_last_null_reference_statistic)
                ),
                "memory_existing_log_posterior": torch.tensor(
                    float(self._m3wm_last_existing_log_posterior)
                ),
                "memory_new_log_posterior": torch.tensor(
                    float(self._m3wm_last_new_log_posterior)
                ),
                "change_trigger_count": torch.tensor(
                    float(self._m3wm_confirmed_change_count)
                ),
                "change_proposal_count": torch.tensor(0.0),
                **{
                    name: torch.tensor(value)
                    for (name, value) in bocpd_edge_metrics.items()
                },
                **{
                    name: torch.tensor(value)
                    for (name, value) in multiscale_fsm_metrics.items()
                },
                "change_rejection_count": torch.tensor(
                    float(self._m3wm_rejected_change_count)
                ),
                "bocpd_a1_rejected_count": torch.tensor(
                    float(self._m3wm_bocpd_a1_rejected_count)
                ),
                "bocpd_a2_rejected_count": torch.tensor(
                    float(self._m3wm_bocpd_a2_rejected_count)
                ),
                "regime_rejected_count": torch.tensor(
                    float(self._m3wm_regime_rejected_count)
                ),
                "dp_identity_dismissed": torch.tensor(
                    float(self._m3wm_dp_identity_dismissed)
                ),
                "dp_identity_created": torch.tensor(
                    float(self._m3wm_dp_identity_created)
                ),
                "dp_identity_switched": torch.tensor(
                    float(self._m3wm_dp_identity_switched)
                ),
                "dp_best_existing_context": torch.tensor(
                    float(self._m3wm_last_dp_best_existing_context)
                ),
                "dp_active_log_posterior": torch.tensor(
                    float(self._m3wm_last_dp_active_log_posterior)
                ),
                "dp_recall_advantage": torch.tensor(
                    float(self._m3wm_last_dp_recall_advantage)
                ),
                "dp_recall_emission_advantage": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_recall_emission_advantage",
                            float("nan"),
                        )
                    )
                ),
                "dp_recall_emission_pass": torch.tensor(
                    float(
                        bool(getattr(self, "_m3wm_last_dp_recall_emission_pass", False))
                    )
                ),
                "dp_recall_candidate_context": torch.tensor(
                    float(getattr(self, "_m3wm_last_dp_recall_candidate_context", -1))
                ),
                "dp_recall_candidate_rms": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_last_dp_recall_candidate_rms", float("nan")
                        )
                    )
                ),
                "dp_recall_candidate_rms_threshold": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_recall_candidate_rms_threshold",
                            float("nan"),
                        )
                    )
                ),
                "dp_recall_candidate_support_pass": torch.tensor(
                    float(
                        bool(
                            getattr(
                                self,
                                "_m3wm_last_dp_recall_candidate_support_pass",
                                False,
                            )
                        )
                    )
                ),
                "context_exemplar_identity_enabled": torch.tensor(0.0),
                "v57_shadow_enabled": torch.tensor(0.0),
                "v57_shadow_only": torch.tensor(0.0),
                "v57_shadow_legacy_v56_authority": torch.tensor(1.0),
                "v57_shadow_revision_binding_authoritative": torch.tensor(0.0),
                "v57_shadow_fail_stopped": torch.tensor(
                    float(bool(getattr(self, "_m3wm_v57_shadow_fail_stopped", False)))
                ),
                "v57_shadow_sidecar_durable": torch.tensor(0.0),
                "v57_shadow_sidecar_queue_depth": torch.tensor(
                    float(len(getattr(self, "_m3wm_v57_shadow_sidecar_queue", ())))
                ),
                "v57_shadow_sidecar_queue_capacity": torch.tensor(
                    float(_M3WM_V57_SHADOW_SIDECAR_CAPACITY)
                ),
                "v57_shadow_registration_state_code": torch.tensor(
                    float(
                        v57_state_code.get(
                            str(v57_registration.get("state", "UNRESOLVED")),
                            v57_state_code["UNKNOWN"],
                        )
                    )
                ),
                "v57_shadow_owner_state_code": torch.tensor(
                    float(
                        v57_state_code.get(
                            str(v57_owner_evaluation.get("state", "UNRESOLVED")),
                            v57_state_code["UNKNOWN"],
                        )
                    )
                ),
                "v57_shadow_carrier_present": torch.tensor(
                    float(bool(v57_shadow_carrier.get("present", False)))
                ),
                "v57_shadow_owner_boundary_count": torch.tensor(
                    float(v57_shadow_carrier.get("owner_boundary_count", 0))
                ),
                "v57_shadow_owner_attempt_count": torch.tensor(
                    float(v57_shadow_carrier.get("attempt_count", 0))
                ),
                "v57_shadow_owner_batch_prepared": torch.tensor(
                    float(bool(v57_shadow_carrier.get("batch_prepared", False)))
                ),
                "v57_shadow_replication_confirmed": torch.tensor(
                    float(
                        bool(v57_owner_evaluation.get("replication_confirmed", False))
                    )
                ),
                "v57_shadow_coverage_gap": torch.tensor(
                    float(bool(str(v57_registration.get("state")) == "COVERAGE_GAP"))
                ),
                "v63_owner_authority_enabled": torch.tensor(
                    float(
                        bool(getattr(self, "_m3wm_v63_owner_authority_enabled", False))
                    )
                ),
                "v63_owner_authority_state_code": torch.tensor(
                    float(
                        {
                            "DISABLED": -2,
                            "NOT_EVALUATED": -1,
                            "UNRESOLVED": 0,
                            "STAY_COMMITTED": 1,
                            "RECALL_COMMITTED": 2,
                            "CREATE_COMMITTED": 3,
                        }.get(str(v63_owner_authority.get("state")), -3)
                    )
                ),
                "v63_owner_authority_context_id": torch.tensor(
                    float(v63_owner_authority.get("context_id", -1))
                ),
                "v65_fast2o_commit_state_code": torch.tensor(
                    float(
                        {"CREATE_COMMITTED_YOUNG": 1}.get(
                            str(v65_fast_commit.get("state")), 0
                        )
                    )
                ),
                "v65_fast2o_commit_context_id": torch.tensor(
                    float(v65_fast_commit.get("context_id", -1))
                ),
                "context_exemplar_valid": torch.tensor(
                    float(bool(exemplar_assignment.get("valid", False)))
                ),
                "terminal_contract_valid": torch.tensor(
                    float(bool(terminal_contract.get("valid", False)))
                ),
                "terminal_contract_action_code": torch.tensor(
                    float(_terminal_action_codes.get(terminal_contract_action, -1))
                ),
                "terminal_contract_reason_code": torch.tensor(
                    float(_terminal_reason_codes.get(terminal_contract_reason, -1))
                ),
                "terminal_contract_first_failure_stage_code": torch.tensor(
                    float(_terminal_stage_codes.get(terminal_failure_stage, 0))
                ),
                "terminal_contract_missing_count": torch.tensor(
                    float(terminal_missing_count)
                ),
                "identity_evidence_confirmed": torch.tensor(
                    float(
                        bool(
                            terminal_contract.get("identity_evidence_confirmed", False)
                        )
                    )
                ),
                "identity_confirmed_but_commit_blocked": torch.tensor(
                    float(
                        bool(
                            terminal_contract.get(
                                "identity_confirmed_but_commit_blocked", False
                            )
                        )
                    )
                ),
                "proposal_bundle_valid": torch.tensor(
                    float(bool(terminal_bundle.get("query_valid", False)))
                ),
                "proposal_bundle_panel_complete": torch.tensor(
                    float(bool(terminal_bundle.get("panel_complete", False)))
                ),
                "proposal_bundle_active_reference_valid": torch.tensor(
                    float(bool(terminal_bundle.get("active_reference_valid", False)))
                ),
                "proposal_bundle_factorized_valid": torch.tensor(
                    float(bool(terminal_bundle.get("factorized_receipt_valid", False)))
                ),
                "proposal_bundle_h_new_valid": torch.tensor(
                    float(bool(terminal_bundle.get("h_new_valid", False)))
                ),
                "proposal_bundle_selection_count": torch.tensor(
                    float(terminal_bundle.get("selection_count", 0))
                ),
                "proposal_bundle_validation_a_count": torch.tensor(
                    float(terminal_bundle.get("validation_a_count", 0))
                ),
                "proposal_bundle_validation_b_count": torch.tensor(
                    float(terminal_bundle.get("validation_b_count", 0))
                ),
                "proposal_bundle_reset_boundary_count": torch.tensor(
                    float(terminal_bundle.get("reset_boundary_count", 0))
                ),
                "proposal_bundle_episode_chunk_count": torch.tensor(
                    float(terminal_bundle.get("episode_chunk_count", 0))
                ),
                "proposal_bundle_physical_witness_pooled": torch.tensor(
                    float(bool(terminal_bundle.get("physical_witness_pooled", False)))
                ),
                "proposal_bundle_physical_witness_episode_chunk_count": torch.tensor(
                    float(
                        terminal_bundle.get("physical_witness_episode_chunk_count", 0)
                    )
                ),
                "proposal_bundle_active_comparable": torch.tensor(
                    float(bool(terminal_bundle.get("active_comparable", False)))
                ),
                "proposal_bundle_active_match": torch.tensor(
                    float(bool(terminal_bundle.get("active_match", False)))
                ),
                "proposal_bundle_active_rejected": torch.tensor(
                    float(bool(terminal_bundle.get("active_rejected", False)))
                ),
                "proposal_bundle_unified_terminal_authority": torch.tensor(
                    float(
                        bool(terminal_bundle.get("unified_terminal_authority", False))
                    )
                ),
                "proposal_supersede_count": torch.tensor(
                    float(getattr(self, "_m3wm_proposal_supersede_count", 0))
                ),
                "proposal_superseded_from_step": torch.tensor(
                    float(getattr(self, "_m3wm_last_proposal_superseded_from_step", -1))
                ),
                "proposal_superseded_to_step": torch.tensor(
                    float(getattr(self, "_m3wm_last_proposal_superseded_to_step", -1))
                ),
                "proposal_superseded_reason_code": torch.tensor(
                    float(
                        {"none": 0, "formal_detector_reanchor": 1}.get(
                            str(
                                getattr(
                                    self,
                                    "_m3wm_last_proposal_superseded_reason",
                                    "none",
                                )
                            ),
                            -1,
                        )
                    )
                ),
                "context_exemplar_action_code": torch.tensor(
                    float(
                        {
                            "not_evaluated": 0,
                            "dismiss_active": 1,
                            "recall_confident": 2,
                            "create_novel": 3,
                            "dismiss_capacity": 4,
                        }.get(exemplar_assignment.get("action"), -1)
                    )
                ),
                "context_exemplar_selected_context": torch.tensor(
                    float(
                        -1
                        if exemplar_assignment.get("selected_context_id") is None
                        else int(exemplar_assignment["selected_context_id"])
                    )
                ),
                "context_exemplar_supported_count": torch.tensor(
                    float(
                        len(
                            exemplar_assignment.get("supported_context_ids", [])
                            if isinstance(
                                exemplar_assignment.get("supported_context_ids", []),
                                list,
                            )
                            else []
                        )
                    )
                ),
                "context_exemplar_active_rms": torch.tensor(
                    float(exemplar_active.get("standardized_rms", float("nan")))
                ),
                "context_exemplar_active_radius": torch.tensor(
                    float(exemplar_active.get("standardized_rms_radius", float("nan")))
                ),
                "context_exemplar_active_support_pass": torch.tensor(
                    float(bool(exemplar_active.get("support_pass", False)))
                ),
                "context_exemplar_active_reference_count": torch.tensor(
                    float(exemplar_active.get("reference_count", 0))
                ),
                "context_exemplar_bank_total_segments": torch.tensor(
                    float(exemplar_bank_total.get("segment_count", 0))
                ),
                "context_exemplar_bank_total_transitions": torch.tensor(
                    float(exemplar_bank_total.get("transition_count", 0))
                ),
                "context_exemplar_active_bank_segments": torch.tensor(
                    float(exemplar_active_bank.get("segment_count", 0))
                ),
                "context_exemplar_active_bank_transitions": torch.tensor(
                    float(exemplar_active_bank.get("transition_count", 0))
                ),
                "context_exemplar_last_admission_step": torch.tensor(
                    float(exemplar_admission.get("proposal_step", -1))
                ),
                "context_exemplar_last_admission_code": torch.tensor(
                    float(exemplar_admission_code)
                ),
                "context_exemplar_last_admission_retained_count": torch.tensor(
                    float(exemplar_admission.get("retained_transition_count", 0))
                ),
                "context_exemplar_last_admission_phase_min": torch.tensor(
                    float(exemplar_admission.get("retained_phase_min", -1))
                ),
                "context_exemplar_last_admission_phase_max": torch.tensor(
                    float(exemplar_admission.get("retained_phase_max", -1))
                ),
                "context_exemplar_failure_stage_code": torch.tensor(
                    float(exemplar_failure_stage_code)
                ),
                "context_exemplar_failure_context": torch.tensor(
                    float(
                        -1
                        if exemplar_assignment.get("failure_context_id") is None
                        else int(exemplar_assignment["failure_context_id"])
                    )
                ),
                "context_exemplar_theta_contract_code": torch.tensor(
                    float(exemplar_theta_contract_code)
                ),
                "context_exemplar_active_phase_match_count": torch.tensor(
                    float(exemplar_active.get("phase_match_count", 0))
                ),
                "context_exemplar_active_eligible_segment_count": torch.tensor(
                    float(exemplar_active.get("eligible_segment_count", 0))
                ),
                "context_exemplar_active_selected_window_count": torch.tensor(
                    float(exemplar_active.get("selected_window_count", 0))
                ),
                "context_exemplar_active_reference_window_count": torch.tensor(
                    float(exemplar_active.get("reference_window_count", 0))
                ),
                "context_exemplar_active_validated_reference_count": torch.tensor(
                    float(exemplar_active.get("validated_reference_count", 0))
                ),
                "context_exemplar_active_query_window_count": torch.tensor(
                    float(exemplar_active.get("query_window_count", 0))
                ),
                "context_exemplar_active_validated_query_count": torch.tensor(
                    float(exemplar_active.get("validated_query_count", 0))
                ),
                "context_exemplar_active_query_support_count": torch.tensor(
                    float(exemplar_active.get("query_support_count", 0))
                ),
                "context_exemplar_active_query_required_support": torch.tensor(
                    float(exemplar_active.get("required_query_support_count", 0))
                ),
                "context_exemplar_query_valid": torch.tensor(
                    float(bool(exemplar_active.get("query_valid", False)))
                ),
                "context_exemplar_snapshot_valid": torch.tensor(
                    float(bool(exemplar_active.get("snapshot_valid", False)))
                ),
                "context_exemplar_theta_valid": torch.tensor(
                    float(bool(exemplar_active.get("theta_valid", False)))
                ),
                "context_exemplar_recompute_count": torch.tensor(
                    float(exemplar_active.get("recompute_count", 0))
                ),
                "context_exemplar_evaluated_this_transaction": torch.tensor(
                    float(
                        bool(
                            exemplar_assignment.get("evaluated_this_transaction", False)
                        )
                    )
                ),
                "context_exemplar_receipt_proposal_step": torch.tensor(
                    float(exemplar_receipt_proposal_step)
                ),
                "context_exemplar_receipt_due_step": torch.tensor(
                    float(exemplar_receipt_due_step)
                ),
                "context_exemplar_receipt_source_context": torch.tensor(
                    float(exemplar_receipt_source_context)
                ),
                "context_exemplar_receipt_proposal_bound": torch.tensor(
                    float(exemplar_receipt_proposal_bound)
                ),
                "context_predictive_action_code": torch.tensor(
                    float(
                        {
                            "stay_active": 1,
                            "recall_confident": 2,
                            "create_novel": 3,
                            "unresolved": 4,
                            "dismiss_capacity": 5,
                            "provisional_stay_active": 6,
                        }.get(exemplar_assignment.get("predictive_action"), 0)
                    )
                ),
                "canonical_physical_departure_enabled": torch.tensor(
                    float(
                        bool(
                            exemplar_assignment.get(
                                "canonical_physical_departure_enabled", False
                            )
                        )
                    )
                ),
                "canonical_physical_departure_status_code": torch.tensor(
                    float(
                        {"DISABLED": 0, "MATCH": 1, "REJECT": 2, "NO_COVERAGE": 3}.get(
                            canonical_physical_status, 3
                        )
                    )
                ),
                "canonical_physical_departure_score": torch.tensor(
                    float(canonical_physical_witness.get("score", float("nan")))
                ),
                "canonical_physical_departure_threshold": torch.tensor(
                    float(canonical_physical_witness.get("threshold", float("nan")))
                ),
                "canonical_physical_departure_support_count": torch.tensor(
                    float(canonical_physical_witness.get("support_count", 0))
                ),
                "canonical_physical_departure_mutual_support_count": torch.tensor(
                    float(canonical_physical_witness.get("mutual_support_count", 0))
                ),
                "context_predictive_active_calibration_reliable": torch.tensor(
                    float(
                        bool(exemplar_typed_active.get("calibration_reliable", False))
                    )
                ),
                "context_predictive_active_lifecycle_state_code": torch.tensor(
                    float(
                        {"PROVISIONAL_ACTIVE": 1, "COMMITTED_ACTIVE": 2}.get(
                            exemplar_assignment.get(
                                "active_lifecycle_state",
                                exemplar_typed_active.get("lifecycle_state"),
                            ),
                            0,
                        )
                    )
                ),
                "context_predictive_active_coverage_state_code": torch.tensor(
                    float(
                        {
                            "COVERED": 1,
                            "OUT_OF_COVERAGE": 2,
                            "INSUFFICIENT_CALIBRATION": 3,
                        }.get(exemplar_typed_active.get("coverage_state"), 0)
                    )
                ),
                "context_predictive_active_state_code": torch.tensor(
                    float(
                        {"SUPPORT": 1, "CONFIDENT_REJECT": 2, "UNRESOLVED": 3}.get(
                            exemplar_typed_active.get("predictive_state"), 0
                        )
                    )
                ),
                "context_predictive_effective_calibration_units": torch.tensor(
                    float(exemplar_coverage_view.get("effective_calibration_units", 0))
                ),
                "context_predictive_calibration_window_count": torch.tensor(
                    float(exemplar_coverage_view.get("window_count", 0))
                ),
                "context_predictive_calibration_extreme_rank_resolution": torch.tensor(
                    float(
                        exemplar_coverage_view.get(
                            "calibration_extreme_rank_resolution", float("nan")
                        )
                    )
                ),
                "context_predictive_coverage_view_state_code": torch.tensor(
                    float(
                        {
                            "SUPPORT": 1,
                            "CONFIDENT_REJECT": 2,
                            "INSUFFICIENT_CALIBRATION": 3,
                            "CALIBRATION_UNRELIABLE": 4,
                            "UNRESOLVED": 5,
                        }.get(exemplar_coverage_view.get("state"), 0)
                    )
                ),
                "context_predictive_dynamics_view_state_code": torch.tensor(
                    float(
                        {
                            "SUPPORT": 1,
                            "CONFIDENT_REJECT": 2,
                            "INSUFFICIENT_CALIBRATION": 3,
                            "CALIBRATION_UNRELIABLE": 4,
                            "UNRESOLVED": 5,
                        }.get(exemplar_dynamics_view.get("state"), 0)
                    )
                ),
                "context_predictive_reward_view_state_code": torch.tensor(
                    float(
                        {
                            "SUPPORT": 1,
                            "CONFIDENT_REJECT": 2,
                            "INSUFFICIENT_CALIBRATION": 3,
                            "CALIBRATION_UNRELIABLE": 4,
                            "UNRESOLVED": 5,
                        }.get(exemplar_reward_view.get("state"), 0)
                    )
                ),
                "context_h_new_valid": torch.tensor(
                    float(bool(exemplar_h_new.get("valid", False)))
                ),
                "context_h_new_support_query_separated": torch.tensor(
                    float(bool(exemplar_h_new.get("support_query_separated", False)))
                ),
                "context_h_new_base_model_read_only": torch.tensor(
                    float(bool(exemplar_h_new.get("base_model_read_only", False)))
                ),
                "context_h_new_runtime_theta_read_only": torch.tensor(
                    float(bool(exemplar_h_new.get("runtime_theta_read_only", False)))
                ),
                "context_h_new_heldout_predictive_support": torch.tensor(
                    float(bool(exemplar_h_new.get("heldout_predictive_support", False)))
                ),
                "context_h_new_future_predictive_support": torch.tensor(
                    float(bool(exemplar_h_new.get("future_predictive_support", False)))
                ),
                "context_h_new_adaptation_steps": torch.tensor(
                    float(exemplar_h_new.get("adaptation_steps", 0))
                ),
                "dp_identity_decision_code": torch.tensor(
                    float(
                        {
                            "none": 0,
                            "dismiss_no_slots": 1,
                            "dismiss_active": 2,
                            "create_novel": 3,
                            "recall_legacy": 4,
                            "recall_confident": 5,
                            "create_ambiguous": 6,
                            "dismiss_capacity": 7,
                            "quarantine": 8,
                            "dismiss_quarantine_retry": 9,
                            "dismiss_invalid_r3_shadow": 10,
                            "dismiss_quarantine_feature_not_ready": 11,
                            "dismiss_quarantine_no_slots": 12,
                            "dismiss_regime_inconsistent_direct_create": 13,
                            "create_active_consensus_ambiguous": 14,
                            "dismiss_invalid_active_consensus_ambiguous": 15,
                            "dismiss_regime_inconsistent_direct_recall": 16,
                        }.get(self._m3wm_last_dp_identity_decision, -1)
                    )
                ),
                "dp_departure_reason_code": torch.tensor(
                    float(
                        {
                            "none": 0,
                            "invalid_evidence": 1,
                            "proposal_active_context_mismatch": 2,
                            "active_slot_missing": 3,
                            "detector_proposal_not_pending": 4,
                            "regime_d_z_gate_failed": 5,
                            "active_identity_gate_failed": 6,
                            "confirmed_departure": 7,
                            "confirmed_departure_capacity_exhausted": 8,
                            "confirmed_strong_dynamics": 9,
                            "same_version_witness_gate_failed": 10,
                            "confirmed_persistent_signed_reward": 11,
                            "persistent_signed_reward_gate_failed": 12,
                            "pending_persistence_gate_failed": 13,
                            "confirmed_adapted_identity": 14,
                            "strong_dynamics_consensus_failed": 15,
                            "adapted_identity_recall_unconfirmed": 16,
                            "confirmed_physical_response": 17,
                            "objective_witness_gate_failed": 18,
                            "confirmed_prequential_objective": 19,
                            "novel_creation_posterior_gate_failed": 20,
                            "context_exemplar_evidence_unavailable": 21,
                            "context_exemplar_active_self_support": 22,
                            "context_exemplar_inactive_self_support": 23,
                            "context_exemplar_open_set_novel": 24,
                            "context_exemplar_capacity_exhausted": 25,
                            "active_calibration_unreliable": 26,
                            "active_out_of_coverage": 27,
                            "active_calibration_insufficient": 28,
                            "active_predictive_support": 29,
                            "active_predictive_unresolved": 30,
                            "inactive_predictive_support": 31,
                            "old_context_evidence_unresolved": 32,
                            "temporary_novel_hypothesis_unconfirmed": 33,
                            "temporary_novel_hypothesis_confirmed": 34,
                            "context_capacity_exhausted": 35,
                            "active_provisional_conditional_support": 36,
                            "crossfit_symmetric_residual_rank_hypothesis_confirmed": 37,
                            "inactive_predictive_support_during_active_provisional": 38,
                            "crossfit_residual_rank_change_unconfirmed": 39,
                        }.get(
                            getattr(self, "_m3wm_last_dp_departure_reason", "none"), -1
                        )
                    )
                ),
                "dp_departure_active_standardized_rms": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_departure_active_standardized_rms",
                            float("nan"),
                        )
                    )
                ),
                "dp_departure_rms_threshold": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_last_dp_departure_rms_threshold", float("nan")
                        )
                    )
                ),
                "dp_departure_regime_d_z": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_last_dp_departure_regime_d_z", float("nan")
                        )
                    )
                ),
                "dp_proposal_fast_dynamics_shift": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_proposal_fast_dynamics_shift",
                            float("nan"),
                        )
                    )
                ),
                "dp_proposal_fast_abs_reward_shift": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_proposal_fast_abs_reward_shift",
                            float("nan"),
                        )
                    )
                ),
                "dp_proposal_slow_signed_reward_shift": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_proposal_slow_signed_reward_shift",
                            float("nan"),
                        )
                    )
                ),
                "dp_same_version_witness_score": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_same_version_witness_score",
                            float("nan"),
                        )
                    )
                ),
                "dp_same_version_witness_threshold": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_same_version_witness_threshold",
                            float("nan"),
                        )
                    )
                ),
                "dp_same_version_witness_dynamics": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_same_version_witness_dynamics",
                            float("nan"),
                        )
                    )
                ),
                "dp_same_version_witness_reward": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_same_version_witness_reward",
                            float("nan"),
                        )
                    )
                ),
                "dp_pending_persistence_score": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_pending_persistence_score",
                            float("nan"),
                        )
                    )
                ),
                "dp_pending_persistence_threshold": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_pending_persistence_threshold",
                            float("nan"),
                        )
                    )
                ),
                "dp_pending_persistence_dynamics": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_pending_persistence_dynamics",
                            float("nan"),
                        )
                    )
                ),
                "dp_pending_persistence_reward": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_pending_persistence_reward",
                            float("nan"),
                        )
                    )
                ),
                "dp_pending_persistence_required": torch.tensor(
                    float(
                        bool(
                            getattr(
                                self, "_m3wm_dp_pending_persistence_required", False
                            )
                        )
                    )
                ),
                "dp_physical_response_departure_enabled": torch.tensor(0.0),
                "dp_physical_witness_shadow_only": torch.tensor(1.0),
                "dp_physical_witness_valid": torch.tensor(
                    float(
                        bool(
                            getattr(self, "_m3wm_last_dp_physical_witness_valid", False)
                        )
                    )
                ),
                "dp_physical_witness_pass": torch.tensor(
                    float(
                        bool(
                            getattr(self, "_m3wm_last_dp_physical_witness_pass", False)
                        )
                    )
                ),
                "dp_physical_witness_score": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_last_dp_physical_witness_score", float("nan")
                        )
                    )
                ),
                "dp_physical_witness_threshold": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_physical_witness_threshold",
                            float("nan"),
                        )
                    )
                ),
                "dp_physical_witness_support": torch.tensor(
                    float(getattr(self, "_m3wm_last_dp_physical_witness_support", 0))
                ),
                "dp_physical_witness_required": torch.tensor(
                    float(getattr(self, "_m3wm_last_dp_physical_witness_required", 0))
                ),
                "dp_physical_witness_mutual_score": torch.tensor(
                    float(
                        getattr(
                            self,
                            "_m3wm_last_dp_physical_witness_mutual_score",
                            float("nan"),
                        )
                    )
                ),
                "dp_physical_witness_mutual_support": torch.tensor(
                    float(
                        getattr(
                            self, "_m3wm_last_dp_physical_witness_mutual_support", 0
                        )
                    )
                ),
                "dp_physical_witness_min_rank": torch.tensor(
                    float(getattr(self, "_m3wm_last_dp_physical_witness_min_rank", 0))
                ),
                "dp_physical_response_departure_used": torch.tensor(
                    float(
                        bool(
                            getattr(
                                self, "_m3wm_last_dp_physical_departure_used", False
                            )
                        )
                    )
                ),
                "dp_prototype_updates": torch.tensor(
                    float(self._m3wm_dp_prototype_updates)
                ),
                "regime_last_dz": torch.tensor(float(self._m3wm_regime_last_dz)),
                "regime_last_s_post": torch.tensor(
                    float(self._m3wm_regime_last_s_post)
                ),
                "context_creation_count": torch.tensor(
                    float(self._m3wm_context_creations)
                ),
                "context_recall_count": torch.tensor(float(self._m3wm_context_recalls)),
            }
        )
        metrics.update(self._m3wm_source_theta_shadow_metrics())
        metrics.update(self._m3wm_active_probe_metrics())
        for slot_index in range(4):
            slot_count = float("nan")
            slot_feature_0 = float("nan")
            metrics[f"memory_slot_{slot_index}_count"] = torch.tensor(slot_count)
            metrics[f"memory_slot_{slot_index}_feature_0"] = torch.tensor(
                slot_feature_0
            )
        trial_controller = getattr(self, "_m3wm_reversible_reuse_trial", None)
        trial_metrics = (
            trial_controller.metrics()
            if trial_controller is not None
            else {
                "active": False,
                "status_code": 0,
                "rho": 0.0,
                "steps": 0,
                "control_noninferior_blocks": 0,
                "control_damaging_blocks": 0,
                "positive_blocks": 0,
                "negative_blocks": 0,
                "current_block_samples": 0,
                "commit_ready": False,
                "last_shadow_advantage": None,
                "last_reward_delta": None,
                "baseline_reward": None,
                "baseline_reward_sample_count": 0,
                "last_block_advantage": None,
                "last_block_reward_delta": None,
                "abort_count": 0,
                "commit_count": 0,
                "nomination_count": 0,
                "last_event": "disabled",
            }
        )

        def _trial_metric_float(name):
            value = trial_metrics.get(name)
            try:
                return float("nan") if value is None else float(value)
            except (TypeError, ValueError, OverflowError):
                return float("nan")

        metrics.update(
            {
                "reversible_reuse_trial_enabled": torch.tensor(0.0),
                "reversible_reuse_trial_active": torch.tensor(
                    float(bool(trial_metrics.get("active", False)))
                ),
                "reversible_reuse_trial_status_code": torch.tensor(
                    float(trial_metrics.get("status_code", 0))
                ),
                "reversible_reuse_trial_source_context": torch.tensor(
                    float(
                        -1
                        if trial_metrics.get("source_context_id") is None
                        else trial_metrics.get("source_context_id")
                    )
                ),
                "reversible_reuse_trial_target_context": torch.tensor(
                    float(
                        -1
                        if trial_metrics.get("target_context_id") is None
                        else trial_metrics.get("target_context_id")
                    )
                ),
                "reversible_reuse_trial_rho": torch.tensor(_trial_metric_float("rho")),
                "reversible_reuse_trial_steps": torch.tensor(
                    float(trial_metrics.get("steps", 0))
                ),
                "reversible_reuse_trial_control_noninferior_blocks": torch.tensor(
                    float(trial_metrics.get("control_noninferior_blocks", 0))
                ),
                "reversible_reuse_trial_control_damaging_blocks": torch.tensor(
                    float(trial_metrics.get("control_damaging_blocks", 0))
                ),
                "reversible_reuse_trial_positive_blocks": torch.tensor(
                    float(trial_metrics.get("positive_blocks", 0))
                ),
                "reversible_reuse_trial_negative_blocks": torch.tensor(
                    float(trial_metrics.get("negative_blocks", 0))
                ),
                "reversible_reuse_trial_current_block_samples": torch.tensor(
                    float(trial_metrics.get("current_block_samples", 0))
                ),
                "reversible_reuse_trial_commit_ready": torch.tensor(
                    float(bool(trial_metrics.get("commit_ready", False)))
                ),
                "reversible_reuse_trial_last_shadow_advantage": torch.tensor(
                    _trial_metric_float("last_shadow_advantage")
                ),
                "reversible_reuse_trial_last_reward_delta": torch.tensor(
                    _trial_metric_float("last_reward_delta")
                ),
                "reversible_reuse_trial_baseline_reward": torch.tensor(
                    _trial_metric_float("baseline_reward")
                ),
                "reversible_reuse_trial_baseline_reward_sample_count": torch.tensor(
                    float(trial_metrics.get("baseline_reward_sample_count", 0))
                ),
                "reversible_reuse_trial_last_block_advantage": torch.tensor(
                    _trial_metric_float("last_block_advantage")
                ),
                "reversible_reuse_trial_last_block_reward_delta": torch.tensor(
                    _trial_metric_float("last_block_reward_delta")
                ),
                "reversible_reuse_trial_abort_count": torch.tensor(
                    float(trial_metrics.get("abort_count", 0))
                ),
                "reversible_reuse_trial_commit_count": torch.tensor(
                    float(trial_metrics.get("commit_count", 0))
                ),
                "reversible_reuse_trial_nomination_count": torch.tensor(
                    float(trial_metrics.get("nomination_count", 0))
                ),
            }
        )
        return metrics

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        torch.save(
            {
                "schema": "m3wm-td-learner-v1",
                "model": self.model.state_dict(),
                "training_state": self.export_train_state(),
            },
            fp,
        )

    def export_train_state(self):
        from common.m3wm_context_agent_bundle import _capture_agent

        generators = {}
        for name in ("_plain_incumbent_score_generator", "proposal_generator"):
            generator = getattr(self, name, None)
            if isinstance(generator, torch.Generator):
                generators[name] = generator
        return _capture_agent(self, generators=generators)

    def import_train_state(self, state, *, restore_rng=True):
        from common.m3wm_context_agent_bundle import _restore_agent

        generators = {}
        for name in ("_plain_incumbent_score_generator", "proposal_generator"):
            generator = getattr(self, name, None)
            if isinstance(generator, torch.Generator):
                generators[name] = generator
        _restore_agent(self, state, generators=generators, restore_rng=restore_rng)

    def load(self, fp):
        payload = (
            fp
            if isinstance(fp, dict)
            else torch.load(fp, map_location=self.device, weights_only=False)
        )
        if payload.get("schema") != "m3wm-td-learner-v1":
            raise ValueError("Unsupported learner checkpoint schema")
        self.import_train_state(payload["training_state"])

    @torch.inference_mode()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        self._activate_runtime_theta()
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        action = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
        return action

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        G, discount = (0, 1)
        termination = torch.zeros(
            self.cfg.num_samples, 1, dtype=torch.float32, device=z.device
        )
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * (1 - termination) * reward
            discount_update = self.discount
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(
                    termination + (self.model.termination(z, task) > 0.5).float(),
                    max=1.0,
                )
        action, _ = self.model.pi(z, task)
        terminal_value = self.model.Q(z, action, task, return_type="avg")
        return G + discount * (1 - termination) * terminal_value

    def consume_plain_incumbent_plan_result(self):
        result = self._plain_incumbent_plan_result
        self._plain_incumbent_plan_result = None
        return result

    def consume_guarded_control_frame(self):
        frame = getattr(self, "_guarded_control_frame", None)
        self._guarded_control_frame = None
        return frame

    def proposal_metrics(self):
        return dict(self._plain_incumbent_proposal_metrics)

    def _synchronize_plain_proposal_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _validate_plain_q_score_shape(self, scores, batch_size):
        num_q = int(self.cfg.num_q)
        if num_q < 1:
            raise RuntimeError("Plain num_q must be positive")
        expected = (num_q, int(batch_size), 1)
        if tuple(scores.shape) != expected:
            raise RuntimeError(
                f"Plain all-head scorer shape mismatch: configured num_q={num_q}, expected={expected}, received={tuple(scores.shape)}"
            )
        return expected

    @torch.no_grad()
    def _proposal_tail_action(self, z, task, epsilon):
        actor_z = z
        mean, log_std = self.model._pi(actor_z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.model.log_std_min, self.model.log_std_dif)
        action = mean + epsilon * log_std.exp()
        return torch.tanh(action)

    @torch.no_grad()
    def score_action_sequences_all_heads(
        self, obs, action_sequences, task=None, *, shared_tail_epsilon=None
    ):
        actions = torch.as_tensor(
            action_sequences, device=self.device, dtype=torch.float32
        )
        if actions.ndim != 3:
            raise ValueError(
                "action_sequences must have shape [horizon,batch,action_dim]"
            )
        horizon, batch_size, action_dim = actions.shape
        if horizon != int(self.cfg.horizon) or action_dim != int(self.cfg.action_dim):
            raise ValueError("action sequence shape does not match Plain planner")
        if batch_size < 1 or not bool(torch.isfinite(actions).all()):
            raise ValueError("action sequence batch must be non-empty and finite")
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.shape[0] != 1:
            raise ValueError("paired Plain rescoring requires exactly one observation")
        task_tensor = task
        if task is not None and (not torch.is_tensor(task)):
            task_tensor = torch.tensor([task], device=self.device)
        z = self.model.encode(obs_tensor, task_tensor).repeat(batch_size, 1)
        returns = torch.zeros(batch_size, 1, device=self.device, dtype=z.dtype)
        discount = 1.0
        termination = torch.zeros_like(returns)
        for index in range(horizon):
            reward = math.two_hot_inv(
                self.model.reward(z, actions[index], task_tensor), self.cfg
            )
            weight = 1.0 - termination
            returns = returns + discount * weight * reward
            z = self.model.next(z, actions[index], task_tensor)
            discount_update = self.discount
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(
                    termination
                    + (self.model.termination(z, task_tensor) > 0.5).float(),
                    max=1.0,
                )
        if shared_tail_epsilon is None:
            if self._plain_incumbent_score_generator is None:
                raise RuntimeError("private Plain proposal scorer RNG is unavailable")
            shared_tail_epsilon = torch.randn(
                1,
                int(self.cfg.action_dim),
                device=self.device,
                generator=self._plain_incumbent_score_generator,
            )
        epsilon = torch.as_tensor(
            shared_tail_epsilon, device=self.device, dtype=z.dtype
        )
        if tuple(epsilon.shape) != (1, int(self.cfg.action_dim)):
            raise ValueError("shared_tail_epsilon has an invalid shape")
        self._guarded_last_tail_epsilon = epsilon.detach().clone()
        epsilon = epsilon.expand(batch_size, -1)
        terminal_action = self._proposal_tail_action(z, task_tensor, epsilon)
        q_logits = self.model.Q(z, terminal_action, task_tensor, return_type="all")
        q_values = math.two_hot_inv(q_logits, self.cfg)
        self._validate_plain_q_score_shape(q_values, batch_size)
        weight = 1.0 - termination
        scores = returns.unsqueeze(0) + discount * weight.unsqueeze(0) * q_values
        self._validate_plain_q_score_shape(scores, batch_size)
        return scores

    @torch.no_grad()
    def rank_plain_extra_search_candidates(self, obs, action_sequences, task=None):
        actions = torch.as_tensor(
            action_sequences, device=self.device, dtype=torch.float32
        )
        if actions.ndim != 3:
            raise ValueError("Plain extra-search candidates must have shape [H,B,A]")
        batch_size = int(actions.shape[1])
        if batch_size != 16:
            raise ValueError("Plain extra-search candidate budget must equal 16")
        scores = self.score_action_sequences_all_heads(obs, actions, task=task)
        self._validate_plain_q_score_shape(scores, batch_size)
        scores_cpu = scores[:, :, 0].detach().cpu()
        if not bool(torch.isfinite(scores_cpu).all()):
            raise FloatingPointError("Plain extra-search score is non-finite")
        candidate_min_scores = scores_cpu.amin(dim=0)
        return int(torch.argmax(candidate_min_scores).item())

    @torch.no_grad()
    def select_plain_incumbent_or_proposal(
        self, obs, plain_result, proposal, *, control_step=None
    ):
        self._guarded_control_frame = {
            "schema": "m3wm_control_frame_v1",
            "status": "unavailable",
            "action_id": int(control_step) if control_step is not None else None,
            "unavailable_reason": "not_scored",
        }
        metrics = {
            "proposal_enabled": 1,
            "proposal_valid": 0,
            "proposal_accepted": 0,
            "proposal_rejected_reason": "disabled",
            "proposal_context_id": -1,
            "proposal_theta_version": -1,
            "proposal_model_version": -1,
            "proposal_count": 0,
            "proposal_control_step": -1,
            "proposal_last_accept_step": int(
                self._plain_incumbent_last_accept_step
                if self._plain_incumbent_last_accept_step > -(10**17)
                else -1
            ),
            "paired_plain_min_delta": float("nan"),
            "paired_plain_score_delta_per_head": "[]",
            "plain_q_head_count": int(self.cfg.num_q),
            "plain_q_head_indices": json.dumps(list(range(int(self.cfg.num_q)))),
            "paired_plain_score_shape": "[]",
            "proposal_plain_rescore_ms": 0.0,
            "proposal_mode": "OFF",
            "proposal_ranking_source": "disabled",
            "proposal_selected_index": -1,
            "proposal_used_recall": 0,
            "proposal_theta_current_linf_delta": float("nan"),
            "executed_action_source": "plain",
        }
        if plain_result is None:
            self._guarded_control_frame["unavailable_reason"] = (
                "no_plain_result" if plain_result is None else "proposal_disabled"
            )
            self._plain_incumbent_proposal_metrics = metrics
            return None if plain_result is None else plain_result["action"].clone()
        incumbent_action = plain_result["action"].detach().clone()
        if not isinstance(proposal, Mapping):
            self._guarded_control_frame["unavailable_reason"] = "no_proposal"
            metrics["proposal_rejected_reason"] = "no_proposal"
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        metrics.update(
            {
                "proposal_context_id": int(proposal.get("context_id", -1)),
                "proposal_theta_version": int(proposal.get("theta_version", -1)),
                "proposal_model_version": int(proposal.get("model_version", -1)),
                "proposal_count": int(proposal.get("proposal_count", 0)),
                "proposal_mode": str(proposal.get("mode", "UNKNOWN")),
                "proposal_ranking_source": str(
                    proposal.get("ranking_source", "unknown")
                ),
                "proposal_selected_index": int(proposal.get("selected_index", -1)),
                "proposal_used_recall": int(proposal.get("used_recall", 0)),
                "proposal_theta_current_linf_delta": float(
                    proposal.get("theta_current_linf_delta", float("nan"))
                ),
                "proposal_control_step": int(proposal.get("control_step", -1)),
                "proposal_last_accept_step": int(
                    self._plain_incumbent_last_accept_step
                    if self._plain_incumbent_last_accept_step > -(10**17)
                    else -1
                ),
            }
        )
        control_step = int(proposal.get("control_step", -1))
        if control_step < 0:
            self._guarded_control_frame["unavailable_reason"] = "missing_control_step"
            metrics["proposal_rejected_reason"] = "missing_control_step"
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        if control_step <= self._plain_incumbent_proposal_warmup_steps:
            self._guarded_control_frame["unavailable_reason"] = "early_warmup"
            metrics["proposal_rejected_reason"] = "early_warmup"
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        adoption_blocked = False
        if (
            control_step - self._plain_incumbent_last_accept_step
            < self._plain_incumbent_proposal_min_accept_interval
        ):
            if not bool(getattr(self, "_guarded_store_witness_pending", False)):
                self._guarded_control_frame["unavailable_reason"] = "adoption_interval"
                metrics["proposal_rejected_reason"] = "adoption_interval"
                self._plain_incumbent_proposal_metrics = metrics
                return incumbent_action
            adoption_blocked = True
        candidate = proposal.get("action_sequence")
        if candidate is None:
            self._guarded_control_frame["unavailable_reason"] = str(
                proposal.get("reason", "no_proposal")
            )
            metrics["proposal_rejected_reason"] = str(
                proposal.get("reason", "no_proposal")
            )
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        candidate = torch.as_tensor(candidate, device=self.device, dtype=torch.float32)
        expected_shape = (int(self.cfg.horizon), int(self.cfg.action_dim))
        if tuple(candidate.shape) != expected_shape:
            self._guarded_control_frame["unavailable_reason"] = "invalid_shape"
            metrics["proposal_rejected_reason"] = "invalid_shape"
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        candidate_checks = (
            torch.stack(
                (torch.isfinite(candidate).all(), (candidate.abs() <= 1.0).all())
            )
            .detach()
            .cpu()
        )
        if not bool(candidate_checks[0]):
            self._guarded_control_frame["unavailable_reason"] = "non_finite_candidate"
            metrics["proposal_rejected_reason"] = "non_finite_candidate"
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        if not bool(candidate_checks[1]):
            self._guarded_control_frame["unavailable_reason"] = (
                "out_of_bounds_candidate"
            )
            metrics["proposal_rejected_reason"] = "out_of_bounds_candidate"
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        metrics["proposal_valid"] = 1
        incumbent = plain_result["action_sequence"].to(
            device=self.device, dtype=candidate.dtype
        )
        paired = torch.stack((incumbent, candidate), dim=1)
        self._synchronize_plain_proposal_device()
        rescore_started = time.perf_counter()
        witness_rng_state = None
        witness_rng_isolated = False
        if adoption_blocked and self._plain_incumbent_score_generator is not None:
            witness_rng_state = (
                self._plain_incumbent_score_generator.get_state().clone()
            )
        try:
            scores = self.score_action_sequences_all_heads(obs, paired)
            self._validate_plain_q_score_shape(scores, 2)
            delta = scores[:, 1, 0] - scores[:, 0, 0]
        except Exception as error:
            self._synchronize_plain_proposal_device()
            metrics["proposal_plain_rescore_ms"] = (
                time.perf_counter() - rescore_started
            ) * 1000.0
            metrics["proposal_rejected_reason"] = f"scorer_error:{type(error).__name__}"
            self._guarded_control_frame["unavailable_reason"] = metrics[
                "proposal_rejected_reason"
            ]
            self._plain_incumbent_proposal_metrics = metrics
            return incumbent_action
        finally:
            if witness_rng_state is not None:
                self._plain_incumbent_score_generator.set_state(witness_rng_state)
                witness_rng_isolated = True
        self._synchronize_plain_proposal_device()
        metrics["proposal_plain_rescore_ms"] = (
            time.perf_counter() - rescore_started
        ) * 1000.0
        metrics["paired_plain_score_shape"] = json.dumps(list(scores.shape))
        scores_cpu = scores.detach().cpu()
        delta_cpu = delta.detach().cpu()
        finite = bool(
            torch.isfinite(scores_cpu).all() and torch.isfinite(delta_cpu).all()
        )
        delta_values = delta_cpu.tolist()
        metrics["paired_plain_score_delta_per_head"] = json.dumps(delta_values)
        metrics["paired_plain_min_delta"] = (
            float(delta_cpu.min().item()) if finite else float("nan")
        )
        if not finite:
            metrics["proposal_rejected_reason"] = "non_finite_score"
        elif not adoption_blocked and bool(
            (delta_cpu > self._plain_incumbent_proposal_tolerance).all()
        ):
            metrics["proposal_accepted"] = 1
            metrics["proposal_rejected_reason"] = ""
            metrics["executed_action_source"] = "m3wm_proposal"
            incumbent_action = candidate[0].detach().cpu().clone()
            self._plain_incumbent_last_accept_step = control_step
            metrics["proposal_last_accept_step"] = control_step
        else:
            metrics["proposal_rejected_reason"] = (
                "adoption_interval"
                if adoption_blocked
                else "not_strictly_better_all_heads"
            )
        raw_shared_candidates = proposal.get("shared_candidates")
        if raw_shared_candidates is None:
            raw_shared_candidates = candidate.unsqueeze(1)
        shared_candidates = (
            torch.as_tensor(
                raw_shared_candidates, device=candidate.device, dtype=candidate.dtype
            )
            .detach()
            .cpu()
            .clone()
        )
        try:
            discount_value = float(self.discount)
        except (TypeError, ValueError):
            discount_value = float(torch.as_tensor(self.discount).reshape(-1)[0].item())
        self._guarded_control_frame = {
            "schema": "m3wm_control_frame_v1",
            "status": "available",
            "action_id": int(control_step),
            "u0_sequence": incumbent.detach().cpu().clone(),
            "up_sequence": candidate.detach().cpu().clone(),
            "uc_sequence": (
                candidate.detach().cpu().clone()
                if bool(metrics["proposal_accepted"])
                else incumbent.detach().cpu().clone()
            ),
            "saved_current_action": incumbent_action.detach().cpu().clone(),
            "shared_candidates": shared_candidates,
            "scores": scores_cpu.clone(),
            "shared_tail_epsilon": self._guarded_last_tail_epsilon.detach()
            .cpu()
            .clone(),
            "discount": discount_value,
            "gate_tolerance": float(self._plain_incumbent_proposal_tolerance),
            "proposal_valid": True,
            "q_pass": bool(
                (delta_cpu > self._plain_incumbent_proposal_tolerance).all()
            ),
            "current_accepted": bool(metrics["proposal_accepted"]),
            "proposal_rejected_reason": metrics["proposal_rejected_reason"],
            "plain_model_version": int(getattr(self, "_m3wm_train_updates", 0)),
            "witness_rng_isolated": bool(witness_rng_isolated),
        }
        self._plain_incumbent_proposal_metrics = metrics
        return incumbent_action

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        eval_action_mode = _validated_m3wm_eval_mpc_action_mode(
            getattr(
                self,
                "_m3wm_eval_mpc_action_mode",
                self.cfg.get("m3wm_eval_mpc_action_mode", "sample_elite"),
            )
        )
        encoded_z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = encoded_z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)
        z = encoded_z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, : self.cfg.num_pi_trajs] = pi_actions
        for _ in range(self.cfg.iterations):
            r = torch.randn(
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs :] = actions_sample
            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value, elite_actions = (value[elite_idxs], actions[:, elite_idxs])
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                score.sum(0) + 1e-09
            )
            std = (
                (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(
                    dim=1
                )
                / (score.sum(0) + 1e-09)
            ).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a = mean[0] if eval_mode and eval_action_mode == "weighted_mean" else actions[0]
        final_mean = mean.detach().clone()
        final_std = std.detach().clone()
        std = std[0]
        if not eval_mode:
            raw_execution_noise_scale = self.cfg.get("m3wm_execution_noise_scale", 1.0)
            try:
                execution_noise_scale = float(raw_execution_noise_scale)
            except (TypeError, ValueError):
                raise ValueError(
                    "m3wm_execution_noise_scale must be a finite non-negative scalar"
                )
            if not np.isfinite(execution_noise_scale) or execution_noise_scale < 0.0:
                raise ValueError(
                    "m3wm_execution_noise_scale must be a finite non-negative scalar"
                )
            if execution_noise_scale > 0.0:
                a = a + execution_noise_scale * std * torch.randn(
                    self.cfg.action_dim, device=std.device
                )
        self._prev_mean.copy_(mean)
        a = a.clamp(-1, 1)
        incumbent_sequence = (
            final_mean.clone()
            if eval_mode and eval_action_mode == "weighted_mean"
            else actions.detach().clone()
        )
        incumbent_sequence[0] = a
        self._plain_incumbent_plan_result = {
            "action": a.detach().cpu().clone(),
            "action_sequence": incumbent_sequence.detach().clone(),
            "proposal_mean": final_mean.detach().clone(),
            "proposal_std": final_std.detach().clone(),
        }
        return a

    def _m3wm_actor_anchor_input(self, zs, task):
        return zs

    @torch.no_grad()
    def _m3wm_update_actor_ema(self):
        if self._m3wm_actor_ema is None:
            return
        tau = float(self._m3wm_actor_ema_tau)
        if tau <= 0.0:
            return
        for ema_parameter, parameter in zip(
            self._m3wm_actor_ema.parameters(), self.model._pi.parameters()
        ):
            ema_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def _m3wm_record_pi_diagnostics(self, *, applied, q_std, ema_penalty):
        self._m3wm_last_pi_update_applied = float(applied)
        self._m3wm_last_pi_q_std = float(
            q_std.detach().mean().item() if torch.is_tensor(q_std) else q_std
        )
        self._m3wm_last_pi_ema_penalty = float(
            ema_penalty.detach().mean().item()
            if torch.is_tensor(ema_penalty)
            else ema_penalty
        )

    def update_pi(self, zs, task, theta=None, valid_mask=None):
        self._m3wm_control_update_count += 1
        action, info = self.model.pi(zs, task)
        if self._m3wm_conservative_q_beta > 0.0:
            q_logits = self.model.Q(
                zs, action, task, return_type="all", detach=True, theta=theta
            )
            q_values = math.two_hot_inv(q_logits, self.cfg)
            q_std = q_values.std(dim=0, unbiased=False)
            qs = q_values.mean(dim=0)
        else:
            qs = self.model.Q(
                zs, action, task, return_type="avg", detach=True, theta=theta
            )
            q_std = torch.zeros_like(qs)
        state_mask = None
        if valid_mask is not None:
            state_mask = valid_mask.to(device=qs.device, dtype=torch.bool)
            if state_mask.ndim == 3 and state_mask.shape[-1] == 1:
                state_mask = state_mask.squeeze(-1)
            if state_mask.ndim != 2 or tuple(state_mask.shape) != tuple(qs.shape[:2]):
                raise ValueError(
                    f"policy valid_mask must have shape [time,batch], received {tuple(state_mask.shape)} for Q {tuple(qs.shape)}"
                )
            if bool(state_mask[0].any()):
                self.scale.update(qs[0][state_mask[0]])
        else:
            self.scale.update(qs[0])
        qs = self.scale(qs)
        if self._m3wm_conservative_q_beta > 0.0:
            qs = qs - float(self._m3wm_conservative_q_beta) * self.scale(q_std)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        per_step = self.cfg.entropy_coef * info["scaled_entropy"] + qs
        ema_penalty = torch.zeros_like(per_step)
        if self._m3wm_actor_ema is not None and self._m3wm_actor_ema_coef > 0.0:
            anchor_input = self._m3wm_actor_anchor_input(zs, task)
            with torch.no_grad():
                ema_mean = torch.tanh(
                    self._m3wm_actor_ema(anchor_input).chunk(2, dim=-1)[0]
                )
            ema_penalty = (info["mean"] - ema_mean).pow(2).mean(dim=-1, keepdim=True)
        if state_mask is None:
            per_step = per_step.mean(dim=(1, 2))
            ema_penalty = ema_penalty.mean(dim=(1, 2))
        else:
            weights = state_mask.to(dtype=per_step.dtype).unsqueeze(-1)
            per_step = (per_step * weights).sum(dim=(1, 2)) / weights.sum(
                dim=(1, 2)
            ).clamp_min(1.0)
            ema_penalty = (ema_penalty * weights).sum(dim=(1, 2)) / weights.sum(
                dim=(1, 2)
            ).clamp_min(1.0)
        pi_loss = (-per_step + float(self._m3wm_actor_ema_coef) * ema_penalty) * rho
        pi_loss = pi_loss.mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)
        self._m3wm_update_actor_ema()
        self._m3wm_record_pi_diagnostics(
            applied=1.0, q_std=q_std, ema_penalty=ema_penalty
        )
        info = {
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
            "pi_update_applied": torch.ones((), device=self.device),
            "pi_q_std": q_std,
            "pi_ema_penalty": ema_penalty,
        }
        return info

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated, task, theta=None):
        action, _ = self.model.pi(next_z, task)
        discount = self.discount
        return reward + discount * (1 - terminated) * self.model.Q(
            next_z, action, task, return_type="min", target=True, theta=theta
        )

    def _update(
        self,
        obs,
        action,
        reward,
        terminated,
        task=None,
        context_severity=None,
        valid_mask=None,
    ):
        enable_m3wm = bool(self.cfg.get("enable_m3wm", False))
        self._m3wm_pure_online_last_update_decision = None
        context_supervision_loss = torch.zeros(
            (), device=obs.device, dtype=torch.float32
        )
        context_supervision_within = context_supervision_loss
        context_supervision_distance = context_supervision_loss
        context_supervision_valid_fraction = context_supervision_loss
        system_id_routing_loss = context_supervision_loss
        if enable_m3wm:
            self._apply_m3wm_training_schedule()
        context_len = int(self.cfg.get("m3wm_context_window", 8)) if enable_m3wm else 0
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=obs.device, dtype=torch.bool)
            if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
            if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(obs.shape[:2]):
                raise ValueError(
                    f"update valid_mask must have shape [time,batch], received {tuple(valid_mask.shape)} for obs {tuple(obs.shape)}"
                )
            if bool(valid_mask.all()):
                valid_mask = None
        transition_valid = None
        if valid_mask is not None:
            transition_valid = valid_mask[:-1] & valid_mask[1:]
        terminal_tail_mask = None
        transition_training_weight = None

        def _masked_mean(value, mask):
            if mask is None:
                return value.mean()
            mask = mask.to(device=value.device, dtype=value.dtype)
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            return (value * mask).sum() / mask.sum().clamp_min(1.0)

        def _transition_loss_mask(index):
            if transition_training_weight is not None:
                return transition_training_weight[index]
            if transition_valid is not None:
                return transition_valid[index]
            return None

        if enable_m3wm:
            expected = context_len + self.cfg.horizon + 1
            if obs.shape[0] != expected:
                raise RuntimeError(
                    f"ReCo-WM expected replay length {expected}, but received {obs.shape[0]}."
                )
            with torch.no_grad():
                context_obs = obs[: context_len + 1]
                context_features = self.model.build_context_features_from_observations(
                    context_obs[:-1],
                    action[:context_len],
                    reward[:context_len],
                    context_obs[1:],
                    task=task,
                )
            context_valid = (
                None if transition_valid is None else transition_valid[:context_len]
            )
            if terminal_tail_mask is not None:
                terminal_context_valid = ~terminal_tail_mask[:context_len]
                context_valid = (
                    terminal_context_valid
                    if context_valid is None
                    else context_valid & terminal_context_valid
                )
            self.model._m3wm_context.train()
            all_theta = self.model.context_theta_from_features(
                context_features, valid_mask=context_valid
            )
            supervision_target = str(
                self.cfg.get("m3wm_context_supervision_target", "theta")
            ).lower()
            if supervision_target not in {"theta", "embedding"}:
                raise ValueError(
                    "m3wm_context_supervision_target must be 'theta' or 'embedding'"
                )
            context_embedding = (
                self.model.context_embedding_from_features(
                    context_features, valid_mask=context_valid
                )
                if supervision_target == "embedding"
                else None
            )
            if not self._m3wm_specialization_active:
                all_theta = self.model.uniform_theta(context_features)
            theta_context = all_theta[-1]
            theta = theta_context.unsqueeze(0).expand(
                self.cfg.horizon,
                -1,
                *self.model.uniform_theta(context_features[-1]).shape[1:],
            )
            system_id_routing_loss = theta_context.new_zeros(())
            routing_coef = float(self.cfg.get("m3wm_system_id_routing_coef", 0.0))
            if (
                self._m3wm_specialization_active
                and routing_coef > 0
                and self.model.m3wm_separate_composition_heads
                and (self.cfg.obs == "state")
                and (context_obs.ndim == 3)
                and (context_valid is None or bool(context_valid.all()))
            ):
                with torch.no_grad():
                    system_id_fn = batched_conditioned_action_response_log_norm
                    system_id_args = (
                        (
                            context_obs[:-1],
                            action[:context_len],
                            context_obs[1:] - context_obs[:-1],
                        )
                        if system_id_fn is batched_conditioned_action_response_log_norm
                        else (action[:context_len], context_obs[1:] - context_obs[:-1])
                    )
                    signatures = system_id_fn(
                        *system_id_args,
                        ridge=float(self.cfg.get("m3wm_system_id_ridge", 0.01)),
                        **(
                            {
                                "min_std": float(
                                    self.cfg.get("m3wm_system_id_min_std", 0.001)
                                )
                            }
                            if system_id_fn
                            is batched_conditioned_action_response_log_norm
                            else {}
                        ),
                    )
                    routing_targets = system_id_routing_targets(
                        signatures,
                        self.model.m3wm_num_primitives,
                        temperature=float(
                            self.cfg.get("m3wm_system_id_target_temperature", 0.5)
                        ),
                        min_std=float(
                            self.cfg.get("m3wm_system_id_target_min_std", 0.05)
                        ),
                    )
                if routing_targets is not None:
                    dynamics_theta = theta_context[..., 0, :]
                    system_id_routing_loss = (
                        -(routing_targets * dynamics_theta.clamp_min(1e-08).log())
                        .sum(dim=-1)
                        .mean()
                    )
            supervision_coef = float(self.cfg.get("m3wm_context_supervision_coef", 0.0))
            if (
                self._m3wm_specialization_active
                and supervision_coef > 0
                and (context_severity is not None)
            ):
                support_severity = context_severity[:context_len]
                if support_severity.ndim == 3 and support_severity.shape[-1] == 1:
                    support_severity = support_severity.squeeze(-1)
                if support_severity.ndim != 2:
                    raise RuntimeError(
                        "ReCo-WM context severity must have shape [time, batch]"
                    )
                finite = torch.isfinite(support_severity).all(dim=0)
                if context_valid is not None:
                    finite = finite & context_valid.all(dim=0)
                safe_severity = torch.where(
                    torch.isfinite(support_severity),
                    support_severity,
                    torch.zeros_like(support_severity),
                )
                stability_tolerance = float(
                    self.cfg.get("m3wm_context_label_stability_tolerance", 0.001)
                )
                stable = (
                    safe_severity.amax(dim=0) - safe_severity.amin(dim=0)
                    <= stability_tolerance
                )
                valid = finite & stable
                (
                    context_supervision_loss,
                    context_supervision_within,
                    context_supervision_distance,
                ) = supervised_context_separation_loss(
                    (
                        context_embedding[-1]
                        if context_embedding is not None
                        else theta_context
                    ),
                    safe_severity[-1],
                    valid=valid,
                    label_threshold=float(
                        self.cfg.get("m3wm_context_label_threshold", 0.1)
                    ),
                    margin=float(self.cfg.get("m3wm_context_supervision_margin", 0.75)),
                    min_samples_per_context=int(
                        self.cfg.get("m3wm_context_supervision_min_samples", 2)
                    ),
                    multiclass=False,
                    label_tolerance=float(
                        self.cfg.get("m3wm_context_label_tolerance", 0.05)
                    ),
                )
                context_supervision_valid_fraction = valid.float().mean()
            reg_len = min(
                context_len, int(self.cfg.get("m3wm_context_short_window", 8))
            )
            theta_for_regularization = all_theta[-reg_len:]
            obs = obs[context_len:]
            action = action[context_len:]
            reward = reward[context_len:]
            terminated = terminated[context_len:]
            if terminal_tail_mask is not None:
                terminal_tail_mask = terminal_tail_mask[context_len:]
            if transition_training_weight is not None:
                transition_training_weight = transition_training_weight[context_len:]
            if transition_valid is not None:
                transition_valid = transition_valid[context_len:]
            if valid_mask is not None:
                state_valid = valid_mask[context_len:]
            else:
                state_valid = None
        else:
            theta = None
            theta_context = None
            theta_for_regularization = None
            state_valid = valid_mask if valid_mask is not None else None
        if terminal_tail_mask is not None:
            terminal_state_valid = torch.cat(
                [
                    ~terminal_tail_mask,
                    torch.ones(
                        (1, terminal_tail_mask.shape[1]),
                        device=terminal_tail_mask.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=0,
            )
            state_valid = (
                terminal_state_valid
                if state_valid is None
                else state_valid & terminal_state_valid
            )
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(
                next_z, reward, terminated, task, theta=theta_context
            )
        if os.environ.get("M3WM_DEBUG_FINITE_CHECKS", "0") == "1":
            for _name, _value in (
                ("obs", obs),
                ("action", action),
                ("reward", reward),
                ("terminated", terminated),
                ("next_z", next_z),
                ("td_targets", td_targets),
            ):
                if not bool(torch.isfinite(_value).all()):
                    _bad = ~torch.isfinite(_value)
                    _bad_count = int(_bad.sum().item())
                    _bad_positions = (
                        _bad.nonzero(as_tuple=False)[:8].detach().cpu().tolist()
                    )
                    raise RuntimeError(
                        f"non-finite {_name} reached TD-MPC2 update; count={_bad_count}, positions={_bad_positions}, valid_mask_present={valid_mask is not None}."
                    )
        self.model.train()
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            theta_t = None if theta is None else theta[t]
            z = self.model.next(z, _action, task, theta=theta_t)
            consistency_error = F.mse_loss(z, _next_z, reduction="none").mean(dim=-1)
            consistency_loss = (
                consistency_loss
                + _masked_mean(consistency_error, _transition_loss_mask(t))
                * self.cfg.rho**t
            )
            zs[t + 1] = z
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all", theta=theta_context)
        reward_preds = self.model.reward(_zs, action, task, theta=theta)
        if self.cfg.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)
        reward_loss, value_loss = (0, 0)
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
            zip(
                reward_preds.unbind(0),
                reward.unbind(0),
                td_targets.unbind(0),
                qs.unbind(1),
            )
        ):
            reward_loss = (
                reward_loss
                + _masked_mean(
                    math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg),
                    _transition_loss_mask(t),
                )
                * self.cfg.rho**t
            )
            for qs_unbind_unbind in qs_unbind.unbind(0):
                value_loss = (
                    value_loss
                    + _masked_mean(
                        math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg),
                        _transition_loss_mask(t),
                    )
                    * self.cfg.rho**t
                )
        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        if self.cfg.episodic:
            termination_loss = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    termination_pred, terminated, reduction="none"
                ),
                transition_valid,
            )
        else:
            termination_loss = 0.0
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        if enable_m3wm:
            theta_balance_loss, theta_smooth_loss, theta_entropy = (
                self.model.context_regularization(
                    theta_for_regularization,
                    valid_mask=(
                        None
                        if context_valid is None
                        else context_valid[-theta_for_regularization.shape[0] :]
                    ),
                )
            )
            if self._m3wm_specialization_active:
                primitive_diversity_loss = self.model.primitive_diversity_loss(
                    _zs.detach(), action, task, valid_mask=transition_valid
                )
            else:
                primitive_diversity_loss = consistency_loss.new_zeros(())
        else:
            zero = consistency_loss.new_zeros(())
            theta_balance_loss = zero
            theta_smooth_loss = zero
            theta_entropy = zero
            primitive_diversity_loss = zero
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.value_coef * value_loss
            + float(self._m3wm_specialization_active)
            * self.cfg.get("m3wm_balance_coef", 0.0)
            * theta_balance_loss
            + float(self._m3wm_specialization_active)
            * self.cfg.get("m3wm_smooth_coef", 0.0)
            * theta_smooth_loss
            + float(self._m3wm_specialization_active)
            * self.cfg.get("m3wm_entropy_coef", 0.0)
            * theta_entropy
            + float(self._m3wm_specialization_active)
            * self.cfg.get("m3wm_diversity_coef", 0.0)
            * primitive_diversity_loss
            + float(self._m3wm_specialization_active)
            * self.cfg.get("m3wm_system_id_routing_coef", 0.0)
            * system_id_routing_loss
            + float(self._m3wm_specialization_active)
            * self.cfg.get("m3wm_context_supervision_coef", 0.0)
            * context_supervision_loss
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        if enable_m3wm:
            self._m3wm_train_updates += 1
        pi_info = self.update_pi(
            zs.detach(),
            task,
            theta=None if theta_context is None else theta_context.detach(),
            valid_mask=state_valid,
        )
        self.model.soft_update_target_Q()
        self.model.eval()
        if terminal_tail_mask is None:
            terminal_tail_fraction = consistency_loss.new_zeros(())
            terminal_training_weight_mean = consistency_loss.new_ones(())
        else:
            terminal_tail_fraction = terminal_tail_mask.float().mean()
            if transition_training_weight is None:
                terminal_training_weight_mean = consistency_loss.new_ones(())
            else:
                terminal_training_weight_mean = transition_training_weight.mean()
        info = {
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "theta_balance_loss": theta_balance_loss,
            "theta_smooth_loss": theta_smooth_loss,
            "theta_train_entropy": theta_entropy,
            "primitive_diversity_loss": primitive_diversity_loss,
            "system_id_routing_loss": system_id_routing_loss,
            "context_supervision_loss": context_supervision_loss,
            "context_supervision_within": context_supervision_within,
            "context_supervision_distance": context_supervision_distance,
            "context_supervision_valid_fraction": context_supervision_valid_fraction,
            "theta_temperature": torch.tensor(
                self._m3wm_current_temperature, device=self.device
            ),
            "base_lr_scale": torch.tensor(
                self._m3wm_current_base_lr_scale, device=self.device
            ),
            "specialization_active": torch.tensor(
                float(self._m3wm_specialization_active), device=self.device
            ),
            "terminal_tail_fraction": terminal_tail_fraction,
            "terminal_training_weight_mean": terminal_training_weight_mean,
        }
        if enable_m3wm:
            mean_theta = theta.detach().mean(dim=(0, 1))
            aggregate_theta = (
                mean_theta.mean(dim=0) if mean_theta.ndim == 2 else mean_theta
            )
            for i, value in enumerate(aggregate_theta):
                info[f"theta_train_{i}"] = value
            if mean_theta.ndim == 2:
                for branch_index, branch_name in enumerate(("dyn", "rew", "val")):
                    for primitive_index, value in enumerate(mean_theta[branch_index]):
                        info[f"theta_train_{branch_name}_{primitive_index}"] = value
        if self.cfg.episodic:
            info.update(
                math.termination_statistics(
                    torch.sigmoid(termination_pred[-1]), terminated[-1]
                )
            )
        info.update(pi_info)
        return {
            key: (
                value.detach().mean()
                if isinstance(value, torch.Tensor)
                else torch.tensor(value)
            )
            for (key, value) in info.items()
        }

    def update(self, buffer):
        batch = buffer.sample()
        valid_mask = None
        if len(batch) == 5:
            obs, action, reward, terminated, task = batch
            context_severity = None
        elif len(batch) == 7:
            obs, action, reward, terminated, task, context_severity, valid_mask = batch
        else:
            obs, action, reward, terminated, task, context_severity = batch
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        if context_severity is not None:
            kwargs["context_severity"] = context_severity
        if valid_mask is not None:
            kwargs["valid_mask"] = valid_mask
        _safe_cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated, **kwargs)
