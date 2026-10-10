from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from time import time
import numpy as np
import torch
from tensordict.tensordict import TensorDict

_CUDAGRAPH_MARK_ENABLED = True


def _safe_cudagraph_mark_step_begin():
    global _CUDAGRAPH_MARK_ENABLED
    return


from common.m3wm_identity_event_artifact import identity_event_payload_sha256
from common.m3wm_episode_control_contract import control_contract
from common.m3wm_pending_control_transfer import PendingControlTransfer
from trainer.base import Trainer


class OnlineTrainer(Trainer):

    @staticmethod
    def _source_theta_shadow_trace_fields():
        fields = [
            "source_theta_shadow_enabled",
            "source_theta_shadow_valid",
            "source_theta_shadow_binding_valid",
            "source_theta_shadow_reason_code",
            "source_theta_shadow_proposal_step",
            "source_theta_shadow_due_step",
            "source_theta_shadow_theta_l1",
            "source_theta_shadow_theta_l2",
            "source_theta_shadow_theta_linf",
            "source_theta_shadow_reward_rescue_enabled",
            "source_theta_shadow_reward_rescue_evaluated",
            "source_theta_shadow_reward_rescue_valid",
            "source_theta_shadow_reward_rescue_authorized",
            "source_theta_shadow_reward_rescue_reason_code",
        ]
        for prefix in (
            "live_same_version",
            "source_same_version",
            "live_persistence",
            "source_persistence",
        ):
            fields.extend(
                [
                    f"source_theta_shadow_{prefix}_valid",
                    f"source_theta_shadow_{prefix}_score",
                    f"source_theta_shadow_{prefix}_dynamics",
                    f"source_theta_shadow_{prefix}_abs_reward",
                    f"source_theta_shadow_{prefix}_signed_reward",
                    f"source_theta_shadow_{prefix}_threshold",
                ]
            )
            if prefix not in {"live_persistence", "source_persistence"}:
                continue
            fields.extend(
                [
                    f"source_theta_shadow_{prefix}_support",
                    f"source_theta_shadow_{prefix}_required",
                ]
            )
            for index in range(4):
                fields.extend(
                    [
                        f"source_theta_shadow_{prefix}_block_{index}_score",
                        f"source_theta_shadow_{prefix}_block_{index}_dynamics",
                        f"source_theta_shadow_{prefix}_block_{index}_abs_reward",
                        f"source_theta_shadow_{prefix}_block_{index}_signed_reward",
                    ]
                )
        return fields

    @staticmethod
    def _recall_candidate_support_trace_fields():
        return [
            "dp_recall_candidate_rms",
            "dp_recall_candidate_rms_threshold",
            "dp_recall_candidate_support_pass",
        ]

    @staticmethod
    def _is_source_theta_shadow_due_event(shadow, current_step):
        if not isinstance(shadow, dict) or not bool(shadow.get("enabled", False)):
            return False
        try:
            due_step = float(shadow.get("due_step"))
            current_step = int(current_step)
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(
            np.isfinite(due_step)
            and due_step == round(due_step)
            and (int(round(due_step)) == current_step)
        )

    _CONTEXT_EXEMPLAR_AUDIT_FIELDS = [
        "context_exemplar_bank_total_segments",
        "context_exemplar_bank_total_transitions",
        "context_exemplar_active_bank_segments",
        "context_exemplar_active_bank_transitions",
        "context_exemplar_last_admission_step",
        "context_exemplar_last_admission_code",
        "context_exemplar_last_admission_retained_count",
        "context_exemplar_last_admission_phase_min",
        "context_exemplar_last_admission_phase_max",
        "context_exemplar_failure_stage_code",
        "context_exemplar_failure_context",
        "context_exemplar_theta_contract_code",
        "context_exemplar_active_phase_match_count",
        "context_exemplar_active_eligible_segment_count",
        "context_exemplar_active_selected_window_count",
        "context_exemplar_active_reference_window_count",
        "context_exemplar_active_validated_reference_count",
        "context_exemplar_active_query_window_count",
        "context_exemplar_active_validated_query_count",
        "context_exemplar_active_query_support_count",
        "context_exemplar_active_query_required_support",
        "context_exemplar_query_valid",
        "context_exemplar_snapshot_valid",
        "context_exemplar_theta_valid",
        "context_exemplar_recompute_count",
        "context_exemplar_evaluated_this_transaction",
        "context_exemplar_receipt_proposal_step",
        "context_exemplar_receipt_due_step",
        "context_exemplar_receipt_source_context",
        "context_exemplar_receipt_proposal_bound",
        "terminal_contract_valid",
        "terminal_contract_action_code",
        "terminal_contract_reason_code",
        "terminal_contract_first_failure_stage_code",
        "terminal_contract_missing_count",
        "identity_evidence_confirmed",
        "identity_confirmed_but_commit_blocked",
        "proposal_bundle_valid",
        "proposal_bundle_panel_complete",
        "proposal_bundle_active_reference_valid",
        "proposal_bundle_factorized_valid",
        "proposal_bundle_h_new_valid",
        "proposal_bundle_selection_count",
        "proposal_bundle_validation_a_count",
        "proposal_bundle_validation_b_count",
        "proposal_bundle_reset_boundary_count",
        "proposal_bundle_episode_chunk_count",
        "proposal_bundle_physical_witness_pooled",
        "proposal_bundle_physical_witness_episode_chunk_count",
        "proposal_bundle_active_comparable",
        "proposal_bundle_active_match",
        "proposal_bundle_active_rejected",
        "proposal_bundle_unified_terminal_authority",
        "proposal_supersede_count",
        "proposal_superseded_from_step",
        "proposal_superseded_to_step",
        "proposal_superseded_reason_code",
        "context_predictive_action_code",
        "context_predictive_active_calibration_reliable",
        "context_predictive_active_lifecycle_state_code",
        "context_predictive_active_coverage_state_code",
        "context_predictive_active_state_code",
        "context_predictive_effective_calibration_units",
        "context_predictive_calibration_window_count",
        "context_predictive_calibration_extreme_rank_resolution",
        "context_predictive_coverage_view_state_code",
        "context_predictive_dynamics_view_state_code",
        "context_predictive_reward_view_state_code",
        "context_h_new_valid",
        "context_h_new_support_query_separated",
        "context_h_new_base_model_read_only",
        "context_h_new_runtime_theta_read_only",
        "context_h_new_heldout_predictive_support",
        "context_h_new_future_predictive_support",
        "context_h_new_adaptation_steps",
    ]
    _ACTIVE_CONTEXT_PROBE_FIELDS = [
        "context_active_probe_enabled",
        "context_active_probe_applied",
        "context_active_probe_reason_code",
        "context_active_probe_proposal_step",
        "context_active_probe_due_step",
        "context_active_probe_hypothesis_count",
        "context_active_probe_elite_count",
        "context_active_probe_selected_index",
        "context_active_probe_selected_score",
    ]
    _REVERSIBLE_REUSE_TRIAL_FIELDS = [
        "reversible_reuse_trial_enabled",
        "reversible_reuse_trial_active",
        "reversible_reuse_trial_status_code",
        "reversible_reuse_trial_source_context",
        "reversible_reuse_trial_target_context",
        "reversible_reuse_trial_rho",
        "reversible_reuse_trial_steps",
        "reversible_reuse_trial_positive_blocks",
        "reversible_reuse_trial_negative_blocks",
        "reversible_reuse_trial_current_block_samples",
        "reversible_reuse_trial_commit_ready",
        "reversible_reuse_trial_last_shadow_advantage",
        "reversible_reuse_trial_last_reward_delta",
        "reversible_reuse_trial_last_block_advantage",
        "reversible_reuse_trial_last_block_reward_delta",
        "reversible_reuse_trial_abort_count",
        "reversible_reuse_trial_commit_count",
        "reversible_reuse_trial_nomination_count",
    ]
    _ONLINE_STEP_FIELDS = [
        "step",
        "protocol_step",
        "episode",
        "reward",
        "done",
        "terminated",
        "truncated",
        "bootstrap_allowed",
        "terminal_observation_is_post_step",
        "forward_velocity",
        "target_velocity",
        "velocity_error",
        "episode_age",
        "steps_since_reset",
        "next_terminal_distance",
        "actual_task",
        "active_context",
        "K",
        "created",
        "recalled",
        "triggered",
        "proposed",
        "context_creation_count",
        "context_recall_count",
        "change_trigger_count",
        "change_proposal_count",
        "change_adaptive_threshold",
        "change_null_mode_code",
        "change_null_effective_threshold",
        "change_null_ready",
        "change_null_source_code",
        "change_null_sample_count",
        "change_null_percentile",
        "change_null_tail_probability",
        "change_null_frozen",
        "pending_change_active",
        "update_paused_pending",
        "agent_update_calls_this_step",
        "model_updates_this_step",
        "optimizer_updates_this_step",
        "agent_update_calls",
        "model_updates",
        "optimizer_updates",
        "pending_update_pauses",
        "control_actor_update_count",
        "control_actor_update_applied",
        "control_actor_q_std",
        "control_actor_ema_penalty",
        "replay_ready",
        "replay_valid_fraction",
        "replay_total_transitions",
        "replay_history_slices",
        "replay_recent_slices",
        "replay_recent_refreshed",
        "replay_recent_refreshes",
        "replay_transaction_pending",
        "replay_staged_transitions",
        "replay_active_context",
        "replay_context_switch_count",
        "main_prefill_pending",
        "main_prefill_recorded_count",
        "main_prefill_applied_count",
        "main_prefill_segment_count",
        "main_prefill_transaction_id",
        "main_prefill_origin_ids_sha256",
        "main_prefill_target_context_id",
        "identity_status",
        "identity_error",
        "identity_active_id",
        "identity_active_context_id",
        "identity_creation_count",
        "identity_recall_count",
        "identity_change_proposal_count",
        "identity_last_action",
        "identity_prediction_loss",
        "identity_theta_norm",
        "identity_theta_entropy",
        "identity_feature_norm",
        "identity_sidecar_updates",
        "identity_replay_transitions",
        "identity_sidecar_wall_ms",
        "behavior_context_id",
        "assigned_context_id",
        "identity_row_disposition",
        "pending_identity_rows",
        "investigation_active",
        "active_uncertain_diagnostic",
        "active_disagreement",
        "context_disagreements",
        "identity_log_probs",
        "identity_scores",
        "identity_raw_scores",
        "identity_new_score",
        "identity_new_increment",
        "identity_grouped_stats",
        "identity_predictor_revisions",
        "identity_applicability_shadow",
        "identity_evidence_peak",
        "identity_new_nomination_step",
        "identity_confirmation_block_index",
        "identity_confirmation_block_first_step",
        "identity_confirmation_block_last_step",
        "identity_confirmation_q_dynamics",
        "identity_confirmation_q_reward",
        "identity_confirmation_joint_evidence",
        "identity_confirmation_dynamics_evidence",
        "identity_confirmation_reward_evidence",
        "identity_confirmation_joint_pass",
        "identity_confirmation_dynamics_pass",
        "identity_confirmation_reward_pass",
        "identity_confirmation_reject_reason",
        "identity_confirmation_kind",
        "identity_confirmation_scope",
        "identity_confirmation_transaction_id",
        "identity_confirmation_source_context_id",
        "identity_confirmation_candidate_context_id",
        "identity_confirmation_source_revision",
        "identity_confirmation_candidate_revision",
        "identity_active_pnn_reward_mean",
        "identity_active_pnn_reward_variance",
        "identity_active_pnn_dynamics_variance",
        "confirmation_method",
        "natural_passed",
        "effective_passed",
        "residual_reference_ready",
        "residual_reference_count",
        "residual_center",
        "residual_tail_rank",
        "residual_direction",
        "automatic_new_success",
        "control_reuse_applied",
        "proposal_enabled",
        "proposal_valid",
        "proposal_accepted",
        "proposal_rejected_reason",
        "proposal_context_id",
        "proposal_theta_version",
        "proposal_model_version",
        "proposal_count",
        "proposal_control_step",
        "proposal_last_accept_step",
        "paired_plain_min_delta",
        "paired_plain_score_delta_per_head",
        "plain_q_head_count",
        "plain_q_head_indices",
        "paired_plain_score_shape",
        "executed_action_source",
        "proposal_sidecar_status",
        "proposal_sidecar_calls",
        "proposal_sidecar_valid_count",
        "proposal_mode",
        "proposal_ranking_source",
        "proposal_selected_index",
        "proposal_used_recall",
        "proposal_theta_current_linf_delta",
        "control_compute_ms",
        "proposal_compute_ms",
        "proposal_input_copy_sync_ms",
        "proposal_sampling_ms",
        "proposal_ranking_ms",
        "proposal_plain_rescore_ms",
        "proposal_other_ms",
        "proposal_total_compute_ms",
        *_ACTIVE_CONTEXT_PROBE_FIELDS,
        *_CONTEXT_EXEMPLAR_AUDIT_FIELDS,
        *_REVERSIBLE_REUSE_TRIAL_FIELDS,
    ]
    _EPISODE_FIELDS = [
        "episode",
        "episode_start_step",
        "episode_end_step",
        "episode_return",
        "episode_reward_mean",
        "episode_length",
        "episode_terminated",
        "episode_truncated",
        "forward_velocity_mean",
        "target_velocity_mean",
        "velocity_error_mean",
        "actual_task",
        "active_contexts",
        "active_context_end",
        "K_end",
        "contexts_created",
        "contexts_recalled",
        "recent_replay_refreshes",
        "changes_triggered",
        "changes_proposed",
        "agent_update_calls",
        "model_updates",
        "optimizer_updates",
        "pending_update_pauses",
    ]

    def __init__(self, *args, **kwargs):
        self.identity_shadow = kwargs.pop("identity_shadow", None)
        self.context_bank = kwargs.pop("context_bank", None)
        self.context_selector = kwargs.pop("context_selector", None)
        self.context_router = kwargs.pop("context_router", None)
        self.trial_controller = kwargs.pop("trial_controller", None)
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._episode_step = 0
        self._start_time = time()
        self._best_eval_reward = float("-inf")
        self._online_resume = False
        if self.context_bank is not None:
            from common.m3wm_context_agent_bank import ContextAgentBank

            if not isinstance(self.context_bank, ContextAgentBank):
                raise TypeError("context_bank must be a ContextAgentBank")
            if self.context_selector is None or not callable(
                getattr(self.context_selector, "decide", None)
            ):
                raise TypeError("ContextAgentBank requires a Context selector")
            if self.context_router is not None and (
                getattr(self.context_router, "bank", None) is not self.context_bank
                or getattr(self.context_router, "selector", None)
                is not self.context_selector
            ):
                raise ValueError(
                    "Context router does not own the configured bank and selector"
                )
            if not bool(getattr(self.buffer, "streaming_enabled", False)):
                raise ValueError("Context-agent mode requires causal streaming replay")
            active = self.context_bank.active_bundle
            if not (
                self.agent is active.control_agent
                and self.buffer is active.control_replay
                and (self.identity_shadow is active.theta_agent)
            ):
                raise ValueError(
                    "Trainer objects are not bound to the active Context bundle"
                )
        self._context_switch_t0_pending = False
        self._protocol_label_steps = {
            40000: "joint-malfunction",
            80000: "wind",
            120000: "velocity",
            160000: "normal",
            200000: "joint-malfunction",
            240000: "wind",
            280000: "velocity",
        }
        self._protocol_label_written = set()
        self._protocol_label_path = None
        self._last_context_decision = None
        self._agent_update_calls = 0
        self._model_updates = 0
        self._optimizer_updates = 0
        self._pending_update_pauses = 0
        self._trial_update_pauses = 0
        self._trial_event_record_index = 0
        self._fit_event_record_index = 0
        self._main_prefill_enabled = True
        if self.context_bank is None:
            raise ValueError(
                "m3wm_context_main_prefill_enabled requires ContextAgentBank"
            )
        self._pending_control_transfer = None
        self._last_main_prefill_receipt = None
        self._last_control_compute_ms = 0.0
        self.guarded_memory = None
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("online_pause_updates_during_pending must be boolean")
        self._pause_updates_during_pending = False
        if not isinstance(True, (bool, np.bool_)):
            raise ValueError(
                "online_refresh_recent_on_confirmed_context must be boolean"
            )
        self._refresh_recent_on_confirmed_context = True
        self._recent_replay_refreshes = 0
        self._last_replay_context_event_counts = {
            "context_creation_count": 0,
            "context_recall_count": 0,
        }
        self._last_replay_transaction_receipt = None
        self._last_replay_context_switch_receipt = None
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("Context identity event capture switch must be boolean")
        self._identity_event_capture_enabled = False
        if not isinstance(False, (bool, np.bool_)):
            raise ValueError("V57 owner shadow switch must be boolean")
        self._v57_owner_shadow_enabled = False
        self._episode_diagnostics = []
        self._last_event_counts = {
            "context_creation_count": 0,
            "context_recall_count": 0,
            "change_trigger_count": 0,
            "change_proposal_count": 0,
        }
        work_dir = Path(str(self.cfg.work_dir))
        work_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir = work_dir
        self._online_steps_path = work_dir / "online_steps.csv"
        self._context_likelihood_path = work_dir / "context_likelihood_trace.csv"
        self._context_likelihood_fields = [
            "step",
            "protocol_step",
            "episode",
            "actual_task",
            "active_context",
            "K",
            "log_prob_C0",
            "log_prob_C1",
            "log_prob_C2",
            "log_prob_C3",
            "log_prob_C4",
            "log_prob_C5",
            "log_prob_C6",
            "log_prob_C7",
            "active_log_prob",
            "identity_new_score",
            "identity_active_disagreement",
        ]
        self._episode_trace_path = work_dir / "online_episodes.csv"
        self._run_status_path = work_dir / "run_status.json"
        self._replay_terminal_ledger_path = work_dir / "replay_terminal_ledger.json"
        self._pure_online_artifact_path = work_dir / "pure_online_runtime_artifact.json"
        self._identity_event_directory = work_dir / "context_identity_events"
        self._v57_shadow_directory = work_dir / "v57_owner_shadow_sidecars"
        self._v57_shadow_run_provenance = None
        self._v57_shadow_next_record_index = 0
        self._v57_shadow_persisted_count = 0
        self._trace_flush_interval = max(
            1, int(self.cfg.get("online_trace_flush_interval", 32))
        )
        self._trace_handles = {}
        self._trace_writers = {}
        self._trace_rows_since_flush = 0
        self._initialize_v57_shadow_persistence()
        self._run_status_write_warning_emitted = False
        self._initialize_csv(self._online_steps_path, self._ONLINE_STEP_FIELDS)
        self._initialize_csv(
            self._context_likelihood_path, self._context_likelihood_fields
        )
        self._initialize_csv(self._episode_trace_path, self._EPISODE_FIELDS)
        self._open_trace_writer(self._online_steps_path, self._ONLINE_STEP_FIELDS)
        self._open_trace_writer(
            self._context_likelihood_path, self._context_likelihood_fields
        )
        self._open_trace_writer(self._episode_trace_path, self._EPISODE_FIELDS)
        self._m3wm_trace_freq = max(1, int(self.cfg.get("m3wm_trace_freq", 10)))
        self._m3wm_trace_path = None
        self._m3wm_trace_fields = None
        if self.cfg.get("enable_m3wm", False):
            self._m3wm_trace_path = work_dir / "theta_trace.csv"
            num_primitives = int(self.cfg.get("m3wm_num_primitives", 4))
            branch_theta_fields = []
            self._m3wm_trace_fields = [
                "step",
                "episode",
                "reward",
                *[f"theta_{i}" for i in range(num_primitives)],
                "theta_entropy",
                "theta_max",
                "theta_margin",
                "dominant_primitive",
                *[f"raw_theta_{i}" for i in range(num_primitives)],
                "raw_theta_entropy",
                "raw_theta_max",
                "raw_theta_margin",
                "raw_dominant_primitive",
                *[f"smoothed_theta_{i}" for i in range(num_primitives)],
                "smoothed_theta_entropy",
                "smoothed_theta_max",
                "smoothed_theta_margin",
                "smoothed_dominant_primitive",
                *branch_theta_fields,
                "context_history_length",
                "context_dyn_residual_rms",
                "context_dyn_residual_mean_abs",
                "context_dyn_residual_std_abs",
                "context_dyn_residual_max_abs",
                "context_dyn_residual_q90",
                "context_action_norm",
                "context_action_gain",
                "context_action_gain_fit",
                "context_action_effect_energy",
                "context_system_id_log_norm",
                "context_reward_residual_signed",
                "context_reward_residual_abs",
                "context_anchor_ready",
                "context_anchor_version",
                "theta_ema_alpha",
                "theta_temperature",
                "base_lr_scale",
                "specialization_active",
                "context_runtime_enabled",
                "online_theta_clean_episode_update_limit",
                "online_theta_clean_episode_updates",
                "online_theta_clean_episode_budget_exhausted",
                "online_theta_transaction_update_limit",
                "online_theta_transaction_updates",
                "online_theta_transaction_budget_exhausted",
                "online_theta_transaction_active",
                "online_theta_transaction_proposal_step",
                "online_theta_terminal_context_hold",
                "online_theta_update_owner_code",
                "online_theta_adapt_eligible",
                "episodic_theta_guard_accept_count",
                "episodic_theta_guard_reject_count",
                "episodic_theta_guard_last_score",
                "episodic_theta_guard_last_decision_code",
                "episodic_theta_guard_champion_score_train_updates",
                "episodic_theta_guard_champion_score_context_version",
                "episodic_theta_guard_champion_score_coordinate_current",
                "episodic_theta_guard_champion_score_train_update_age",
                "episodic_theta_guard_champion_score_context_version_age",
                "episodic_theta_baseline_first_enabled",
                "episodic_theta_baseline_first_hold",
                "episodic_theta_baseline_first_status_code",
                "episodic_theta_baseline_first_complete_blocks",
                "episodic_theta_baseline_first_partial_samples",
                "context_memory_size",
                "active_context_id",
                "change_triggered",
                "change_statistic",
                "change_fast_dynamics_shift",
                "change_fast_abs_reward_shift",
                "change_slow_signed_reward_shift",
                "change_signed_reward_slow_enabled",
                "dp_departure_reason_code",
                "dp_departure_active_standardized_rms",
                "dp_departure_rms_threshold",
                "dp_departure_regime_d_z",
                "dp_active_log_posterior",
                "dp_recall_advantage",
                "dp_recall_emission_advantage",
                "dp_recall_emission_pass",
                "dp_recall_candidate_context",
                *self._recall_candidate_support_trace_fields(),
                "context_exemplar_identity_enabled",
                "context_exemplar_valid",
                "context_exemplar_action_code",
                "context_exemplar_selected_context",
                "context_exemplar_supported_count",
                "context_exemplar_active_rms",
                "context_exemplar_active_radius",
                "context_exemplar_active_support_pass",
                "context_exemplar_active_reference_count",
                *self._CONTEXT_EXEMPLAR_AUDIT_FIELDS,
                *self._ACTIVE_CONTEXT_PROBE_FIELDS,
                "dp_proposal_fast_dynamics_shift",
                "dp_proposal_fast_abs_reward_shift",
                "dp_proposal_slow_signed_reward_shift",
                "dp_same_version_witness_score",
                "dp_same_version_witness_threshold",
                "dp_same_version_witness_dynamics",
                "dp_same_version_witness_reward",
                "dp_pending_persistence_score",
                "dp_pending_persistence_threshold",
                "dp_pending_persistence_dynamics",
                "dp_pending_persistence_reward",
                "dp_pending_persistence_required",
                "dp_physical_witness_shadow_only",
                "dp_physical_response_departure_enabled",
                "dp_physical_witness_valid",
                "dp_physical_witness_pass",
                "dp_physical_witness_score",
                "dp_physical_witness_threshold",
                "dp_physical_witness_support",
                "dp_physical_witness_required",
                "dp_physical_witness_mutual_score",
                "dp_physical_witness_mutual_support",
                "dp_physical_witness_min_rank",
                "dp_physical_response_departure_used",
                "objective_witness_enabled",
                "objective_witness_phase_code",
                "objective_witness_stable_samples",
                "objective_witness_proposal_samples",
                "objective_witness_proposal_available",
                "objective_witness_last_changed",
                "objective_witness_last_blocks_passed",
                "objective_witness_last_total_score",
                "objective_witness_proposal_count",
                "objective_witness_confirm_count",
                "objective_witness_reject_count",
                "objective_witness_unavailable_count",
                "bocpd_short_run_probability",
                "bocpd_map_run_length",
                "bocpd_posterior_size",
                "bocpd_edge_armed",
                "bocpd_edge_stable_low_steps",
                "bocpd_edge_suppressed_count",
                "bocpd_edge_rearm_count",
                "multiscale_fsm_armed",
                "multiscale_fsm_cooldown_remaining",
                "multiscale_fsm_above_threshold_count",
                "multiscale_fsm_below_threshold_count",
                "change_adaptive_threshold",
                "change_null_mode_code",
                "change_null_alpha",
                "change_null_effective_threshold",
                "change_null_ready",
                "change_null_source_code",
                "change_null_sample_count",
                "change_null_context_sample_count",
                "change_null_robot_sample_count",
                "change_null_percentile",
                "change_null_tail_probability",
                "change_null_dynamics_percentile",
                "change_null_dynamics_tail_probability",
                "change_null_reward_percentile",
                "change_null_reward_tail_probability",
                "change_null_frozen",
                "change_null_clean_observation_count",
                "change_null_rejected_observation_count",
                "change_null_std_floor_hits",
                "change_null_statistic_count",
                "change_null_max_statistic",
                "change_trigger_count",
                "change_proposal_count",
                "change_rejection_count",
                "pending_trigger_steps",
                "pending_change_active",
                "memory_existing_log_posterior",
                "memory_new_log_posterior",
                "context_creation_count",
                "context_recall_count",
                *self._REVERSIBLE_REUSE_TRIAL_FIELDS,
                *self._source_theta_shadow_trace_fields(),
            ]
            self._initialize_csv(self._m3wm_trace_path, self._m3wm_trace_fields)
            self._open_trace_writer(self._m3wm_trace_path, self._m3wm_trace_fields)
            print(
                f"[ReCo-WM ThetaTrace] writing online theta to {self._m3wm_trace_path} every {self._m3wm_trace_freq} steps"
            )
        print(
            f"[OnlineStepTrace] writing every counted transition to {self._online_steps_path}"
        )
        self._write_run_status_progress(force=True)

    @staticmethod
    def _initialize_csv(path, fields):
        with path.open("w", newline="", encoding="utf-8-sig") as trace_file:
            csv.DictWriter(trace_file, fieldnames=fields).writeheader()
            trace_file.flush()

    def _open_trace_writer(self, path, fields):
        path = Path(path)
        try:
            trace_file = path.open(
                "a", newline="", encoding="utf-8-sig", buffering=1024 * 1024
            )
        except (OSError, UnicodeError):
            return False
        self._trace_handles[path] = trace_file
        self._trace_writers[path] = csv.DictWriter(trace_file, fieldnames=fields)
        return True

    def _append_csv(self, path, fields, row):
        path = Path(path)
        writer = self._trace_writers.get(path)
        if writer is None:
            with path.open("a", newline="", encoding="utf-8-sig") as trace_file:
                csv.DictWriter(trace_file, fieldnames=fields).writerow(row)
                trace_file.flush()
            return
        writer.writerow(row)
        self._trace_rows_since_flush += 1
        if self._trace_rows_since_flush >= self._trace_flush_interval:
            self._flush_trace_files()

    def _flush_trace_files(self):
        for trace_file in getattr(self, "_trace_handles", {}).values():
            try:
                trace_file.flush()
            except (OSError, ValueError):
                pass
        self._trace_rows_since_flush = 0

    def _close_trace_files(self):
        self._flush_trace_files()
        for trace_file in list(getattr(self, "_trace_handles", {}).values()):
            try:
                trace_file.close()
            except (OSError, ValueError):
                pass
        self._trace_handles = {}
        self._trace_writers = {}

    def __del__(self):
        try:
            self._close_trace_files()
        except Exception:
            pass

    def _replay_terminal_state(self, metrics):
        core = getattr(self.buffer, "_replay_transactions", None)
        total = int(getattr(self.buffer, "total_transitions", 0))
        observed = getattr(core, "observed_transition_count", total)
        committed = getattr(core, "committed_transition_count", total)
        conservation = getattr(core, "transition_conservation_holds", True)
        replay_active_context = getattr(self.buffer, "replay_active_context_id", None)
        if str(self.cfg.get("m3wm_mode", "")).lower() in {
            "pure_online",
            "jump_drift",
        } and (not bool(getattr(self.buffer, "transactional_replay_enabled", False))):
            replay_active_context = metrics.get("active_context_id", None)
        return {
            "transactional_replay_enabled": bool(
                getattr(self.buffer, "transactional_replay_enabled", False)
            ),
            "replay_integrity_ok": bool(
                getattr(self.buffer, "replay_integrity_ok", True)
            ),
            "replay_flush_state": str(
                getattr(self.buffer, "replay_flush_state", "idle")
            ),
            "replay_resolution_count": int(
                getattr(self.buffer, "replay_resolution_count", 0)
            ),
            "replay_flushed_transition_count": int(
                getattr(self.buffer, "replay_flushed_transition_count", 0)
            ),
            "agent_pending": bool(self._agent_has_pending_change(metrics)),
            "buffer_pending": bool(
                getattr(self.buffer, "replay_transaction_pending", False)
            ),
            "staged_transition_count": int(
                getattr(self.buffer, "replay_staged_transitions", 0)
            ),
            "total_transition_count": total,
            "observed_transition_count": int(observed),
            "committed_transition_count": int(committed),
            "transition_conservation_holds": bool(conservation),
            "replay_active_context_id": (
                None if replay_active_context is None else int(replay_active_context)
            ),
            "replay_context_switch_count": int(
                getattr(self.buffer, "replay_context_switch_count", 0)
            ),
            "context_memory_size": self._metric_int(
                metrics.get("context_memory_size", 0), default=0
            ),
            "active_context_id": self._metric_int(
                metrics.get("active_context_id", -1), default=-1
            ),
            "context_creation_count": self._metric_int(
                metrics.get("context_creation_count", 0), default=0
            ),
            "context_recall_count": self._metric_int(
                metrics.get("context_recall_count", 0), default=0
            ),
        }

    @staticmethod
    def _replay_context_alignment_holds(state):
        replay_active = state["replay_active_context_id"]
        active = state["active_context_id"]
        if replay_active == active:
            return True
        return bool(
            replay_active == 0
            and active == -1
            and (state["context_memory_size"] == 0)
            and (state["context_creation_count"] == 0)
            and (state["context_recall_count"] == 0)
            and (state["replay_context_switch_count"] == 0)
        )

    @staticmethod
    def _replay_context_alignment_applicable(cfg):
        return not bool(
            str(cfg.get("m3wm_mode", "")).lower() == "interface_direct_theta"
        )

    @staticmethod
    def _json_replay_proposal_token(token):
        if not (
            isinstance(token, tuple)
            and len(token) == 2
            and (token[0] == "m3wm-proposal")
            and isinstance(token[1], (int, np.integer))
            and (not isinstance(token[1], (bool, np.bool_)))
            and (int(token[1]) >= 1)
        ):
            raise RuntimeError("Terminal replay receipt has an invalid proposal token.")
        return [str(token[0]), int(token[1])]

    def _write_replay_terminal_ledger(self, *, pre_cleanup, cleanup_performed, receipt):
        if not bool(self.cfg.get("enable_m3wm", False)):
            payload = {
                "schema": "tdmpc2_baseline_terminal_ledger",
                "version": 1,
                "run_id": str(self._run_status_path.parent.name),
                "final_step": int(self._step),
                "m3wm_enabled": False,
                "integrity": {
                    "healthy": True,
                    "m3wm_context_alignment": "not_applicable",
                },
            }
            self._replay_terminal_ledger_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return payload
        if self._replay_terminal_ledger_path.exists():
            raise FileExistsError(
                f"Terminal replay ledger already exists: {self._replay_terminal_ledger_path}"
            )
        try:
            status = json.loads(self._run_status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Cannot bind terminal replay ledger to run status."
            ) from exc
        if not isinstance(status, dict) or not str(status.get("run_id", "")):
            raise RuntimeError("Terminal replay ledger requires a non-empty run_id.")
        final = self._replay_terminal_state(self._context_metrics())
        cleanup_performed = bool(cleanup_performed)
        if receipt is None:
            cleanup = {
                "performed": cleanup_performed,
                "resolution": "not_required",
                "proposal_token": None,
                "source_context_id": None,
                "target_context_id": None,
                "segment_id": None,
                "flushed_transition_count": 0,
                "receipt_observed_transition_count": None,
                "receipt_committed_transition_count": None,
            }
        else:
            cleanup = {
                "performed": cleanup_performed,
                "resolution": str(receipt.outcome),
                "proposal_token": self._json_replay_proposal_token(
                    receipt.proposal_token
                ),
                "source_context_id": int(receipt.source_context_id),
                "target_context_id": int(receipt.target_context_id),
                "segment_id": int(receipt.segment_id),
                "flushed_transition_count": int(receipt.flushed_count),
                "receipt_observed_transition_count": int(
                    receipt.observed_transition_count
                ),
                "receipt_committed_transition_count": int(
                    receipt.committed_transition_count
                ),
            }
        expected_steps = int(self.cfg.get("steps", 0))
        pre_pending_agrees = bool(
            pre_cleanup["agent_pending"] == pre_cleanup["buffer_pending"]
        )
        final_pending_agrees = bool(final["agent_pending"] == final["buffer_pending"])
        terminal_transactions_closed = bool(
            not final["agent_pending"]
            and (not final["buffer_pending"])
            and (final["staged_transition_count"] == 0)
        )
        adapter_healthy_and_idle = bool(
            pre_cleanup["replay_integrity_ok"]
            and final["replay_integrity_ok"]
            and (final["replay_flush_state"] == "idle")
        )
        if cleanup_performed:
            cleanup_resolution_valid = bool(
                receipt is not None
                and cleanup["resolution"] == "dismissed"
                and (cleanup["source_context_id"] == cleanup["target_context_id"])
            )
            flush_matches_pre_cleanup_stage = bool(
                cleanup["flushed_transition_count"]
                == pre_cleanup["staged_transition_count"]
                and cleanup["flushed_transition_count"] > 0
            )
            receipt_counts_match_final = bool(
                cleanup["receipt_observed_transition_count"]
                == final["observed_transition_count"]
                and cleanup["receipt_committed_transition_count"]
                == final["committed_transition_count"]
            )
            adapter_counters_match = bool(
                final["replay_resolution_count"]
                == pre_cleanup["replay_resolution_count"] + 1
                and final["replay_flushed_transition_count"]
                == pre_cleanup["replay_flushed_transition_count"]
                + pre_cleanup["staged_transition_count"]
            )
        else:
            cleanup_resolution_valid = bool(
                receipt is None
                and cleanup["resolution"] == "not_required"
                and (not pre_cleanup["buffer_pending"])
            )
            flush_matches_pre_cleanup_stage = bool(
                cleanup["flushed_transition_count"] == 0
                and pre_cleanup["staged_transition_count"] == 0
            )
            receipt_counts_match_final = receipt is None
            adapter_counters_match = bool(
                final["replay_resolution_count"]
                == pre_cleanup["replay_resolution_count"]
                and final["replay_flushed_transition_count"]
                == pre_cleanup["replay_flushed_transition_count"]
            )
        context_unchanged_by_cleanup = all(
            (
                pre_cleanup[key] == final[key]
                for key in (
                    "replay_active_context_id",
                    "replay_context_switch_count",
                    "context_memory_size",
                    "active_context_id",
                    "context_creation_count",
                    "context_recall_count",
                )
            )
        )
        context_alignment_applicable = self._replay_context_alignment_applicable(
            self.cfg
        )
        integrity = {
            "pre_cleanup_pending_agrees": pre_pending_agrees,
            "final_pending_agrees": final_pending_agrees,
            "cleanup_matches_pre_cleanup_pending": bool(
                cleanup_performed == pre_cleanup["buffer_pending"]
            ),
            "cleanup_resolution_valid": cleanup_resolution_valid,
            "flush_matches_pre_cleanup_stage": flush_matches_pre_cleanup_stage,
            "receipt_counts_match_final": receipt_counts_match_final,
            "context_unchanged_by_cleanup": context_unchanged_by_cleanup,
            "pre_cleanup_conservation_holds": bool(
                pre_cleanup["transition_conservation_holds"]
            ),
            "terminal_transactions_closed": terminal_transactions_closed,
            "final_conservation_holds": bool(final["transition_conservation_holds"]),
            "adapter_healthy_and_idle": adapter_healthy_and_idle,
            "adapter_counters_match_cleanup": adapter_counters_match,
            "pre_cleanup_active_context_aligned": not context_alignment_applicable
            or self._replay_context_alignment_holds(pre_cleanup),
            "final_active_context_aligned": not context_alignment_applicable
            or self._replay_context_alignment_holds(final),
            "all_observed_transitions_committed": bool(
                final["total_transition_count"] == expected_steps
                and final["observed_transition_count"] == expected_steps
                and (final["committed_transition_count"] == expected_steps)
            ),
        }
        integrity["healthy"] = all(integrity.values())
        payload = {
            "schema": "m3wm_replay_terminal_ledger",
            "version": 1,
            "run_id": str(status["run_id"]),
            "expected_steps": expected_steps,
            "final_step": int(self._step),
            "context_alignment_applicable": context_alignment_applicable,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pre_cleanup": dict(pre_cleanup),
            "cleanup": cleanup,
            "final": final,
            "integrity": integrity,
        }
        temporary = self._replay_terminal_ledger_path.with_name(
            f".{self._replay_terminal_ledger_path.name}.tmp"
        )
        temporary.write_text(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._replay_terminal_ledger_path)
        if not integrity["healthy"]:
            raise RuntimeError(
                f"Terminal replay ledger failed integrity checks: {integrity}"
            )
        return payload

    def _write_run_status_progress(
        self, *, force=False, status="running", completed=False, stop_reason=None
    ):
        expected_steps = int(self.cfg.get("steps", 0))
        if not force and self._step != expected_steps and (self._step % 100 != 0):
            return
        self._flush_trace_files()
        previous = {}
        try:
            if self._run_status_path.is_file():
                loaded = json.loads(self._run_status_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    previous = loaded
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            payload = {
                "schema": "m3wm_online_four_task_run_status",
                "version": 1,
                "run_id": str(previous.get("run_id", "")),
                "status": str(status),
                "last_step": int(self._step),
                "expected_steps": expected_steps,
                "completed": bool(completed),
                "error": None,
                "stop_reason": stop_reason,
                "pid": int(previous.get("pid", os.getpid())),
                "created_utc": str(previous.get("created_utc", now)),
                "updated_utc": now,
            }
            temporary = self._run_status_path.with_name(
                f".{self._run_status_path.name}.tmp"
            )
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._run_status_path)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            if not self._run_status_write_warning_emitted:
                print(f"[RunStatusWarning] progress status unavailable: {exc}")
                self._run_status_write_warning_emitted = True

    def _set_env_global_step(self, step):
        env = self.env
        visited = set()
        while env is not None and id(env) not in visited:
            visited.add(id(env))
            setter = getattr(type(env), "set_global_step", None)
            if callable(setter):
                setter(env, int(step))
                if not getattr(self, "_ns_step_sync_announced", False):
                    print(
                        "[NonStationaryStepSync] trainer step synchronization enabled"
                    )
                    self._ns_step_sync_announced = True
                return True
            env = getattr(env, "env", None)
        return False

    @staticmethod
    def _metric_scalar(value, default=float("nan")):
        if value is None:
            return float(default)
        try:
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().reshape(-1)[0].item())
            if isinstance(value, np.ndarray):
                return float(value.reshape(-1)[0])
            return float(value)
        except (TypeError, ValueError, IndexError):
            return float(default)

    @classmethod
    def _metric_int(cls, value, default=0):
        scalar = cls._metric_scalar(value, default=float(default))
        return int(default) if not np.isfinite(scalar) else int(round(scalar))

    @staticmethod
    def _nanmean(rows, key):
        values = np.asarray(
            [row.get(key, float("nan")) for row in rows], dtype=np.float64
        )
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    @staticmethod
    def _ordered_unique(values):
        return list(
            dict.fromkeys((value for value in values if value not in (None, "")))
        )

    def _context_metrics(self):
        getter = getattr(self.agent, "context_metrics", None)
        metrics = getter() if callable(getter) else {}
        proposal_getter = getattr(self.agent, "proposal_metrics", None)
        if callable(proposal_getter):
            metrics = {**metrics, **proposal_getter()}
        if self.identity_shadow is not None:
            shadow_getter = getattr(self.identity_shadow, "metrics", None)
            if callable(shadow_getter):
                metrics = {**metrics, **shadow_getter()}
        proposal_sidecar_ms = self._metric_scalar(
            metrics.get("proposal_compute_ms", 0.0), default=0.0
        )
        proposal_plain_rescore_ms = self._metric_scalar(
            metrics.get("proposal_plain_rescore_ms", 0.0), default=0.0
        )
        metrics["proposal_total_compute_ms"] = (
            proposal_sidecar_ms + proposal_plain_rescore_ms
        )
        return {**metrics, "control_compute_ms": self._last_control_compute_ms}

    def _act_with_optional_proposal(self, obs, *, t0, eval_mode=False):
        started = time()
        action = self.agent.act(obs, t0=t0, eval_mode=eval_mode)
        self._last_control_compute_ms = (time() - started) * 1000.0
        consume = getattr(self.agent, "consume_plain_incumbent_plan_result", None)
        select = getattr(self.agent, "select_plain_incumbent_or_proposal", None)
        if not callable(consume) or not callable(select):
            raise RuntimeError("Plain proposal interface is unavailable")
        plain_result = consume()
        if self.identity_shadow is None:
            return select(obs, plain_result, None)
        proposal = self.identity_shadow.propose(
            obs,
            plain_result["proposal_mean"],
            plain_result["proposal_std"],
            mode=str(self.cfg.get("m3wm_plain_incumbent_proposal_mode", "D")),
            plain_ranker=(
                self.agent.rank_plain_extra_search_candidates
                if str(self.cfg.get("m3wm_plain_incumbent_proposal_mode", "D")).upper()
                == "B"
                else None
            ),
        )
        proposal["control_step"] = int(self._step + 1)
        if bool(getattr(self.identity_shadow, "full_tdmpc2_enabled", False)) and (
            bool(getattr(self.identity_shadow, "failed", False))
            or proposal.get("reason") != "ok"
        ):
            raise RuntimeError(
                f"registered Proposal-theta learner failed closed: {proposal.get('reason')}"
            )
        self.agent._guarded_store_witness_pending = False
        saved_current = select(obs, plain_result, proposal, control_step=self._step + 1)
        consume_frame = getattr(self.agent, "consume_guarded_control_frame", None)
        control_frame = consume_frame() if callable(consume_frame) else None
        return saved_current

    def _agent_has_pending_change(self, metrics=None):
        if not self.cfg.get("enable_m3wm", False):
            return False
        if metrics is not None:
            return (
                self._metric_scalar(
                    metrics.get("pending_change_active", 0.0), default=0.0
                )
                > 0.5
            )
        checker = getattr(self.agent, "_m3wm_has_pending_change", None)
        if callable(checker):
            return bool(checker())
        metrics = self._context_metrics()
        return (
            self._metric_scalar(metrics.get("pending_change_active", 0.0), default=0.0)
            > 0.5
        )

    def _agent_model_updates(self):
        if not self.cfg.get("enable_m3wm", False):
            return None
        value = getattr(self.agent, "_m3wm_train_updates", None)
        if isinstance(value, (int, np.integer)) and (not isinstance(value, bool)):
            return int(value)
        return None

    def _record_successful_agent_update(self):
        self._agent_update_calls += 1
        internal_updates = self._agent_model_updates()
        if self.context_bank is not None:
            self._model_updates += 1
        elif internal_updates is None:
            self._model_updates += 1
        else:
            self._model_updates = internal_updates
        self._optimizer_updates += 2

    def _write_context_switch_receipt(self, decision, receipt):
        path = self._work_dir / "context_agent_switches.jsonl"
        to_record = getattr(decision, "to_record", None)
        evidence = (
            dict(decision.evidence)
            if hasattr(decision, "evidence")
            else to_record() if callable(to_record) else {}
        )
        payload = {
            "step": int(self._step + 1),
            "action": str(decision.action),
            "decision_source": str(
                getattr(decision, "decision_source", "pnn_evidence_s0")
            ),
            "reason": str(decision.reason),
            "source_context_id": int(receipt.source_context_id),
            "target_context_id": int(receipt.target_context_id),
            "source_bundle_hash": str(receipt.source_bundle_hash),
            "target_bundle_hash": str(receipt.target_bundle_hash),
            "runtime_segment_id": int(receipt.runtime_segment_id),
            "target_loaded_from_disk": bool(receipt.target_loaded_from_disk),
            "switch_wall_ms": float(receipt.switch_wall_ms),
            "evidence": evidence,
            "env_reset_performed": False,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_context_confirmation_receipt(self, decision, *, step):
        to_record = getattr(decision, "to_record", None)
        if not callable(to_record):
            return
        record = to_record()
        block = record.get("confirmation_block")
        if not isinstance(block, dict):
            return
        payload = {
            "step": int(step),
            "action": str(record.get("action", "")),
            "source_context_id": record.get("source_context_id"),
            "candidate_context_id": record.get("candidate_context_id"),
            "behavior_context_id": record.get("behavior_context_id"),
            "raw_action": record.get("raw_action"),
            "admitted_action": record.get("admitted_action"),
            "hold_reason": record.get("hold_reason"),
            "confirmation_block": block,
        }
        path = self._work_dir / "context_agent_confirmation_blocks.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_trial_records(self):
        if self.trial_controller is None:
            return
        records = self.trial_controller.records()
        if self._trial_event_record_index >= len(records):
            return
        path = self._work_dir / "context_agent_trials.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            for record in records[self._trial_event_record_index :]:
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        self._trial_event_record_index = len(records)

    def _write_context_fit_records(self):
        if self.context_selector is None:
            return
        getter = getattr(self.context_selector, "fit_records", None)
        if not callable(getter):
            return
        records = getter()
        if self._fit_event_record_index >= len(records):
            return
        path = self._work_dir / "context_agent_pnn_fits.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            for record in records[self._fit_event_record_index :]:
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        self._fit_event_record_index = len(records)

    def _write_main_prefill_record(self, payload):
        path = self._work_dir / "context_main_prefill_transfers.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _bundle_learning_component_hashes(bundle):
        from common.m3wm_context_agent_bundle import _digest

        def state(owner, name):
            value = getattr(owner, name, None)
            return value.state_dict() if value is not None else None

        control = bundle.control_agent
        theta = bundle.theta_agent
        theta_side = getattr(theta, "side_agent", None)
        return {
            "control": _digest(
                {
                    "model": state(control, "model"),
                    "scale": state(control, "scale"),
                    "optim": state(control, "optim"),
                    "pi_optim": state(control, "pi_optim"),
                }
            ),
            "theta": _digest(
                {
                    "model": state(theta_side, "model"),
                    "scale": state(theta_side, "scale"),
                    "optim": state(theta_side, "optim"),
                    "pi_optim": state(theta_side, "pi_optim"),
                    "current_theta": getattr(theta, "current_theta", None),
                    "proposal_theta_version": getattr(
                        theta, "proposal_theta_version", None
                    ),
                }
            ),
        }

    @staticmethod
    def _replay_transition_count(replay):
        return int(
            getattr(
                replay, "total_transitions", getattr(replay, "_total_transitions", 0)
            )
        )

    @staticmethod
    def _theta_replay_transition_count(theta):
        replay = getattr(theta, "side_buffer", None)
        return OnlineTrainer._replay_transition_count(replay)

    def _new_replay_seed(self, obs, command_action):
        action = torch.full_like(
            torch.as_tensor(command_action).detach().cpu(), float("nan")
        )
        return self.to_td(obs, action=action)

    def _bind_new_context_with_main_prefill(self, bundle, *, plan, commit_prev_obs):
        self.agent = bundle.control_agent
        self.buffer = bundle.control_replay
        self.identity_shadow = bundle.theta_agent
        bundle.reset_activation_transients()
        if self._replay_transition_count(self.buffer) != 0:
            raise RuntimeError("NEW main-prefill target replay is not empty")
        if getattr(self.buffer, "_active_episode_id", None) is not None:
            raise RuntimeError("NEW main-prefill target replay already owns an episode")
        if self._theta_replay_transition_count(self.identity_shadow) != 0:
            raise RuntimeError("NEW main-prefill target theta replay is not empty")
        component_hashes_before = self._bundle_learning_component_hashes(bundle)
        main_updates_before = int(self._agent_update_calls)
        theta_step_before = int(getattr(self.identity_shadow, "step", 0))
        theta_updates_before = int(
            getattr(
                getattr(self.identity_shadow, "side_agent", None),
                "_m3wm_train_updates",
                0,
            )
        )
        for segment_index, rows in enumerate(plan.segments):
            if getattr(self.buffer, "_active_episode_id", None) is not None:
                self.buffer.end_episode()
            seed_td = self._new_replay_seed(rows[0].prev_obs, rows[0].command_action)
            self.buffer.begin_episode(seed_td.detach().clone().cpu())
            for row in rows:
                self.buffer.append_transition(row.transition_td.detach().clone().cpu())
            is_tail = segment_index == len(plan.segments) - 1
            if not is_tail or not plan.tail_connects_commit:
                self.buffer.end_episode()
        commit_action_shape = plan.segments[-1][-1].command_action
        commit_seed = self._new_replay_seed(commit_prev_obs, commit_action_shape)
        if not plan.tail_connects_commit:
            self.buffer.begin_episode(commit_seed.detach().clone().cpu())
        self.identity_shadow.begin_episode_td(commit_seed)
        self._context_switch_t0_pending = True
        component_hashes_after = self._bundle_learning_component_hashes(bundle)
        if component_hashes_after != component_hashes_before:
            raise RuntimeError("main replay prefill mutated trainable components")
        if int(self._agent_update_calls) != main_updates_before:
            raise RuntimeError("main replay prefill performed a control update")
        if int(getattr(self.identity_shadow, "step", 0)) != theta_step_before:
            raise RuntimeError("main replay prefill advanced theta.step")
        if (
            int(
                getattr(
                    getattr(self.identity_shadow, "side_agent", None),
                    "_m3wm_train_updates",
                    0,
                )
            )
            != theta_updates_before
        ):
            raise RuntimeError("main replay prefill performed a theta update")
        observed_count = self._replay_transition_count(self.buffer)
        if observed_count != plan.prefill_count:
            raise RuntimeError(
                f"main replay prefill transition count mismatch: {observed_count} != {plan.prefill_count}"
            )
        receipt = {
            "schema": "m3wm-main-replay-prefill-receipt-v1",
            "event": "PREFILL_APPLIED",
            "transaction_id": plan.transaction_id,
            "source_context_id": int(plan.source_context_id),
            "target_context_id": int(bundle.context_id),
            "nomination_transition_id": int(plan.nomination_transition_id),
            "commit_transition_id": int(plan.commit_transition_id),
            "source_revision": int(plan.source_revision),
            "init_capsule_hash": plan.init_capsule_hash,
            "prefill_count": int(plan.prefill_count),
            "prefill_origin_ids_sha256": plan.origin_ids_sha256,
            "prefill_segments": plan.audit_segments(),
            "tail_connects_commit": bool(plan.tail_connects_commit),
            "child_main_replay_count_before_commit": observed_count,
            "child_theta_replay_count_before_commit": int(
                self._theta_replay_transition_count(self.identity_shadow)
            ),
            "main_update_calls_added": int(
                self._agent_update_calls - main_updates_before
            ),
            "theta_step_added": int(
                getattr(self.identity_shadow, "step", 0) - theta_step_before
            ),
            "theta_updates_added": int(
                getattr(
                    getattr(self.identity_shadow, "side_agent", None),
                    "_m3wm_train_updates",
                    0,
                )
                - theta_updates_before
            ),
            "component_hashes_before": component_hashes_before,
            "component_hashes_after": component_hashes_after,
            "first_sample_unique_transition_ids": None,
            "first_sample_real_loss_rows": None,
        }
        bundle.metadata.update(
            {
                "main_replay_prefill_policy": "nomination_to_precommit_source_rows",
                "main_replay_prefill_count": int(plan.prefill_count),
                "main_replay_prefill_origin_ids_sha256": plan.origin_ids_sha256,
                "main_replay_prefill_transaction_id": plan.transaction_id,
            }
        )
        self._last_main_prefill_receipt = receipt
        self._write_main_prefill_record(receipt)

    def _handle_main_prefill_route(
        self, route_receipt, *, selector_row, control_payload, commit_prev_obs
    ):
        if bool(getattr(self.context_router, "observer_only", False)):
            return False
        if route_receipt is None:
            return False
        decision = route_receipt.decision
        action = str(decision.action)
        pending = self._pending_control_transfer
        if pending is None:
            if action != "CONFIRMING_NEW":
                return False
            capsule = self.context_bank.pending_init_capsule_record()
            if capsule is None:
                raise RuntimeError("NEW nomination has no captured InitCapsule")
            pending = PendingControlTransfer(
                transaction_id=str(decision.new_transaction_id),
                source_context_id=int(decision.source_context_id),
                nomination_transition_id=int(decision.new_nomination_step),
                source_revision=int(decision.new_source_revision),
                init_capsule_hash=str(capsule["capsule_hash"]),
            )
            if not pending.matches(
                transaction_id=capsule["transaction_id"],
                source_context_id=capsule["source_context_id"],
                nomination_transition_id=capsule["nomination_transition_id"],
                source_revision=capsule["source_revision"],
                init_capsule_hash=capsule["capsule_hash"],
            ):
                raise RuntimeError("control transfer and InitCapsule identities differ")
            self._pending_control_transfer = pending
            pending.record(
                selector_row,
                control_payload,
                behavior_context_id=int(route_receipt.behavior_context_id),
            )
            self._write_main_prefill_record(pending.audit_record(event="NOMINATED"))
            return False
        block = decision.confirmation_block or {}
        if (
            action == "CONFIRMING"
            and block.get("event") == "new_confirmation_interrupted"
        ):
            expected = (
                pending.nomination_transition_id,
                pending.source_context_id,
                pending.source_revision,
            )
            observed = (
                block.get("nomination_step"),
                block.get("source_context_id"),
                block.get("source_revision"),
            )
            if observed != expected:
                raise RuntimeError(
                    "NEW cancellation does not match the control transfer"
                )
            self._write_main_prefill_record(
                pending.audit_record(
                    event="DISCARDED", reason="new_confirmation_interrupted_by_history"
                )
            )
            self._pending_control_transfer = None
            return False
        if action in {"NEW_REJECTED", "RETURN"}:
            self._write_main_prefill_record(
                pending.audit_record(
                    event="DISCARDED", reason=f"selector_{action.lower()}"
                )
            )
            self._pending_control_transfer = None
            return False
        if action == "NEW":
            if str(decision.new_transaction_id) != pending.transaction_id:
                raise RuntimeError(
                    "NEW commit changed the control transfer transaction"
                )
            if route_receipt.bank_switch_receipt is None:
                raise RuntimeError("confirmed NEW has no bank switch receipt")
            plan = pending.build_plan(
                commit_transition_id=int(selector_row["transition_id"]),
                commit_prev_obs=commit_prev_obs,
            )
            self._bind_new_context_with_main_prefill(
                self.context_bank.active_bundle,
                plan=plan,
                commit_prev_obs=commit_prev_obs,
            )
            self._pending_control_transfer = None
            return True
        if int(self.context_bank.active_context_id) != pending.source_context_id:
            raise RuntimeError(
                "control transfer source lost authority before NEW commit"
            )
        pending.record(
            selector_row,
            control_payload,
            behavior_context_id=int(route_receipt.behavior_context_id),
        )
        return False

    def _discard_pending_main_prefill(self, *, reason):
        pending = self._pending_control_transfer
        if pending is None:
            return False
        self._write_main_prefill_record(
            pending.audit_record(event="DISCARDED", reason=str(reason))
        )
        self._pending_control_transfer = None
        if self.context_bank is not None and hasattr(
            self.context_bank, "discard_pending_init_capsule"
        ):
            self.context_bank.discard_pending_init_capsule(reason=str(reason))
        return True

    @staticmethod
    def _close_bundle_episode(bundle):
        if getattr(bundle.control_replay, "_active_episode_id", None) is not None:
            bundle.control_replay.end_episode()
        theta_buffer = getattr(bundle.theta_agent, "side_buffer", None)
        if getattr(theta_buffer, "_active_episode_id", None) is not None:
            bundle.theta_agent.end_episode()

    def _prepare_s0_context_switch(self, decision):
        self._close_bundle_episode(self.context_bank.active_bundle)

    def _bind_context_bundle(self, bundle, *, seed_obs=None):
        self.agent = bundle.control_agent
        self.buffer = bundle.control_replay
        self.identity_shadow = bundle.theta_agent
        bundle.reset_activation_transients()
        if seed_obs is not None:
            seed_td = self.to_td(seed_obs)
            self.buffer.begin_episode(seed_td.detach().clone().cpu())
            self.identity_shadow.begin_episode_td(seed_td)
            self._context_switch_t0_pending = True

    def _record_protocol_label(self, *, protocol_step):
        step = int(protocol_step)
        label = self._protocol_label_steps.get(step)
        if label is None or step in self._protocol_label_written:
            return None
        self._protocol_label_written.add(step)
        wind_steps = ()
        record = {
            "schema": "m3wm-protocol-label-audit-v1",
            "step": step,
            "environment_label": label,
            "phase_occurrence": 1 if step < 160000 else 2,
            "label_source": "registered_protocol_schedule",
            "selector_label_input": False,
            "training_label_input": False,
        }
        if self._protocol_label_path is None:
            self._protocol_label_path = self._work_dir / "protocol_labels.jsonl"
        self._protocol_label_path.parent.mkdir(parents=True, exist_ok=True)
        with self._protocol_label_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def _selector_transition_row(
        self, *, transition_step, prev_obs, action, reward, next_obs, info
    ):
        return {
            "transition_id": int(transition_step),
            "obs": torch.as_tensor(prev_obs).detach().cpu().clone().flatten(),
            "action": torch.as_tensor(action).detach().cpu().clone().flatten(),
            "reward": float(torch.as_tensor(reward).detach().cpu().item()),
            "next_obs": torch.as_tensor(next_obs).detach().cpu().clone().flatten(),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
        }

    def _flush_successful_trial(self, resolution):
        active = self.context_bank.active_bundle
        self._bind_context_bundle(active, seed_obs=None)
        previous_episode = None
        for selector_row, payload in zip(resolution.rows, resolution.control_payloads):
            if payload is None:
                raise RuntimeError("successful trial is missing a control payload")
            episode_id = int(payload["episode_id"])
            if previous_episode != episode_id:
                self._close_bundle_episode(active)
                seed_td = self.to_td(payload["prev_obs"])
                self.buffer.begin_episode(seed_td.detach().clone().cpu())
                self.identity_shadow.begin_episode_td(seed_td)
            previous_episode = episode_id
            transition_td = payload["transition_td"].detach().clone().cpu()
            self.buffer.append_transition(transition_td)
            self.identity_shadow.append_transition_td(transition_td)
            self.context_bank.append_transition(
                transition_td,
                behavior_context_id=int(resolution.candidate_context_id),
                assigned_context_id=int(resolution.assigned_context_id),
                episode_id=episode_id,
                terminated=bool(selector_row["terminated"]),
                truncated=bool(selector_row["truncated"]),
            )
            if bool(selector_row["terminated"]) or bool(selector_row["truncated"]):
                self._close_bundle_episode(active)
                previous_episode = None

    def _restore_after_failed_trial(self, *, current_obs, done):
        active = self.context_bank.active_bundle
        self._bind_context_bundle(active, seed_obs=None if bool(done) else current_obs)

    def _activate_trial_behavior(self):
        working = self.trial_controller.working_bundle
        if working is None:
            raise RuntimeError("trial lease has no working Context bundle")
        self._close_bundle_episode(self.context_bank.active_bundle)
        self._bind_context_bundle(working, seed_obs=None)
        self._write_trial_records()

    def _maybe_refresh_recent_replay(self, metrics):
        current = {
            key: self._metric_int(metrics.get(key, 0), default=0)
            for key in self._last_replay_context_event_counts
        }
        previous = dict(self._last_replay_context_event_counts)
        self._last_replay_context_event_counts.update(current)
        if bool(getattr(self.buffer, "transactional_replay_enabled", False)):
            return False
        created_after_bootstrap = bool(
            previous["context_creation_count"] >= 1
            and current["context_creation_count"] > previous["context_creation_count"]
        )
        recalled = bool(
            current["context_recall_count"] > previous["context_recall_count"]
        )
        if not (
            self._refresh_recent_on_confirmed_context
            and (created_after_bootstrap or recalled)
        ):
            return False
        refresh = getattr(self.buffer, "refresh_recent_segment", None)
        if not callable(refresh):
            raise RuntimeError(
                "online_refresh_recent_on_confirmed_context requires a streaming buffer with refresh_recent_segment()"
            )
        refresh()
        self._recent_replay_refreshes += 1
        return True

    def _replay_proposal_token(self, metrics):
        value = self._metric_scalar(metrics.get("bocpd_proposal_step", float("nan")))
        if not np.isfinite(value) or value < 1 or value != round(value):
            raise RuntimeError("Pending Context has no valid replay proposal token.")
        return ("m3wm-proposal", int(round(value)))

    def _replay_active_proposal_token(self, metrics):
        active = getattr(self.buffer, "replay_pending_proposal_token", None)
        if active is not None:
            return active
        return self._replay_proposal_token(metrics)

    def _sync_transactional_replay(
        self, metrics_before, metrics_after, transition_td=None
    ):
        if not bool(getattr(self.buffer, "transactional_replay_enabled", False)):
            if transition_td is not None:
                self.buffer.append_transition(transition_td)
            return False
        pending_before = self._agent_has_pending_change(metrics_before)
        pending_after = self._agent_has_pending_change(metrics_after)
        if not pending_before and pending_after:
            token = self._replay_proposal_token(metrics_after)
            source = self._metric_int(
                metrics_before.get("active_context_id", 0), default=0
            )
            source = max(0, source)
            self.buffer.begin_pending_transaction(token, source_context_id=source)
            if transition_td is not None:
                self.buffer.append_transition(transition_td, proposal_token=token)
            return False
        if pending_before:
            token = self._replay_active_proposal_token(metrics_before)
            if transition_td is not None:
                self.buffer.append_transition(transition_td, proposal_token=token)
            if pending_after:
                return False
            created = self._metric_int(
                metrics_after.get("context_creation_count", 0)
            ) > self._metric_int(metrics_before.get("context_creation_count", 0))
            recalled = self._metric_int(
                metrics_after.get("context_recall_count", 0)
            ) > self._metric_int(metrics_before.get("context_recall_count", 0))
            target = self._metric_int(
                metrics_after.get("active_context_id", -1), default=-1
            )
            context_committed = bool((created or recalled) and target >= 0)
            receipt = self.buffer.resolve_pending_transaction(
                token,
                context_committed=context_committed,
                target_context_id=target if context_committed else None,
            )
            self._last_replay_transaction_receipt = receipt
            if context_committed:
                self._recent_replay_refreshes += 1
            return context_committed
        if bool(getattr(self.buffer, "replay_transaction_pending", False)):
            raise RuntimeError(
                "Replay retained a proposal after the agent transaction closed."
            )
        created = self._metric_int(
            metrics_after.get("context_creation_count", 0)
        ) > self._metric_int(metrics_before.get("context_creation_count", 0))
        recalled = self._metric_int(
            metrics_after.get("context_recall_count", 0)
        ) > self._metric_int(metrics_before.get("context_recall_count", 0))
        if created or recalled:
            source = self._metric_int(
                metrics_before.get("active_context_id", -1), default=-1
            )
            target = self._metric_int(
                metrics_after.get("active_context_id", -1), default=-1
            )
            creations_before = self._metric_int(
                metrics_before.get("context_creation_count", 0)
            )
            creations_after = self._metric_int(
                metrics_after.get("context_creation_count", 0)
            )
            bootstrap_create = bool(
                created
                and (not recalled)
                and (source == -1)
                and (target == 0)
                and (creations_before == 0)
                and (creations_after == 1)
            )
            if bootstrap_create:
                replay_active = getattr(self.buffer, "replay_active_context_id", None)
                if replay_active is None or int(replay_active) != 0:
                    raise RuntimeError("Bootstrap Context does not match replay C0.")
                if transition_td is not None:
                    self.buffer.append_transition(transition_td)
                return False
            owner_changed = bool(source >= 0 and target >= 0 and (source != target))
            replay_active = getattr(self.buffer, "replay_active_context_id", None)
            if transition_td is not None:
                if not owner_changed:
                    if (
                        replay_active is not None
                        and target >= 0
                        and (int(replay_active) != target)
                    ):
                        raise RuntimeError(
                            f"Non-switch Context event disagrees with replay active owner: agent={target}, replay={replay_active}"
                        )
                    self.buffer.append_transition(transition_td)
                    return False
                if replay_active is not None:
                    if int(replay_active) == target:
                        self.buffer.append_transition(transition_td)
                        return False
                    if int(replay_active) != source:
                        raise RuntimeError(
                            f"Row-bearing Context switch source does not match replay active owner: agent={source}, replay={replay_active}"
                        )
                switch = getattr(
                    self.buffer, "commit_context_switch_after_transition", None
                )
                if not callable(switch):
                    raise RuntimeError(
                        "Transactional replay cannot mirror a row-bearing Context commit."
                    )
                receipt = switch(
                    transition_td, source_context_id=source, target_context_id=target
                )
                self._last_replay_context_switch_receipt = receipt
                self._recent_replay_refreshes += 1
                return True
            if created and recalled:
                raise RuntimeError(
                    "One idle replay boundary cannot both create and recall."
                )
            if not owner_changed:
                return False
            if source < 0 or target < 0 or source == target:
                raise RuntimeError(
                    "Idle Context event has no valid active-Context switch."
                )
            if replay_active is not None:
                if int(replay_active) == target:
                    return False
                if int(replay_active) != source:
                    raise RuntimeError(
                        f"Idle Context switch source does not match replay active owner: agent={source}, replay={replay_active}"
                    )
            switch = getattr(self.buffer, "commit_context_switch_at_boundary", None)
            if not callable(switch):
                raise RuntimeError(
                    "Transactional replay cannot mirror an idle Context switch."
                )
            receipt = switch(source_context_id=source, target_context_id=target)
            self._last_replay_context_switch_receipt = receipt
            self._recent_replay_refreshes += 1
            return True
        if transition_td is not None:
            self.buffer.append_transition(transition_td)
        return False

    @staticmethod
    def _validate_v57_agent_sidecar(artifact):
        from tdmpc2 import validate_m3wm_v57_shadow_sidecar

        return validate_m3wm_v57_shadow_sidecar(artifact)

    def _initialize_v57_shadow_persistence(self):
        return

    def _drain_v57_shadow_sidecars(self):
        return 0

    def _assert_v57_shadow_persistence_complete(self):
        return

    def _persist_context_identity_event_artifact(self):
        pop = getattr(self.agent, "pop_context_identity_event_artifact", None)
        if not callable(pop):
            return None
        artifact = pop()
        if artifact is None:
            return None
        raise RuntimeError(
            "Agent produced an identity event while capture is disabled."
        )

    def _write_pure_online_runtime_artifact(self):
        getter = getattr(self.agent, "pure_online_artifact_snapshot", None)
        if not callable(getter):
            return None
        artifact = getter()
        if artifact is None:
            return None
        manifest_path = self._work_dir / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, dict) or "run_id" not in manifest:
            raise RuntimeError("run manifest is invalid")
        payload = {
            **artifact,
            "run_provenance": {
                "run_id": str(manifest.get("run_id", "")),
                "run_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "source_sha256": dict(manifest.get("source_sha256", {})),
            },
        }
        payload["payload_sha256"] = identity_event_payload_sha256(payload)
        temporary = self._pure_online_artifact_path.with_name(
            f".{self._pure_online_artifact_path.name}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._pure_online_artifact_path)
        return self._pure_online_artifact_path

    def _step_diagnostics(self, info, reward, metrics, done=None):
        actual_task = info.get("actual_task", info.get("task", "unknown"))
        if done is None:
            done = bool(
                info.get("done", False)
                or info.get("terminated", False)
                or info.get("truncated", False)
            )
        contract = control_contract(bool(done), info)
        return {
            "reward": self._metric_scalar(reward),
            "done": int(contract.done),
            "terminated": int(contract.terminated),
            "truncated": int(contract.truncated),
            "bootstrap_allowed": int(contract.bootstrap_allowed),
            "terminal_observation_is_post_step": int(
                contract.terminal_observation_is_post_step
            ),
            "forward_velocity": self._metric_scalar(info.get("forward_velocity")),
            "target_velocity": self._metric_scalar(info.get("target_velocity")),
            "velocity_error": self._metric_scalar(info.get("velocity_error")),
            "episode_age": int(self._episode_step + 1),
            "steps_since_reset": int(self._episode_step + 1),
            "next_terminal_distance": self._metric_scalar(
                info.get(
                    "steps_to_terminal", info.get("time_to_terminal", float("nan"))
                )
            ),
            "actual_task": str(actual_task),
            "active_context": self._metric_int(
                metrics.get("active_context_id", -1), default=-1
            ),
            "K": self._metric_int(metrics.get("context_memory_size", 0), default=0),
            "context_creation_count": self._metric_int(
                metrics.get("context_creation_count", 0), default=0
            ),
            "context_recall_count": self._metric_int(
                metrics.get("context_recall_count", 0), default=0
            ),
            "change_trigger_count": self._metric_int(
                metrics.get("change_trigger_count", 0), default=0
            ),
            "change_proposal_count": self._metric_int(
                metrics.get("change_proposal_count", 0), default=0
            ),
            "change_adaptive_threshold": self._metric_scalar(
                metrics.get("change_adaptive_threshold", float("nan"))
            ),
            "change_null_mode_code": self._metric_scalar(
                metrics.get("change_null_mode_code", float("nan"))
            ),
            "change_null_effective_threshold": self._metric_scalar(
                metrics.get("change_null_effective_threshold", float("nan"))
            ),
            "change_null_ready": self._metric_scalar(
                metrics.get("change_null_ready", float("nan"))
            ),
            "change_null_source_code": self._metric_scalar(
                metrics.get("change_null_source_code", float("nan"))
            ),
            "change_null_sample_count": self._metric_scalar(
                metrics.get("change_null_sample_count", float("nan"))
            ),
            "change_null_percentile": self._metric_scalar(
                metrics.get("change_null_percentile", float("nan"))
            ),
            "change_null_tail_probability": self._metric_scalar(
                metrics.get("change_null_tail_probability", float("nan"))
            ),
            "change_null_frozen": self._metric_scalar(
                metrics.get("change_null_frozen", float("nan"))
            ),
            "change_triggered": int(
                self._metric_scalar(metrics.get("change_triggered", 0), default=0) > 0.5
            ),
            "pending_change_active": int(self._agent_has_pending_change(metrics)),
            "control_actor_update_count": self._metric_scalar(
                metrics.get("control_actor_update_count", 0), default=0
            ),
            "control_actor_update_applied": self._metric_scalar(
                metrics.get("control_actor_update_applied", 0), default=0
            ),
            "control_actor_q_std": self._metric_scalar(
                metrics.get("control_actor_q_std", float("nan"))
            ),
            "control_actor_ema_penalty": self._metric_scalar(
                metrics.get("control_actor_ema_penalty", float("nan"))
            ),
            **{
                field: self._metric_int(metrics.get(field, 0), default=0)
                for field in self._CONTEXT_EXEMPLAR_AUDIT_FIELDS
            },
            **{
                field: self._metric_scalar(metrics.get(field, float("nan")))
                for field in self._ACTIVE_CONTEXT_PROBE_FIELDS
            },
            **{
                field: self._metric_scalar(metrics.get(field, float("nan")))
                for field in self._REVERSIBLE_REUSE_TRIAL_FIELDS
            },
            "identity_status": str(metrics.get("identity_status", "disabled")),
            "identity_error": str(metrics.get("identity_error", "")),
            "identity_active_id": self._metric_int(
                metrics.get("identity_active_id", -1), default=-1
            ),
            "identity_active_context_id": self._metric_int(
                metrics.get("identity_active_context_id", -1), default=-1
            ),
            "identity_creation_count": self._metric_int(
                metrics.get("identity_creation_count", 0), default=0
            ),
            "identity_recall_count": self._metric_int(
                metrics.get("identity_recall_count", 0), default=0
            ),
            "identity_change_proposal_count": self._metric_int(
                metrics.get("identity_change_proposal_count", 0), default=0
            ),
            "identity_last_action": str(metrics.get("identity_last_action", "UNKNOWN")),
            "identity_prediction_loss": self._metric_scalar(
                metrics.get("identity_prediction_loss", float("nan"))
            ),
            "identity_theta_norm": self._metric_scalar(
                metrics.get("identity_theta_norm", float("nan"))
            ),
            "identity_theta_entropy": self._metric_scalar(
                metrics.get("identity_theta_entropy", float("nan"))
            ),
            "identity_feature_norm": self._metric_scalar(
                metrics.get("identity_feature_norm", float("nan"))
            ),
            "identity_sidecar_updates": self._metric_int(
                metrics.get("identity_sidecar_updates", 0), default=0
            ),
            "identity_replay_transitions": self._metric_int(
                metrics.get("identity_replay_transitions", 0), default=0
            ),
            "identity_sidecar_wall_ms": self._metric_scalar(
                metrics.get("identity_sidecar_wall_ms", float("nan"))
            ),
            "behavior_context_id": -1,
            "assigned_context_id": -1,
            "identity_row_disposition": "NONE",
            "pending_identity_rows": 0,
            "investigation_active": 0,
            "active_uncertain_diagnostic": 0,
            "active_disagreement": float("nan"),
            "context_disagreements": "{}",
            "identity_log_probs": "{}",
            "identity_scores": "{}",
            "identity_raw_scores": "{}",
            "identity_new_score": float("nan"),
            "identity_new_increment": float("nan"),
            "identity_grouped_stats": "{}",
            "identity_predictor_revisions": "{}",
            "identity_applicability_shadow": "{}",
            "identity_evidence_peak": float("nan"),
            "identity_new_nomination_step": -1,
            "identity_confirmation_block_index": -1,
            "identity_confirmation_block_first_step": -1,
            "identity_confirmation_block_last_step": -1,
            "identity_confirmation_q_dynamics": float("nan"),
            "identity_confirmation_q_reward": float("nan"),
            "identity_confirmation_joint_evidence": float("nan"),
            "identity_confirmation_dynamics_evidence": float("nan"),
            "identity_confirmation_reward_evidence": float("nan"),
            "identity_confirmation_joint_pass": 0,
            "identity_confirmation_dynamics_pass": 0,
            "identity_confirmation_reward_pass": 0,
            "identity_confirmation_reject_reason": "",
            "identity_confirmation_kind": "",
            "identity_confirmation_scope": "",
            "identity_confirmation_transaction_id": "",
            "identity_confirmation_source_context_id": -1,
            "identity_confirmation_candidate_context_id": -1,
            "identity_confirmation_source_revision": -1,
            "identity_confirmation_candidate_revision": -1,
            "identity_active_pnn_reward_mean": float("nan"),
            "identity_active_pnn_reward_variance": float("nan"),
            "identity_active_pnn_dynamics_variance": float("nan"),
            "confirmation_method": "",
            "natural_passed": 0,
            "effective_passed": 0,
            "residual_reference_ready": 0,
            "residual_reference_count": 0,
            "residual_center": float("nan"),
            "residual_tail_rank": float("nan"),
            "residual_direction": 0,
            "automatic_new_success": 0,
            "control_reuse_applied": self._metric_int(
                metrics.get("control_reuse_applied", 0), default=0
            ),
            "proposal_enabled": self._metric_int(
                metrics.get("proposal_enabled", 0), default=0
            ),
            "proposal_valid": self._metric_int(
                metrics.get("proposal_valid", 0), default=0
            ),
            "proposal_accepted": self._metric_int(
                metrics.get("proposal_accepted", 0), default=0
            ),
            "proposal_rejected_reason": str(
                metrics.get("proposal_rejected_reason", "")
            ),
            "proposal_context_id": self._metric_int(
                metrics.get("proposal_context_id", -1), default=-1
            ),
            "proposal_theta_version": self._metric_int(
                metrics.get("proposal_theta_version", -1), default=-1
            ),
            "proposal_model_version": self._metric_int(
                metrics.get("proposal_model_version", -1), default=-1
            ),
            "proposal_count": self._metric_int(
                metrics.get("proposal_count", 0), default=0
            ),
            "paired_plain_min_delta": self._metric_scalar(
                metrics.get("paired_plain_min_delta", float("nan"))
            ),
            "paired_plain_score_delta_per_head": str(
                metrics.get("paired_plain_score_delta_per_head", "[]")
            ),
            "plain_q_head_count": self._metric_int(
                metrics.get("plain_q_head_count", 0), default=0
            ),
            "plain_q_head_indices": str(metrics.get("plain_q_head_indices", "[]")),
            "paired_plain_score_shape": str(
                metrics.get("paired_plain_score_shape", "[]")
            ),
            "executed_action_source": str(
                metrics.get("executed_action_source", "plain")
            ),
            "proposal_sidecar_status": str(
                metrics.get("proposal_sidecar_status", "disabled")
            ),
            "proposal_sidecar_calls": self._metric_int(
                metrics.get("proposal_sidecar_calls", 0), default=0
            ),
            "proposal_sidecar_valid_count": self._metric_int(
                metrics.get("proposal_sidecar_valid_count", 0), default=0
            ),
            "proposal_mode": str(metrics.get("proposal_mode", "OFF")),
            "proposal_ranking_source": str(
                metrics.get("proposal_ranking_source", "disabled")
            ),
            "proposal_selected_index": self._metric_int(
                metrics.get("proposal_selected_index", -1), default=-1
            ),
            "proposal_used_recall": self._metric_int(
                metrics.get("proposal_used_recall", 0), default=0
            ),
            "proposal_theta_current_linf_delta": self._metric_scalar(
                metrics.get("proposal_theta_current_linf_delta", float("nan"))
            ),
            "control_compute_ms": self._metric_scalar(
                metrics.get("control_compute_ms", float("nan"))
            ),
            "proposal_compute_ms": self._metric_scalar(
                metrics.get("proposal_compute_ms", float("nan"))
            ),
            "proposal_input_copy_sync_ms": self._metric_scalar(
                metrics.get("proposal_input_copy_sync_ms", float("nan"))
            ),
            "proposal_sampling_ms": self._metric_scalar(
                metrics.get("proposal_sampling_ms", float("nan"))
            ),
            "proposal_ranking_ms": self._metric_scalar(
                metrics.get("proposal_ranking_ms", float("nan"))
            ),
            "proposal_plain_rescore_ms": self._metric_scalar(
                metrics.get("proposal_plain_rescore_ms", float("nan"))
            ),
            "proposal_other_ms": self._metric_scalar(
                metrics.get("proposal_other_ms", float("nan"))
            ),
            "proposal_total_compute_ms": self._metric_scalar(
                metrics.get("proposal_total_compute_ms", float("nan"))
            ),
        }

    def _write_online_step_trace(
        self,
        diagnostics,
        *,
        protocol_step,
        update_paused_pending,
        updates_before,
        recent_refreshed,
    ):
        self._write_context_likelihood_trace(diagnostics, protocol_step=protocol_step)
        created = (
            diagnostics["context_creation_count"]
            > self._last_event_counts["context_creation_count"]
        )
        recalled = (
            diagnostics["context_recall_count"]
            > self._last_event_counts["context_recall_count"]
        )
        triggered = bool(
            diagnostics["change_triggered"]
            or diagnostics["change_trigger_count"]
            > self._last_event_counts["change_trigger_count"]
        )
        proposed = (
            diagnostics["change_proposal_count"]
            > self._last_event_counts["change_proposal_count"]
        )
        for field in self._last_event_counts:
            self._last_event_counts[field] = diagnostics[field]
        calls_before, models_before, optimizers_before = updates_before
        row = {
            "step": int(protocol_step + 1),
            "protocol_step": int(protocol_step),
            "episode": int(self._ep_idx),
            **{
                key: diagnostics[key]
                for key in (
                    "reward",
                    "done",
                    "terminated",
                    "truncated",
                    "bootstrap_allowed",
                    "terminal_observation_is_post_step",
                    "forward_velocity",
                    "target_velocity",
                    "velocity_error",
                    "episode_age",
                    "steps_since_reset",
                    "next_terminal_distance",
                    "actual_task",
                    "active_context",
                    "K",
                    "context_creation_count",
                    "context_recall_count",
                    "change_trigger_count",
                    "change_proposal_count",
                    "pending_change_active",
                    "change_adaptive_threshold",
                    "change_null_mode_code",
                    "change_null_effective_threshold",
                    "change_null_ready",
                    "change_null_source_code",
                    "change_null_sample_count",
                    "change_null_percentile",
                    "change_null_tail_probability",
                    "change_null_frozen",
                    "control_actor_update_count",
                    "control_actor_update_applied",
                    "control_actor_q_std",
                    "control_actor_ema_penalty",
                    "identity_status",
                    "identity_error",
                    "identity_active_id",
                    "identity_active_context_id",
                    "identity_creation_count",
                    "identity_recall_count",
                    "identity_change_proposal_count",
                    "identity_last_action",
                    "identity_prediction_loss",
                    "identity_theta_norm",
                    "identity_theta_entropy",
                    "identity_feature_norm",
                    "identity_sidecar_updates",
                    "identity_replay_transitions",
                    "identity_sidecar_wall_ms",
                    "behavior_context_id",
                    "assigned_context_id",
                    "identity_row_disposition",
                    "pending_identity_rows",
                    "investigation_active",
                    "active_uncertain_diagnostic",
                    "active_disagreement",
                    "context_disagreements",
                    "identity_log_probs",
                    "identity_scores",
                    "identity_raw_scores",
                    "identity_new_score",
                    "identity_new_increment",
                    "identity_grouped_stats",
                    "identity_predictor_revisions",
                    "identity_applicability_shadow",
                    "identity_evidence_peak",
                    "identity_new_nomination_step",
                    "identity_confirmation_block_index",
                    "identity_confirmation_block_first_step",
                    "identity_confirmation_block_last_step",
                    "identity_confirmation_q_dynamics",
                    "identity_confirmation_q_reward",
                    "identity_confirmation_joint_evidence",
                    "identity_confirmation_dynamics_evidence",
                    "identity_confirmation_reward_evidence",
                    "identity_confirmation_joint_pass",
                    "identity_confirmation_dynamics_pass",
                    "identity_confirmation_reward_pass",
                    "identity_confirmation_reject_reason",
                    "identity_active_pnn_reward_mean",
                    "identity_active_pnn_reward_variance",
                    "identity_active_pnn_dynamics_variance",
                    "control_reuse_applied",
                    "proposal_enabled",
                    "proposal_valid",
                    "proposal_accepted",
                    "proposal_rejected_reason",
                    "proposal_context_id",
                    "proposal_theta_version",
                    "proposal_model_version",
                    "proposal_count",
                    "paired_plain_min_delta",
                    "paired_plain_score_delta_per_head",
                    "plain_q_head_count",
                    "plain_q_head_indices",
                    "paired_plain_score_shape",
                    "executed_action_source",
                    "proposal_sidecar_status",
                    "proposal_sidecar_calls",
                    "proposal_sidecar_valid_count",
                    "proposal_mode",
                    "proposal_ranking_source",
                    "proposal_selected_index",
                    "proposal_used_recall",
                    "proposal_theta_current_linf_delta",
                    "control_compute_ms",
                    "proposal_compute_ms",
                    "proposal_input_copy_sync_ms",
                    "proposal_sampling_ms",
                    "proposal_ranking_ms",
                    "proposal_plain_rescore_ms",
                    "proposal_other_ms",
                    "proposal_total_compute_ms",
                    *self._ACTIVE_CONTEXT_PROBE_FIELDS,
                    *self._CONTEXT_EXEMPLAR_AUDIT_FIELDS,
                    *self._REVERSIBLE_REUSE_TRIAL_FIELDS,
                )
            },
            "created": int(created),
            "recalled": int(recalled),
            "triggered": int(triggered),
            "proposed": int(proposed),
            "update_paused_pending": int(update_paused_pending),
            "agent_update_calls_this_step": int(
                self._agent_update_calls - calls_before
            ),
            "model_updates_this_step": int(self._model_updates - models_before),
            "optimizer_updates_this_step": int(
                self._optimizer_updates - optimizers_before
            ),
            "agent_update_calls": int(self._agent_update_calls),
            "model_updates": int(self._model_updates),
            "optimizer_updates": int(self._optimizer_updates),
            "pending_update_pauses": int(self._pending_update_pauses),
            "replay_ready": int(
                bool(getattr(self.buffer, "ready", self.buffer.num_eps > 0))
            ),
            "replay_valid_fraction": float(
                getattr(self.buffer, "last_sample_valid_fraction", 1.0)
            ),
            "replay_total_transitions": int(
                getattr(self.buffer, "total_transitions", 0)
            ),
            "replay_history_slices": int(
                getattr(self.buffer, "_last_sample_composition", {}).get(
                    "history_slices", 0
                )
            ),
            "replay_recent_slices": int(
                getattr(self.buffer, "_last_sample_composition", {}).get(
                    "recent_slices", 0
                )
            ),
            "replay_recent_refreshed": int(bool(recent_refreshed)),
            "replay_recent_refreshes": int(self._recent_replay_refreshes),
            "replay_transaction_pending": int(
                bool(getattr(self.buffer, "replay_transaction_pending", False))
            ),
            "replay_staged_transitions": int(
                getattr(self.buffer, "replay_staged_transitions", 0)
            ),
            "replay_active_context": self._metric_int(
                getattr(self.buffer, "replay_active_context_id", -1), default=-1
            ),
            "replay_context_switch_count": int(
                getattr(self.buffer, "replay_context_switch_count", 0)
            ),
            "main_prefill_pending": int(self._pending_control_transfer is not None),
            "main_prefill_recorded_count": int(
                len(self._pending_control_transfer.rows)
                if self._pending_control_transfer is not None
                else 0
            ),
            "main_prefill_applied_count": int(
                self._last_main_prefill_receipt.get("prefill_count", 0)
                if self._last_main_prefill_receipt is not None
                and int(self._last_main_prefill_receipt.get("commit_transition_id", -1))
                == int(protocol_step + 1)
                else 0
            ),
            "main_prefill_segment_count": int(
                len(self._last_main_prefill_receipt.get("prefill_segments", ()))
                if self._last_main_prefill_receipt is not None
                and int(self._last_main_prefill_receipt.get("commit_transition_id", -1))
                == int(protocol_step + 1)
                else 0
            ),
            "main_prefill_transaction_id": (
                self._pending_control_transfer.transaction_id
                if self._pending_control_transfer is not None
                else (
                    self._last_main_prefill_receipt.get("transaction_id", "")
                    if self._last_main_prefill_receipt is not None
                    and int(
                        self._last_main_prefill_receipt.get("commit_transition_id", -1)
                    )
                    == int(protocol_step + 1)
                    else ""
                )
            ),
            "main_prefill_origin_ids_sha256": (
                self._last_main_prefill_receipt.get("prefill_origin_ids_sha256", "")
                if self._last_main_prefill_receipt is not None
                and int(self._last_main_prefill_receipt.get("commit_transition_id", -1))
                == int(protocol_step + 1)
                else ""
            ),
            "main_prefill_target_context_id": int(
                self._last_main_prefill_receipt.get("target_context_id", -1)
                if self._last_main_prefill_receipt is not None
                and int(self._last_main_prefill_receipt.get("commit_transition_id", -1))
                == int(protocol_step + 1)
                else -1
            ),
        }
        for key in (
            "context_disagreements",
            "identity_log_probs",
            "identity_scores",
            "identity_raw_scores",
            "identity_grouped_stats",
            "identity_predictor_revisions",
            "identity_applicability_shadow",
        ):
            value = row.get(key, "{}")
            if isinstance(value, str):
                try:
                    json.loads(value)
                except (TypeError, ValueError):
                    row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                row[key] = json.dumps(value or {}, ensure_ascii=False, sort_keys=True)
        self._append_csv(self._online_steps_path, self._ONLINE_STEP_FIELDS, row)
        self._episode_diagnostics.append(dict(row))

    def _write_context_likelihood_trace(self, diagnostics, *, protocol_step):
        raw = diagnostics.get("identity_log_probs", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                raw = {}
        if not isinstance(raw, dict):
            raw = {}

        def value(context_id):
            candidate = raw.get(str(context_id), raw.get(context_id, float("nan")))
            try:
                return float(candidate)
            except (TypeError, ValueError):
                return float("nan")

        row = {
            "step": int(protocol_step + 1),
            "protocol_step": int(protocol_step),
            "episode": int(self._ep_idx),
            "actual_task": str(diagnostics.get("actual_task", "")),
            "active_context": int(diagnostics.get("active_context", -1)),
            "K": int(diagnostics.get("K", 0)),
            **{f"log_prob_C{i}": value(i) for i in range(8)},
            "active_log_prob": (
                value(int(diagnostics.get("active_context", -1)))
                if int(diagnostics.get("active_context", -1)) >= 0
                else float("nan")
            ),
            "identity_new_score": diagnostics.get("identity_new_score", float("nan")),
            "identity_active_disagreement": diagnostics.get(
                "active_disagreement", float("nan")
            ),
        }
        self._append_csv(
            self._context_likelihood_path, self._context_likelihood_fields, row
        )

    def _write_m3wm_theta_trace(self, reward):
        if self._m3wm_trace_path is None:
            return
        current_step = int(self._step + 1)
        shadow_due = self._is_source_theta_shadow_due_event(
            getattr(self.agent, "_m3wm_last_source_theta_shadow", None), current_step
        )
        if current_step % self._m3wm_trace_freq != 0 and (not shadow_due):
            return
        metrics = self._context_metrics()
        row = {
            "step": current_step,
            "episode": int(self._ep_idx),
            "reward": self._metric_scalar(reward),
        }
        for field in self._m3wm_trace_fields:
            if field in row:
                continue
            row[field] = self._metric_scalar(metrics.get(field, float("nan")))
        self._append_csv(self._m3wm_trace_path, self._m3wm_trace_fields, row)

    def common_metrics(self):
        elapsed_time = time() - self._start_time
        metrics = dict(
            step=self._step,
            episode=self._ep_idx,
            elapsed_time=elapsed_time,
            steps_per_second=self._step / elapsed_time if elapsed_time else 0.0,
            agent_update_calls=self._agent_update_calls,
            model_updates=self._model_updates,
            optimizer_updates=self._optimizer_updates,
            pending_update_pauses=self._pending_update_pauses,
            trial_update_pauses=self._trial_update_pauses,
            recent_replay_refreshes=self._recent_replay_refreshes,
        )
        if self.context_bank is not None:
            metrics.update(
                {
                    "context_bank_active_id": int(self.context_bank.active_context_id),
                    "context_bank_count": len(self.context_bank.context_ids()),
                    "context_bank_replay_segment": int(
                        self.context_bank.replay_bank.segment_id
                    ),
                    "context_bank_replay_rows": int(
                        self.context_bank.replay_bank.total_rows
                    ),
                    "context_bank_switch_count": int(
                        self.context_bank.replay_bank.switch_count
                    ),
                }
            )
        if self.trial_controller is not None:
            metrics.update(
                {
                    "trial_active": int(self.trial_controller.active),
                    "trial_steps_total": int(self.trial_controller.trial_steps_total),
                    "trial_discarded_training_rows": int(
                        self.trial_controller.discarded_training_rows
                    ),
                }
            )
        return metrics

    def _write_context_agent_audit_summary(self):
        if self.context_selector is None:
            return None
        payload = {
            "schema": "m3wm-context-agent-audit-v1",
            "run_id": str(self._run_status_path.parent.name),
            "step": int(self._step),
            "selector": (
                self.context_selector.audit_metrics()
                if callable(getattr(self.context_selector, "audit_metrics", None))
                else {}
            ),
            "trial": (
                self.trial_controller.audit_metrics()
                if self.trial_controller is not None
                and callable(getattr(self.trial_controller, "audit_metrics", None))
                else {}
            ),
            "active_context_id": int(self.context_bank.active_context_id),
            "context_ids": [
                int(context_id) for context_id in self.context_bank.context_ids()
            ],
        }
        path = self._work_dir / "context_agent_audit.json"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def _write_episode_trace(self, episode_return):
        rows = self._episode_diagnostics
        if not rows:
            return
        contexts = self._ordered_unique([str(row["active_context"]) for row in rows])
        tasks = self._ordered_unique([row["actual_task"] for row in rows])
        row = {
            "episode": int(self._ep_idx),
            "episode_start_step": int(rows[0]["step"] - 1),
            "episode_end_step": int(rows[-1]["step"]),
            "episode_return": self._metric_scalar(episode_return),
            "episode_reward_mean": self._nanmean(rows, "reward"),
            "episode_length": int(len(rows)),
            "episode_terminated": int(bool(rows[-1].get("terminated", 0))),
            "episode_truncated": int(bool(rows[-1].get("truncated", 0))),
            "forward_velocity_mean": self._nanmean(rows, "forward_velocity"),
            "target_velocity_mean": self._nanmean(rows, "target_velocity"),
            "velocity_error_mean": self._nanmean(rows, "velocity_error"),
            "actual_task": "|".join(tasks),
            "active_contexts": "|".join(contexts),
            "active_context_end": int(rows[-1]["active_context"]),
            "K_end": int(rows[-1]["K"]),
            "contexts_created": int(sum((r["created"] for r in rows))),
            "contexts_recalled": int(sum((r["recalled"] for r in rows))),
            "recent_replay_refreshes": int(
                sum((r["replay_recent_refreshed"] for r in rows))
            ),
            "changes_triggered": int(sum((r["triggered"] for r in rows))),
            "changes_proposed": int(sum((r["proposed"] for r in rows))),
            "agent_update_calls": int(self._agent_update_calls),
            "model_updates": int(self._model_updates),
            "optimizer_updates": int(self._optimizer_updates),
            "pending_update_pauses": int(self._pending_update_pauses),
        }
        self._append_csv(self._episode_trace_path, self._EPISODE_FIELDS, row)

    def _complete_episode(self, info, train_metrics):
        if info.get("terminated", False) and (not self.cfg.episodic):
            raise ValueError(
                "Termination detected but episodic=false. Set episodic=true to enable environment terminations."
            )
        episode_reward = torch.stack(
            [td["reward"].reshape(()) for td in self._tds[1:]]
        ).sum()
        train_metrics.update(
            episode_reward=episode_reward,
            episode_success=info.get("success", 0.0),
            episode_length=len(self._tds) - 1,
            episode_terminated=info.get("terminated", False),
            episode_truncated=info.get("truncated", False),
        )
        self._write_episode_trace(episode_reward)
        train_metrics.update(self.common_metrics())
        if hasattr(self.agent, "context_metrics"):
            train_metrics.update(self.agent.context_metrics())
        self.logger.log(train_metrics, "train")
        if bool(getattr(self.buffer, "streaming_enabled", False)):
            if getattr(self.buffer, "_active_episode_id", None) is not None:
                self.buffer.end_episode()
            if self.context_bank is not None:
                self._ep_idx += 1
            else:
                self._ep_idx = self.buffer.num_eps
        else:
            self._ep_idx = self.buffer.add(torch.cat(self._tds))
        self._episode_diagnostics = []

    def eval(self):
        ep_rewards, ep_successes, ep_lengths = ([], [], [])
        context_state = (
            self.agent.export_train_state()
            if hasattr(self.agent, "export_train_state")
            else None
        )
        for i in range(self.cfg.eval_episodes):
            if context_state is not None:
                self.agent.import_train_state(context_state)
            obs, done, ep_reward, t = (self.env.reset(), False, 0, 0)
            while not done:
                _safe_cudagraph_mark_step_begin()
                action = self._act_with_optional_proposal(
                    obs, t0=t == 0, eval_mode=True
                )
                prev_obs = obs
                obs, reward, done, info = self.env.step(action)
                info = dict(info)
                info.update(control_contract(done, info).as_info())
                if hasattr(self.agent, "observe_transition"):
                    self.agent.observe_transition(
                        prev_obs,
                        action,
                        reward,
                        obs,
                        terminal=bool(info.get("terminated", False)),
                    )
                self._drain_v57_shadow_sidecars()
                ep_reward += reward
                t += 1
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get("success", 0.0))
            ep_lengths.append(t)
        if context_state is not None:
            self.agent.import_train_state(context_state)
        metrics = dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )
        if hasattr(self.agent, "context_metrics"):
            metrics.update(self.agent.context_metrics())
        return metrics

    def to_td(
        self,
        obs,
        action=None,
        reward=None,
        terminated=None,
        truncated=None,
        origin_transition_id=None,
    ):
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        elif not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32)
        if reward is None:
            reward = torch.tensor(float("nan"))
        elif not torch.is_tensor(reward):
            reward = torch.tensor(float(reward), dtype=torch.float32)
        if terminated is None:
            terminated = torch.tensor(float("nan"))
        elif not torch.is_tensor(terminated):
            terminated = torch.tensor(float(terminated), dtype=torch.float32)
        if truncated is None:
            truncated = torch.tensor(float("nan"))
        elif not torch.is_tensor(truncated):
            truncated = torch.tensor(float(truncated), dtype=torch.float32)
        reward = reward.reshape(())
        terminated = terminated.reshape(())
        truncated = truncated.reshape(())
        origin_transition_id = (
            -1 if origin_transition_id is None else int(origin_transition_id)
        )
        return TensorDict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            terminated=terminated.unsqueeze(0),
            truncated=truncated.unsqueeze(0),
            origin_transition_id=torch.tensor(
                [origin_transition_id], dtype=torch.int64
            ),
            batch_size=(1,),
        )

    def train(self):
        train_metrics, done, eval_next = ({}, True, False)
        info = {}
        while self._step < self.cfg.steps:
            self._set_env_global_step(self._step)
            if done:
                if self._step > 0:
                    self._complete_episode(info, train_metrics)
                    if self.identity_shadow is not None:
                        theta_buffer = getattr(
                            self.identity_shadow, "side_buffer", None
                        )
                        if (
                            getattr(theta_buffer, "_active_episode_id", None)
                            is not None
                        ):
                            self.identity_shadow.end_episode()
                if self.identity_shadow is not None:
                    self.identity_shadow.reset_episode()
                if hasattr(self.agent, "reset_context"):
                    reset_metrics_before = self._context_metrics()
                    self.agent.reset_context(clear_memory=False)
                    self._drain_v57_shadow_sidecars()
                    reset_metrics_after = self._context_metrics()
                    self._sync_transactional_replay(
                        reset_metrics_before, reset_metrics_after
                    )
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False
                obs = self.env.reset()
                self._episode_step = 0
                self._set_env_global_step(self._step)
                self._tds = [self.to_td(obs)]
                if bool(getattr(self.buffer, "streaming_enabled", False)):
                    self.buffer.begin_episode(self._tds[0])
                    if self.identity_shadow is not None:
                        self.identity_shadow.begin_episode_td(self._tds[0])
            trial_was_active = bool(
                self.trial_controller is not None and self.trial_controller.active
            )
            behavior_context_id = None
            if self.context_bank is not None:
                behavior_context_id = int(
                    self.trial_controller.behavior_context_id
                    if trial_was_active
                    else self.context_bank.active_context_id
                )
            if self.context_bank is not None:
                self._record_protocol_label(protocol_step=int(self._step))
                trial_was_active = bool(
                    self.trial_controller is not None and self.trial_controller.active
                )
                behavior_context_id = int(
                    self.trial_controller.behavior_context_id
                    if trial_was_active
                    else self.context_bank.active_context_id
                )
            if self._online_resume or self._step >= self.cfg.seed_steps:
                action = self._act_with_optional_proposal(
                    obs, t0=len(self._tds) == 1 or self._context_switch_t0_pending
                )
                self._context_switch_t0_pending = False
            else:
                action = self.env.rand_act()
            protocol_step = int(self._step)
            updates_before = (
                self._agent_update_calls,
                self._model_updates,
                self._optimizer_updates,
            )
            prev_obs = obs
            identity_action = (
                action.detach().clone()
                if torch.is_tensor(action)
                else np.asarray(action, dtype=np.float32).copy()
            )
            obs, reward, done, info = self.env.step(action)
            info = dict(info)
            info.update(control_contract(done, info).as_info())
            transition_td = self.to_td(
                obs,
                action,
                reward,
                info.get("terminated", False),
                info.get("truncated", False),
                origin_transition_id=self._step + 1,
            )
            route_receipt = None
            main_prefill_bound = False
            if self.context_bank is not None:
                self.context_bank.set_global_env_step(self._step + 1)
                if self.context_router is None:
                    raise RuntimeError("PNN Context router is required")
                else:
                    selector_row = self._selector_transition_row(
                        transition_step=self._step + 1,
                        prev_obs=prev_obs,
                        action=identity_action,
                        reward=reward,
                        next_obs=obs,
                        info=info,
                    )
                    control_payload = {
                        "transition_td": transition_td.detach().clone().cpu(),
                        "prev_obs": torch.as_tensor(prev_obs).detach().cpu().clone(),
                        "episode_id": int(self._ep_idx),
                    }
                    route_receipt = self.context_router.route(
                        selector_row, control_payload=control_payload
                    )
                    main_prefill_bound = self._handle_main_prefill_route(
                        route_receipt,
                        selector_row=selector_row,
                        control_payload=control_payload,
                        commit_prev_obs=prev_obs,
                    )
                    self._last_context_decision = route_receipt.decision
                    self._write_context_fit_records()
                    self._write_context_confirmation_receipt(
                        route_receipt.decision, step=self._step + 1
                    )
                    if route_receipt.bank_switch_receipt is not None:
                        self._write_context_switch_receipt(
                            route_receipt.decision, route_receipt.bank_switch_receipt
                        )
                    if route_receipt.trial_resolution is not None:
                        resolution = route_receipt.trial_resolution
                        if resolution.outcome in {"NEW", "RETURN"}:
                            self._flush_successful_trial(resolution)
                        else:
                            self._restore_after_failed_trial(current_obs=obs, done=done)
                        self._write_trial_records()
                    elif route_receipt.bank_switch_receipt is not None and (
                        not main_prefill_bound
                    ):
                        self._bind_context_bundle(
                            self.context_bank.active_bundle, seed_obs=prev_obs
                        )
                        if route_receipt.decision.action == "RETURN":
                            self.identity_shadow.mark_context_return()
            metrics_before_observation = self._context_metrics()
            if not trial_was_active and hasattr(self.agent, "observe_transition"):
                self.agent.observe_transition(
                    prev_obs,
                    action,
                    reward,
                    obs,
                    terminal=bool(info.get("terminated", False)),
                )
            if not trial_was_active and self.identity_shadow is not None:
                self.identity_shadow.observe_transition(
                    prev_obs,
                    identity_action,
                    reward,
                    obs,
                    terminated=bool(info.get("terminated", False)),
                    truncated=bool(info.get("truncated", False)),
                    episode_id=self._ep_idx,
                )
            self._drain_v57_shadow_sidecars()
            self._persist_context_identity_event_artifact()
            metrics = self._context_metrics()
            if self.context_bank is not None and (not trial_was_active):
                self.context_bank.append_transition(
                    transition_td,
                    behavior_context_id=int(behavior_context_id),
                    assigned_context_id=int(self.context_bank.active_context_id),
                    episode_id=int(self._ep_idx),
                    terminated=bool(info.get("terminated", False)),
                    truncated=bool(info.get("truncated", False)),
                )
            self._tds.append(transition_td)
            if not trial_was_active and self.identity_shadow is not None:
                self.identity_shadow.append_transition_td(transition_td)
            if trial_was_active:
                recent_refreshed = False
            elif bool(getattr(self.buffer, "streaming_enabled", False)):
                recent_refreshed = self._sync_transactional_replay(
                    metrics_before_observation, metrics, transition_td
                )
            else:
                recent_refreshed = self._maybe_refresh_recent_replay(metrics)
            replay_ready = bool(getattr(self.buffer, "ready", self.buffer.num_eps > 0))
            updates_eligible = bool(
                replay_ready
                and (self._online_resume or self._step >= self.cfg.seed_steps)
            )
            pending_change = self._agent_has_pending_change(metrics)
            update_paused_pending = bool(
                updates_eligible
                and pending_change
                and self._pause_updates_during_pending
            )
            update_paused_trial = bool(trial_was_active)
            if update_paused_trial:
                self._trial_update_pauses += 1
            elif update_paused_pending:
                self._pending_update_pauses += 1
            elif updates_eligible:
                num_updates = 1
                for _ in range(1):
                    _train_metrics = self.agent.update(self.buffer)
                    self._record_successful_agent_update()
                    if (
                        self._last_main_prefill_receipt is not None
                        and self._last_main_prefill_receipt.get(
                            "first_sample_unique_transition_ids"
                        )
                        is None
                        and (
                            int(
                                self._last_main_prefill_receipt.get(
                                    "target_context_id", -1
                                )
                            )
                            == int(self.context_bank.active_context_id)
                        )
                    ):
                        first_ids = list(
                            getattr(
                                self.buffer, "last_sample_unique_transition_ids", ()
                            )
                        )
                        first_rows = int(
                            getattr(self.buffer, "last_sample_real_loss_rows", 0)
                        )
                        self._last_main_prefill_receipt[
                            "first_sample_unique_transition_ids"
                        ] = first_ids
                        self._last_main_prefill_receipt[
                            "first_sample_real_loss_rows"
                        ] = first_rows
                        first_receipt = dict(self._last_main_prefill_receipt)
                        first_receipt["event"] = "FIRST_UPDATE_SAMPLE"
                        first_receipt["agent_update_call"] = int(
                            self._agent_update_calls
                        )
                        self._write_main_prefill_record(first_receipt)
                train_metrics.update(_train_metrics)
                if self.identity_shadow is not None:
                    _plain_cpu_rng = torch.random.get_rng_state()
                    _plain_cuda_rng = (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    )
                    try:
                        self.identity_shadow.update()
                    finally:
                        torch.random.set_rng_state(_plain_cpu_rng)
                        if _plain_cuda_rng is not None:
                            torch.cuda.set_rng_state_all(_plain_cuda_rng)
                    if bool(
                        getattr(self.identity_shadow, "full_tdmpc2_enabled", False)
                    ) and bool(getattr(self.identity_shadow, "failed", False)):
                        raise RuntimeError(
                            f"registered Proposal-theta learner update failed: {self.identity_shadow.last_error}"
                        )
            metrics = self._context_metrics()
            diagnostics = self._step_diagnostics(info, reward, metrics, done=done)
            if route_receipt is not None:
                decision_record = route_receipt.decision.to_record()
                grouped = decision_record.get("grouped_stats", {}) or {}
                active_key = str(
                    decision_record.get(
                        "source_context_id", decision_record.get("transition_id", 0)
                    )
                )
                active_stats = (
                    grouped.get(active_key, {}) if isinstance(grouped, dict) else {}
                )
                block = decision_record.get("confirmation_block") or {}
                nomination_step = decision_record.get("new_nomination_step")
                if isinstance(block, dict) and block:
                    q_d = block.get(
                        "candidate_mean_q_dynamics", block.get("source_mean_q_dynamics")
                    )
                    q_r = block.get(
                        "candidate_mean_q_reward", block.get("source_mean_q_reward")
                    )
                    joint = block.get("mean_llr", block.get("mean_new_increment"))
                    block_index = block.get("block_index", -1)
                    block_first = block.get("block_first_step", -1)
                    block_last = block.get("block_last_step", -1)
                    if nomination_step is None and isinstance(block, dict):
                        nomination_step = block.get("nomination_step")
                    reject_reason = block.get("reject_reason", [])
                else:
                    q_d = active_stats.get("q_dynamics_mean")
                    q_r = active_stats.get("q_reward")
                    joint = None
                    block_index = block_first = block_last = -1
                    reject_reason = []
                try:
                    q_d_value = float(q_d)
                    q_r_value = float(q_r)
                    dyn_evidence = q_d_value - 4.0
                    rew_evidence = q_r_value - 4.0
                    joint_evidence = 0.5 * (17.0 * q_d_value + q_r_value - 72.0)
                    joint_pass = int(joint_evidence > 0.0)
                    dyn_pass = int(q_d_value <= 4.0)
                    rew_pass = int(q_r_value <= 4.0)
                except (TypeError, ValueError):
                    q_d_value = q_r_value = dyn_evidence = rew_evidence = (
                        joint_evidence
                    ) = float("nan")
                    joint_pass = dyn_pass = rew_pass = 0
                diagnostics.update(
                    {
                        "identity_new_nomination_step": int(
                            -1 if nomination_step is None else nomination_step
                        ),
                        "identity_confirmation_block_index": int(
                            -1 if block_index is None else block_index
                        ),
                        "identity_confirmation_block_first_step": int(
                            -1 if block_first is None else block_first
                        ),
                        "identity_confirmation_block_last_step": int(
                            -1 if block_last is None else block_last
                        ),
                        "identity_confirmation_q_dynamics": q_d_value,
                        "identity_confirmation_q_reward": q_r_value,
                        "identity_confirmation_joint_evidence": joint_evidence,
                        "identity_confirmation_dynamics_evidence": dyn_evidence,
                        "identity_confirmation_reward_evidence": rew_evidence,
                        "identity_confirmation_joint_pass": joint_pass,
                        "identity_confirmation_dynamics_pass": dyn_pass,
                        "identity_confirmation_reward_pass": rew_pass,
                        "identity_confirmation_reject_reason": json.dumps(
                            reject_reason or [], ensure_ascii=False, sort_keys=True
                        ),
                        "identity_confirmation_kind": str(
                            block.get("kind", "") if isinstance(block, dict) else ""
                        ),
                        "identity_confirmation_scope": "block" if block else "",
                        "identity_confirmation_transaction_id": str(
                            decision_record.get("new_transaction_id", "")
                        ),
                        "identity_confirmation_source_context_id": int(
                            (
                                -1
                                if block.get("source_context_id") is None
                                else block["source_context_id"]
                            )
                            if isinstance(block, dict)
                            else -1
                        ),
                        "identity_confirmation_candidate_context_id": int(
                            (
                                -1
                                if block.get("candidate_context_id") is None
                                else block["candidate_context_id"]
                            )
                            if isinstance(block, dict)
                            else -1
                        ),
                        "identity_confirmation_source_revision": int(
                            (
                                -1
                                if block.get("source_revision") is None
                                else block["source_revision"]
                            )
                            if isinstance(block, dict)
                            else -1
                        ),
                        "identity_confirmation_candidate_revision": int(
                            (
                                -1
                                if block.get("candidate_revision") is None
                                else block["candidate_revision"]
                            )
                            if isinstance(block, dict)
                            else -1
                        ),
                        "identity_active_pnn_reward_mean": active_stats.get(
                            "reward_mean", float("nan")
                        ),
                        "identity_active_pnn_reward_variance": active_stats.get(
                            "reward_variance",
                            active_stats.get("reward_total_variance", float("nan")),
                        ),
                        "identity_active_pnn_dynamics_variance": active_stats.get(
                            "dynamics_variance_mean", float("nan")
                        ),
                    }
                )
                diagnostics.update(
                    {
                        "behavior_context_id": int(route_receipt.behavior_context_id),
                        "assigned_context_id": (
                            None
                            if route_receipt.assigned_context_id is None
                            else int(route_receipt.assigned_context_id)
                        ),
                        "identity_row_disposition": str(
                            route_receipt.identity_row_disposition
                        ),
                        "pending_identity_rows": int(
                            route_receipt.pending_identity_rows
                        ),
                        "investigation_active": bool(
                            decision_record.get("investigation_active", False)
                        ),
                        "active_uncertain_diagnostic": bool(
                            decision_record.get("active_uncertain_diagnostic", False)
                        ),
                        "active_disagreement": decision_record.get(
                            "active_disagreement"
                        ),
                        "context_disagreements": decision_record.get(
                            "context_disagreements", {}
                        ),
                        "identity_log_probs": decision_record.get("log_probs", {}),
                        "identity_scores": decision_record.get("scores", {}),
                        "identity_raw_scores": decision_record.get("raw_scores", {}),
                        "identity_new_score": decision_record.get("new_score"),
                        "identity_new_increment": decision_record.get("new_increment"),
                        "identity_grouped_stats": decision_record.get(
                            "grouped_stats", {}
                        ),
                        "identity_predictor_revisions": decision_record.get(
                            "predictor_revisions", {}
                        ),
                        "identity_applicability_shadow": decision_record.get(
                            "applicability_shadow", {}
                        ),
                        "identity_evidence_peak": decision_record.get("evidence_peak"),
                        "confirmation_method": decision_record.get(
                            "confirmation_method"
                        ),
                        "natural_passed": decision_record.get("natural_passed"),
                        "effective_passed": decision_record.get("effective_passed"),
                        "residual_reference_ready": decision_record.get(
                            "residual_reference_ready"
                        ),
                        "residual_reference_count": decision_record.get(
                            "residual_reference_count"
                        ),
                        "residual_center": decision_record.get("residual_center"),
                        "residual_tail_rank": decision_record.get("residual_tail_rank"),
                        "residual_direction": decision_record.get("residual_direction"),
                        "automatic_new_success": decision_record.get(
                            "automatic_new_success"
                        ),
                    }
                )
                selector_audit = self.context_selector.audit_metrics()
                diagnostics.update(
                    {
                        "active_context": int(self.context_bank.active_context_id),
                        "K": int(len(self.context_bank.context_ids())),
                        "context_creation_count": int(
                            selector_audit.get("new_count", 0)
                        ),
                        "context_recall_count": int(
                            selector_audit.get("return_count", 0)
                        ),
                        "identity_active_context_id": int(
                            self.context_bank.active_context_id
                        ),
                        "identity_creation_count": int(
                            selector_audit.get("new_count", 0)
                        ),
                        "identity_recall_count": int(
                            selector_audit.get("return_count", 0)
                        ),
                        "identity_last_action": str(
                            decision_record.get("action", "UNKNOWN")
                        ),
                    }
                )
            self._write_m3wm_theta_trace(reward)
            self._write_online_step_trace(
                diagnostics,
                protocol_step=protocol_step,
                update_paused_pending=update_paused_pending,
                updates_before=updates_before,
                recent_refreshed=recent_refreshed,
            )
            self._episode_step += 1
            if (
                route_receipt is not None
                and route_receipt.trial_requested
                and (not bool(done))
                and self.context_router.start_trial(
                    route_receipt.decision, env_steps_seen=self._step + 1
                )
            ):
                self._activate_trial_behavior()
            else:
                self._write_trial_records()
            self._step += 1
            self._write_run_status_progress()
        if self._pending_control_transfer is not None:
            self._write_main_prefill_record(
                self._pending_control_transfer.audit_record(
                    event="DISCARDED", reason="run_step_budget_reached"
                )
            )
            self.context_bank.discard_pending_init_capsule(
                reason="run_step_budget_reached",
                transaction_id=self._pending_control_transfer.transaction_id,
            )
            self._pending_control_transfer = None
        if self.trial_controller is not None and self.trial_controller.active:
            resolution = self.trial_controller.resolve(
                end_step=int(self._step),
                outcome="ABORTED",
                reason="run_step_budget_reached",
            )
            self._restore_after_failed_trial(current_obs=obs, done=done)
            self._write_trial_records()
        finalize_identity = getattr(self.agent, "pure_online_finalize_terminal", None)
        if callable(finalize_identity):
            finalize_identity()
        terminal_metrics_before = self._context_metrics()
        terminal_replay_before = self._replay_terminal_state(terminal_metrics_before)
        terminal_cleanup_performed = bool(terminal_replay_before["buffer_pending"])
        terminal_receipt = None
        if terminal_cleanup_performed:
            if not hasattr(self.agent, "reset_context"):
                raise RuntimeError(
                    "Cannot close terminal replay proposal without agent reset_context"
                )
            abort_pending = getattr(
                self.agent, "_m3wm_abort_pending_at_episode_reset", None
            )
            if callable(abort_pending):
                abort_pending()
            self._last_replay_transaction_receipt = None
            self.agent.reset_context(clear_memory=False)
            self._drain_v57_shadow_sidecars()
            terminal_metrics_after = self._context_metrics()
            self._sync_transactional_replay(
                terminal_metrics_before, terminal_metrics_after
            )
            terminal_receipt = self._last_replay_transaction_receipt
            if terminal_receipt is None:
                raise RuntimeError("Terminal replay proposal produced no receipt")
            if bool(getattr(self.buffer, "replay_transaction_pending", False)):
                raise RuntimeError(
                    "Terminal replay proposal did not close exactly once"
                )
        if done and self._tds and (len(self._tds) > 1):
            self._complete_episode(info, train_metrics)
        elif bool(getattr(self.buffer, "streaming_enabled", False)):
            self.buffer.end_episode()
        self._write_replay_terminal_ledger(
            pre_cleanup=terminal_replay_before,
            cleanup_performed=terminal_cleanup_performed,
            receipt=terminal_receipt,
        )
        self._write_pure_online_runtime_artifact()
        self._assert_v57_shadow_persistence_complete()
        if self.identity_shadow is not None:
            self.identity_shadow.finalize(self._work_dir)
        if self.context_bank is not None:
            if self.context_router is not None:
                self.context_router.sync_active_predictor_state()
            self.context_bank.finalize()
            self._write_context_agent_audit_summary()
        self.logger.finish(self.agent)
        self._close_trace_files()
        self._write_run_status_progress(
            force=True,
            status="completed",
            completed=True,
            stop_reason="step_budget_reached",
        )
