from __future__ import annotations
from typing import Any

SCHEMA = "m3wm_factorized_physical_law_id_shadow"
VERSION = 2
SELECTION_LENGTH = 32
VALIDATION_PANEL_LENGTH = 64
TOTAL_LENGTH = SELECTION_LENGTH + 2 * VALIDATION_PANEL_LENGTH
FACTORS = ("dyn_gain", "dyn_drift", "reward_law")
MULTISTEP_HORIZON = 8
PANEL_EPOCH_SCHEMA = "m3wm_factorized_probe_panel_epoch"
PANEL_EPOCH_VERSION = 1
REFERENCE_SCHEMA = "m3wm_factorized_law_shadow_reference_carrier"
REFERENCE_VERSION = 1
REFERENCE_MAX_ROWS_PER_CONTEXT = 512
REFERENCE_ROW_POLICY = "core_plus_coverage_representatives"
MATCHED_MIN_ROWS_PER_PANEL = 8
MATCHED_RADIUS_QUANTILE = 0.95
MATCHED_RADIUS_SCALE = 1.5
MATCHED_RADIUS_FLOOR = 1.0
MATCHED_RADIUS_CAP = 8.0


def empty_shadow_reference_state() -> dict[str, Any]:
    return {
        "schema": REFERENCE_SCHEMA,
        "version": REFERENCE_VERSION,
        "max_rows_per_context": REFERENCE_MAX_ROWS_PER_CONTEXT,
        "contexts": {},
    }
