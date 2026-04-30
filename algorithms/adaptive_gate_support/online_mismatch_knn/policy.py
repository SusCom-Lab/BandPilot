"""Policy helpers shared by BandPilot online-kNN replay paths.

The helpers here are intentionally small and deterministic: they convert an
online mismatch-risk row into a skip/trigger decision and decide which observed
query rows should be admitted into the next bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from algorithms.adaptive_gate_support.utils import _as_bool, _as_float, _as_int


@dataclass(frozen=True)
class ActivationCriteria:
    """Bank-level criteria for enabling online-kNN decisions."""

    unsafe_skip_rate_pct_max: float = 0.0
    over_trigger_rate_pct_max: float = 50.0
    support_insufficient_case_count_max: int = 0
    min_consecutive_pass_banks: int = 2
    sticky_activation: bool = True


def _evaluate_shadow_mismatch(
    *,
    sample: Mapping[str, object],
    online_risk_row: Mapping[str, object],
    risk_threshold: float,
) -> Tuple[bool, str]:
    """Return whether EHA can safely skip PTS under the online mismatch model."""

    if not _as_bool(sample.get("eha_feasible", False)):
        return False, "eha_infeasible"
    if _as_int(sample.get("eha_node_count", 0)) <= 1:
        return True, "fast_path_single_node"
    if _as_bool(online_risk_row.get("support_insufficient", False)):
        return False, "support_insufficient_high_risk"
    if _as_bool(online_risk_row.get("online_mismatch_low_trust", False)):
        return False, "low_trust_high_risk"

    risk_value = _as_float(online_risk_row.get("online_mismatch_knn_risk", 1.0))
    skip_pts = risk_value <= float(risk_threshold)
    return skip_pts, "low_online_risk" if skip_pts else "high_online_risk"


def _build_admission_rows(
    *,
    bank_version: int,
    bank_phase: str,
    query_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Select query rows that can be admitted as labeled bank support."""

    admitted_rows: List[Dict[str, object]] = []
    for query_row in query_rows:
        if bank_phase == "pre_activation":
            should_admit = True
            admission_phase = "pre_activation_all_bandpilot"
        else:
            should_admit = (
                (not _as_bool(query_row.get("formal_skip_pts", False)))
                and _as_bool(query_row.get("eha_feasible", False))
            )
            admission_phase = "post_activation_dual_observed_only"

        if not should_admit:
            continue

        admitted_rows.append(
            {
                **dict(query_row),
                "source_bank_version": bank_version,
                "admission_phase": admission_phase,
                "dual_observed": bool(_as_bool(query_row.get("eha_feasible", False))),
            }
        )
    return admitted_rows
