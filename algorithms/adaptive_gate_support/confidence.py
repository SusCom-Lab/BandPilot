"""Confidence-feature construction for BandPilot online-kNN decisions.

The main compare and scalability paths need a compact feature row describing
how safe it is for EHA to skip PTS. This module keeps that feature builder
separate from historical calibration scripts so public code can import it
without pulling in internal benchmark runners.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from algorithms.adaptive_gate_support.utils import (
    _as_bool,
    _as_float,
    _as_int,
    _as_json_list,
)


_MAD_SCALE = 1.4826
_DEFAULT_MAX_EHA_CANDIDATES = 200


@dataclass(frozen=True)
class ConfidenceRuleCandidate:
    """Threshold rule used by legacy offline diagnostics.

    The runtime path only needs feature construction, but keeping this compact
    data class preserves compatibility with existing report code that records
    confidence-rule identifiers.
    """

    z_margin_threshold: float
    p_stable_threshold: float
    coverage_threshold: float
    require_not_budget_binding: bool = True

    @property
    def rule_id(self) -> str:
        """Return a stable identifier for CSV/report rows."""

        budget_suffix = "nb1" if self.require_not_budget_binding else "nb0"
        return (
            f"zm{self.z_margin_threshold:.2f}_"
            f"ps{self.p_stable_threshold:.2f}_"
            f"cov{self.coverage_threshold:.2f}_"
            f"{budget_suffix}"
        )


def _safe_median(values: Sequence[float]) -> float:
    """Return a median value, or zero when no values are available."""

    if not values:
        return 0.0
    return float(statistics.median(values))


def _robust_sigma_hat(scores: Sequence[float], *, relative_noise_floor: float) -> float:
    """Estimate score noise from top-k bandwidth predictions.

    The estimator combines median absolute deviation with a relative floor based
    on the top score, preventing zero variance when the predictor emits nearly
    identical candidate scores.
    """

    if not scores:
        return 1.0

    top1 = float(scores[0])
    floor = max(abs(top1), 1.0) * float(relative_noise_floor)
    if len(scores) == 1:
        return max(floor, 1e-9)

    median = _safe_median(scores)
    mad = _safe_median([abs(float(score) - median) for score in scores])
    robust_scale = _MAD_SCALE * mad
    return max(floor, robust_scale, 1e-9)


def _derive_topk_scores(sample: Mapping[str, object], *, top_k: int) -> list[float]:
    """Extract descending top-k EHA prediction scores from a replay sample."""

    raw_scores = sample.get("eha_topk_pred_bws_json", [])
    scores = [float(value) for value in _as_json_list(raw_scores)]
    if not scores:
        best = _as_float(sample.get("eha_best_pred_bw", 0.0))
        second = _as_float(sample.get("eha_second_pred_bw", 0.0))
        if best > 0:
            scores.append(best)
        if second > 0:
            scores.append(second)
    return sorted(scores, reverse=True)[: max(1, int(top_k))]


def _expected_plan_budget(sample: Mapping[str, object]) -> int:
    """Return the expected number of EHA candidate-plan probes for the sample."""

    phase2_mode = str(sample.get("eha_phase2_mode", "")).strip().lower()
    contention_mode = str(sample.get("contention_mode", "")).strip().lower()
    if phase2_mode == "flat":
        return 8 if contention_mode in {"common", "intensive"} else 5
    if phase2_mode == "hierarchical":
        return 8
    return 1


def compute_confidence_feature_row(
    sample: Mapping[str, object],
    *,
    top_k: int,
    bootstrap_draws: int,
    relative_noise_floor: float,
    bw_improvement_threshold_pct_of_bandpilot: float,
) -> Dict[str, object]:
    """Build the confidence and PTS-helpfulness features for one replay sample."""

    topk_scores = _derive_topk_scores(sample, top_k=top_k)
    sigma_hat = _robust_sigma_hat(topk_scores, relative_noise_floor=relative_noise_floor)

    if len(topk_scores) >= 2:
        z_margin = (float(topk_scores[0]) - float(topk_scores[1])) / sigma_hat
    else:
        z_margin = 0.0

    if len(topk_scores) >= 2:
        rng_seed = (
            17 * _as_int(sample.get("seed_used", 0))
            + 131 * _as_int(sample.get("test_num", 0))
            + 997 * _as_int(sample.get("repeat_idx", 0))
        )
        rng = random.Random(rng_seed)
        stable_count = 0
        for _ in range(max(1, int(bootstrap_draws))):
            perturbed = [
                float(score) + rng.gauss(0.0, sigma_hat)
                for score in topk_scores
            ]
            if int(max(range(len(perturbed)), key=lambda idx: perturbed[idx])) == 0:
                stable_count += 1
        p_stable = stable_count / max(1, int(bootstrap_draws))
    else:
        p_stable = 0.0

    k_values = [int(value) for value in _as_json_list(sample.get("eha_k_values_json", []))]
    phase2_mode = str(sample.get("eha_phase2_mode", "")).strip().lower()
    max_k_options = 2.0 if phase2_mode in {"flat", "hierarchical"} else 1.0
    k_coverage_score = min(1.0, len(k_values) / max_k_options) if max_k_options > 0 else 1.0

    expected_plan_budget = _expected_plan_budget(sample)
    candidate_plan_count = _as_int(sample.get("eha_candidate_plan_count", 0))
    plan_coverage_score = min(1.0, candidate_plan_count / max(1.0, float(expected_plan_budget)))
    coverage_score = 0.5 * k_coverage_score + 0.5 * plan_coverage_score
    budget_binding = bool(
        candidate_plan_count >= expected_plan_budget
        or _as_int(sample.get("eha_num_candidates", 0)) >= _DEFAULT_MAX_EHA_CANDIDATES
    )

    bandpilot_final_bw = _as_float(sample.get("bandpilot_final_bw", 0.0))
    eha_final_bw = _as_float(sample.get("eha_final_bw", 0.0))
    pts_gain_pct_of_bandpilot = (
        100.0 * (bandpilot_final_bw - eha_final_bw) / bandpilot_final_bw
        if bandpilot_final_bw > 0
        else 0.0
    )
    pts_helpful = (
        _as_bool(sample.get("eha_feasible", False))
        and pts_gain_pct_of_bandpilot >= float(bw_improvement_threshold_pct_of_bandpilot)
        and str(sample.get("eha_combo_signature", "")).strip()
        != str(sample.get("bandpilot_combo_signature", "")).strip()
    )

    return {
        "cluster_type": str(sample.get("cluster_type", "")),
        "policy_bucket": str(sample.get("policy_bucket", "")),
        "contention_mode": str(sample.get("contention_mode", "")),
        "test_num": _as_int(sample.get("test_num", 0)),
        "repeat_idx": _as_int(sample.get("repeat_idx", 0)),
        "eha_feasible": _as_bool(sample.get("eha_feasible", False)),
        "eha_node_count": _as_int(sample.get("eha_node_count", 0)),
        "eha_num_candidates": _as_int(sample.get("eha_num_candidates", 0)),
        "eha_phase2_mode": str(sample.get("eha_phase2_mode", "")),
        "eha_hierarchical_path": _as_bool(sample.get("eha_hierarchical_path", False)),
        "eha_candidate_plan_count": candidate_plan_count,
        "eha_kplus1_probe_count": _as_int(sample.get("eha_kplus1_probe_count", 0)),
        "eha_k_values_json": json.dumps(k_values),
        "eha_topk_pred_bws_json": json.dumps(topk_scores),
        "sigma_hat": sigma_hat,
        "z_margin": z_margin,
        "p_stable": float(p_stable),
        "coverage_score": float(coverage_score),
        "budget_binding": bool(budget_binding),
        "expected_plan_budget": int(expected_plan_budget),
        "pts_gain_pct_of_bandpilot": float(pts_gain_pct_of_bandpilot),
        "pts_helpful": bool(pts_helpful),
    }
