"""Benchmark-local online-kNN helper for legacy scalability studies.

The main public path now uses `algorithms.search.improved_searching_algo` with
`RuntimeAdaptiveKNNState`. This helper remains for benchmark-local replay and
diagnostic compatibility.

    Diagnostic usage:
- `cfg = ScalabilityAdaptiveKNNConfig.from_mapping(policy_cfg)`
- `bank = ScalabilityAdaptiveBank(config=cfg, bank_id="Het-4Mix:repeat_0")`
- `decision = bank.replay_case(case_context=..., eha_result=..., hu_pts_result=...)`
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from algorithms.adaptive_knn import AdaptiveKNNConfig, build_adaptive_knn_feature_rows
from algorithms.adaptive_gate_support.online_mismatch_knn.online_bank import (
    OnlineMismatchBank,
)
from algorithms.adaptive_gate_support.online_mismatch_knn.policy import (
    _evaluate_shadow_mismatch,
)


def _combo_signature(combo: Optional[np.ndarray]) -> str:
    """Build a CSV-safe signature for selected GPUs in a 0/1 combo."""

    if combo is None:
        return ""
    combo_arr = np.asarray(combo, dtype=int)
    return ",".join(str(int(idx)) for idx in np.where(combo_arr == 1)[0].tolist())


def _index_signature(indices: Sequence[int]) -> str:
    """Build a CSV-safe signature for a sequence of GPU indices."""

    return ",".join(str(int(value)) for value in list(indices))


@dataclass(frozen=True)
class ScalabilityAdaptiveKNNConfig:
    """Benchmark-local online-kNN policy configuration.

    The helper mirrors `adaptive_knn` feature construction and stores the
    bank/decision knobs needed for legacy scalability diagnostics.
    """

    mode: str = "online_knn"
    top_k: int = 5
    bootstrap_draws: int = 256
    relative_noise_floor: float = 0.02
    bw_improvement_threshold_pct_of_hu_pts: float = 0.5
    k_neighbors: int = 5
    same_contention_only: bool = True
    min_support: int = 5
    low_trust_conflict_risk_threshold: float = 0.30
    risk_threshold: float = 0.15
    output_algorithm_label: str = "BandPilot"
    fallback_backend_label: str = "PTS"
    policy_name: str = "scalability_online_knn"

    @classmethod
    def from_mapping(
        cls,
        config: Optional[Mapping[str, object]],
    ) -> "ScalabilityAdaptiveKNNConfig":
        """Build a benchmark-local online-kNN config from a mapping."""

        raw = dict(config or {})
        return cls(
            mode=str(raw.get("mode", "online_knn")).strip().lower() or "online_knn",
            top_k=max(1, int(raw.get("top_k", 5))),
            bootstrap_draws=max(1, int(raw.get("bootstrap_draws", 256))),
            relative_noise_floor=max(0.0, float(raw.get("relative_noise_floor", 0.02))),
            bw_improvement_threshold_pct_of_hu_pts=float(
                raw.get("bw_improvement_threshold_pct_of_hu_pts", 0.5)
            ),
            k_neighbors=max(1, int(raw.get("k_neighbors", 5))),
            same_contention_only=bool(raw.get("same_contention_only", True)),
            min_support=max(1, int(raw.get("min_support", 5))),
            low_trust_conflict_risk_threshold=float(
                raw.get("low_trust_conflict_risk_threshold", 0.30)
            ),
            risk_threshold=float(raw.get("risk_threshold", 0.15)),
            output_algorithm_label=(
                str(raw.get("output_algorithm_label", "BandPilot")).strip()
                or "BandPilot"
            ),
            fallback_backend_label=(
                str(raw.get("fallback_backend_label", "PTS")).strip()
                or "PTS"
            ),
            policy_name=(
                str(raw.get("policy_name", "scalability_online_knn")).strip()
                or "scalability_online_knn"
            ),
        )

    def build_feature_config(self) -> AdaptiveKNNConfig:
        """Build the benchmark-local adaptive feature configuration."""

        return AdaptiveKNNConfig.from_mapping(
            {
                "top_k": int(self.top_k),
                "bootstrap_draws": int(self.bootstrap_draws),
                "relative_noise_floor": float(self.relative_noise_floor),
                "bw_improvement_threshold_pct_of_bandpilot": float(
                    self.bw_improvement_threshold_pct_of_hu_pts
                ),
                "k_neighbors": int(self.k_neighbors),
                "same_contention_only": bool(self.same_contention_only),
                "min_support": int(self.min_support),
                "low_trust_conflict_risk_threshold": float(
                    self.low_trust_conflict_risk_threshold
                ),
                "risk_threshold": float(self.risk_threshold),
                "output_algorithm_label": str(self.output_algorithm_label),
            }
        )


@dataclass
class ScalabilityAdaptiveBank:
    """Online-kNN bank scoped to one scalability repeat stream."""

    config: ScalabilityAdaptiveKNNConfig
    bank_id: str
    _feature_config: AdaptiveKNNConfig = field(init=False, repr=False)
    _bank: OnlineMismatchBank = field(init=False, repr=False)
    case_index: int = 0

    def __post_init__(self) -> None:
        """Initialize the online bank and reset the case counter."""

        self._feature_config = self.config.build_feature_config()
        self._bank = OnlineMismatchBank.empty(self._feature_config.build_online_bank_config())

    def _build_runtime_sample(
        self,
        *,
        case_context: Mapping[str, Any],
        eha_result: Mapping[str, Any],
        hu_pts_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Convert one benchmark case into an online-kNN replay sample.

        The sample keeps compare-compatible confidence features and maps PTS
        outputs into the `bandpilot_*` compatibility fields expected by the
        shared feature builder.
        """

        eha_meta = dict(eha_result.get("eha_meta", {}) or {})
        avail_gpu = [int(value) for value in case_context.get("avail_gpu", [])]
        background_gpu = [int(value) for value in case_context.get("background_gpu", [])]
        sample = {
            "cluster_type": str(case_context.get("cluster_type", "")),
            "contention_mode": str(case_context.get("contention_mode", "")).strip().lower(),
            "total_gpu": int(case_context.get("total_gpu", 0)),
            "if_dynamic": bool(case_context.get("if_dynamic", True)),
            "search_if_real_data": bool(case_context.get("search_if_real_data", False)),
            "seed_used": int(case_context.get("seed", 0)),
            "test_num": int(case_context.get("k", 0)),
            "repeat_idx": int(case_context.get("repeat_idx", 0)),
            "avail_gpu_count": int(len(avail_gpu)),
            "avail_signature": _index_signature(avail_gpu),
            "background_gpu_count": int(len(background_gpu)),
            "background_signature": _index_signature(background_gpu),
            "occupancy_seed": int(case_context.get("occupancy_seed", 0)),
            "probe_job_id": int(case_context.get("probe_job_id", 0)),
            "eha_feasible": bool(eha_result.get("combo") is not None),
            "eha_node_count": int(eha_meta.get("node_count", 0)),
            "eha_min_node_density": int(eha_meta.get("min_node_density", 0)),
            "eha_num_candidates": int(eha_meta.get("num_candidates", 0)),
            "eha_bw_cv": float(eha_meta.get("bw_cv", 0.0)),
            "eha_top5_gap": float(eha_meta.get("top5_gap", 0.0)),
            "eha_best_pred_bw": float(eha_meta.get("best_pred_bw", 0.0)),
            "eha_second_pred_bw": float(eha_meta.get("second_pred_bw", 0.0)),
            "eha_topk_pred_bws_json": eha_meta.get("topk_pred_bws_json", "[]"),
            "eha_phase2_mode": str(eha_meta.get("phase2_mode", "")),
            "eha_hierarchical_path": bool(eha_meta.get("hierarchical_path", False)),
            "eha_candidate_plan_count": int(eha_meta.get("candidate_plan_count", 0)),
            "eha_estimated_subset_calls": int(eha_meta.get("estimated_subset_calls", 0)),
            "eha_kplus1_probe_count": int(eha_meta.get("kplus1_probe_count", 0)),
            "eha_k_values_json": eha_meta.get("k_values_json", "[]"),
            "eha_search_latency_s": float(eha_result.get("measured_wall_time_s", 0.0)),
            "eha_predict_time_s": float(eha_result.get("predictor_time_s", 0.0)),
            "eha_contention_time_s": float(eha_result.get("contention_time_s", 0.0)),
            "eha_final_bw": float(eha_result.get("final_bw", 0.0)),
            "eha_standalone_bw": float(eha_result.get("standalone_bw", eha_result.get("final_bw", 0.0))),
            "selected_gpu_count_eha": int(case_context.get("k", 0)),
            "eha_combo_signature": str(eha_result.get("combo_signature", "")),
            "pts_search_latency_s": float(hu_pts_result.get("measured_wall_time_s", 0.0)),
            "pts_predict_time_s": float(hu_pts_result.get("predictor_time_s", 0.0)),
            "pts_contention_time_s": float(hu_pts_result.get("contention_time_s", 0.0)),
            "pts_final_bw": float(hu_pts_result.get("final_bw", 0.0)),
            "pts_standalone_bw": float(
                hu_pts_result.get("standalone_bw", hu_pts_result.get("final_bw", 0.0))
            ),
            "selected_gpu_count_pts": int(case_context.get("k", 0)),
            "pts_combo_signature": str(hu_pts_result.get("combo_signature", "")),
            "bandpilot_search_latency_s": float(hu_pts_result.get("measured_wall_time_s", 0.0)),
            "bandpilot_predict_time_s": float(hu_pts_result.get("predictor_time_s", 0.0)),
            "bandpilot_contention_time_s": float(hu_pts_result.get("contention_time_s", 0.0)),
            "bandpilot_final_bw": float(hu_pts_result.get("final_bw", 0.0)),
            "bandpilot_standalone_bw": float(
                hu_pts_result.get("standalone_bw", hu_pts_result.get("final_bw", 0.0))
            ),
            "selected_gpu_count_bandpilot": int(case_context.get("k", 0)),
            "bandpilot_combo_signature": str(hu_pts_result.get("combo_signature", "")),
        }
        return sample

    def replay_case(
        self,
        *,
        case_context: Mapping[str, Any],
        eha_result: Mapping[str, Any],
        hu_pts_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Replay one case through the benchmark-local online-kNN bank."""

        sample = self._build_runtime_sample(
            case_context=case_context,
            eha_result=eha_result,
            hu_pts_result=hu_pts_result,
        )
        feature_row = build_adaptive_knn_feature_rows(
            samples=[sample],
            config=self._feature_config,
        )[0]

        decision_start = time.perf_counter()
        online_risk_row = self._bank.evaluate_query(
            query_sample=sample,
            query_feature_row=feature_row,
        )
        skip_pts, trigger_reason = _evaluate_shadow_mismatch(
            sample=sample,
            online_risk_row=online_risk_row,
            risk_threshold=float(self.config.risk_threshold),
        )
        decision_overhead_ms = 1000.0 * (time.perf_counter() - decision_start)

        selected_backend = "EHA" if skip_pts else str(self.config.fallback_backend_label)
        chosen_result = eha_result if skip_pts else hu_pts_result

        # Append the current case so later cases can query an updated bank.
        self._bank.append_labeled_rows([{**sample, **feature_row}])

        decision = {
            "algorithm": str(self.config.output_algorithm_label),
            "adaptive_policy_name": str(self.config.policy_name),
            "selected_backend": selected_backend,
            "pts_triggered": bool(selected_backend == str(self.config.fallback_backend_label)),
            "trigger_reason": str(trigger_reason),
            "decision_overhead_ms": float(decision_overhead_ms),
            "online_bank_id": str(self.bank_id),
            "online_case_index": int(self.case_index),
            "online_support": int(online_risk_row.get("online_mismatch_support", 0)),
            "online_risk": float(online_risk_row.get("online_mismatch_knn_risk", 1.0)),
            "adaptive_fallback_reason": (
                str(trigger_reason)
                if selected_backend == str(self.config.fallback_backend_label)
                else ""
            ),
            "online_low_trust": bool(online_risk_row.get("online_mismatch_low_trust", False)),
            "support_insufficient": bool(online_risk_row.get("support_insufficient", False)),
            "measured_wall_time_s": float(chosen_result.get("measured_wall_time_s", 0.0))
            + float(decision_overhead_ms) / 1000.0,
            "predictor_time_s": float(chosen_result.get("predictor_time_s", 0.0)),
            "predictor_calls": int(chosen_result.get("predictor_calls", 0)),
            "contention_time_s": float(chosen_result.get("contention_time_s", 0.0)),
            "eha_phase_time_s": float(chosen_result.get("eha_phase_time_s", 0.0)),
            "pts_phase_time_s": float(chosen_result.get("pts_phase_time_s", 0.0)),
            "final_bw": float(chosen_result.get("final_bw", 0.0)),
            "standalone_bw": float(chosen_result.get("standalone_bw", chosen_result.get("final_bw", 0.0))),
            "combo_signature": str(chosen_result.get("combo_signature", "")),
            "bw_cv": float(feature_row.get("eha_bw_cv", 0.0)),
            "top5_gap": float(feature_row.get("eha_top5_gap", 0.0)),
            "num_candidates": float(feature_row.get("eha_num_candidates", 0)),
            "node_count": float(feature_row.get("eha_node_count", 0)),
            "min_node_density": float(sample.get("eha_min_node_density", 0)),
            "hu_pts_usage_rate": 1.0 if selected_backend == str(self.config.fallback_backend_label) else 0.0,
        }
        self.case_index += 1
        return decision
