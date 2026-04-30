"""Runtime adaptive kNN state for the main search path.

`RuntimeAdaptiveKNNState` is passed to
`improved_searching_algo(..., adaptive_pts=True)` when the search path should
decide online whether PTS can be skipped. The state tracks banks,
activation, shadow observations, and labeled dual-observed cases. Callers own
bank boundaries and must call `finish_bank()` explicitly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from algorithms.adaptive_knn import AdaptiveKNNConfig, build_adaptive_knn_feature_rows
from algorithms.adaptive_gate_support.online_mismatch_knn.online_bank import (
    OnlineMismatchBank,
)


def _as_bool(value: object, default: bool = False) -> bool:
    """Coerce a config value to bool with a fallback."""
    if value in ("", None):
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce a config value to float with a fallback."""
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: object, default: int = 0) -> int:
    """Coerce a config value to int with a fallback."""
    if value in ("", None):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean or zero for an empty sequence."""
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    """Return a percentile or zero for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _index_signature(indices: Sequence[int]) -> str:
    """Build a stable comma-separated signature for GPU indices."""
    return ",".join(str(int(value)) for value in list(indices))


def _combo_signature(combo: Optional[np.ndarray]) -> str:
    """Build a stable signature for selected GPUs in a 0/1 combo vector."""
    if combo is None:
        return ""
    combo_arr = np.asarray(combo, dtype=int)
    return ",".join(str(int(idx)) for idx in np.where(combo_arr == 1)[0].tolist())


def _build_admission_rows(
    *,
    bank_version: int,
    bank_phase: str,
    query_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Select which query rows are admitted to the bank before and after activation."""
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


def _evaluate_shadow_mismatch(
    *,
    sample: Mapping[str, object],
    online_risk_row: Mapping[str, object],
    risk_threshold: float,
) -> Tuple[bool, str]:
    """Evaluate the deterministic trigger decision from mismatch-kNN risk."""
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


@dataclass(frozen=True)
class RuntimeActivationCriteria:
    """Bank-level activation criteria for runtime adaptation."""

    unsafe_skip_rate_pct_max: float = 0.0
    over_trigger_rate_pct_max: float = 50.0
    support_insufficient_case_count_max: int = 0
    min_consecutive_pass_banks: int = 2
    sticky_activation: bool = True


@dataclass(frozen=True)
class RuntimeAdaptiveDecision:
    """Runtime-adaptive decision and diagnostics for one dispatch."""

    trigger_pts: bool
    trigger_reason: str
    shadow_trigger_pts: bool
    shadow_trigger_reason: str
    online_risk: float
    support_count: int
    support_insufficient: bool
    online_low_trust: bool
    bank_version: int
    bank_phase: str
    bank_active_before: bool
    train_size_before: int
    case_index: int
    decision_overhead_ms: float


@dataclass(frozen=True)
class RuntimeAdaptiveKNNConfig:
    """Runtime kNN-adaptive policy configuration.

    The runtime path shares feature construction and online-bank semantics with
    the offline `adaptive_knn` replay, but updates the bank incrementally during
    the live compare stream.
    """

    top_k: int = 5
    bootstrap_draws: int = 256
    relative_noise_floor: float = 0.02
    bw_improvement_threshold_pct_of_bandpilot: float = 0.5
    k_neighbors: int = 5
    same_contention_only: bool = True
    min_support: int = 5
    low_trust_conflict_risk_threshold: float = 0.30
    risk_threshold: float = 0.15
    activation_criteria: RuntimeActivationCriteria = RuntimeActivationCriteria()
    policy_name: str = "adaptive_knn"

    @classmethod
    def from_mapping(
        cls,
        config: Optional[Mapping[str, object]],
    ) -> "RuntimeAdaptiveKNNConfig":
        """Build runtime configuration from an `adaptive_runtime_policy` mapping."""
        raw = dict(config or {})
        activation_cfg = dict(raw.get("activation_criteria", {}))
        return cls(
            top_k=max(1, int(raw.get("top_k", 5))),
            bootstrap_draws=max(1, int(raw.get("bootstrap_draws", 256))),
            relative_noise_floor=max(0.0, float(raw.get("relative_noise_floor", 0.02))),
            bw_improvement_threshold_pct_of_bandpilot=float(
                raw.get("bw_improvement_threshold_pct_of_bandpilot", 0.5)
            ),
            k_neighbors=max(1, int(raw.get("k_neighbors", 5))),
            same_contention_only=bool(raw.get("same_contention_only", True)),
            min_support=max(1, int(raw.get("min_support", 5))),
            low_trust_conflict_risk_threshold=float(
                raw.get("low_trust_conflict_risk_threshold", 0.30)
            ),
            risk_threshold=float(raw.get("risk_threshold", 0.15)),
            activation_criteria=RuntimeActivationCriteria(
                unsafe_skip_rate_pct_max=float(
                    activation_cfg.get("shadow_unsafe_skip_rate_pct_max", 0.0)
                ),
                over_trigger_rate_pct_max=float(
                    activation_cfg.get("shadow_over_trigger_rate_pct_max", 50.0)
                ),
                support_insufficient_case_count_max=int(
                    activation_cfg.get("shadow_support_insufficient_case_count_max", 0)
                ),
                min_consecutive_pass_banks=max(
                    1,
                    int(activation_cfg.get("min_consecutive_pass_banks", 2)),
                ),
                sticky_activation=bool(activation_cfg.get("sticky_activation", True)),
            ),
            policy_name=str(raw.get("policy_name", "adaptive_knn")).strip() or "adaptive_knn",
        )

    def build_feature_config(self) -> AdaptiveKNNConfig:
        """Create the feature-builder config used by runtime confidence rows."""
        return AdaptiveKNNConfig.from_mapping(
            {
                "top_k": int(self.top_k),
                "bootstrap_draws": int(self.bootstrap_draws),
                "relative_noise_floor": float(self.relative_noise_floor),
                "bw_improvement_threshold_pct_of_bandpilot": float(
                    self.bw_improvement_threshold_pct_of_bandpilot
                ),
                "k_neighbors": int(self.k_neighbors),
                "same_contention_only": bool(self.same_contention_only),
                "min_support": int(self.min_support),
                "low_trust_conflict_risk_threshold": float(
                    self.low_trust_conflict_risk_threshold
                ),
                "risk_threshold": float(self.risk_threshold),
                "output_algorithm_label": "RuntimeAdaptive",
                "activation_criteria": {
                    "shadow_unsafe_skip_rate_pct_max": float(
                        self.activation_criteria.unsafe_skip_rate_pct_max
                    ),
                    "shadow_over_trigger_rate_pct_max": float(
                        self.activation_criteria.over_trigger_rate_pct_max
                    ),
                    "shadow_support_insufficient_case_count_max": int(
                        self.activation_criteria.support_insufficient_case_count_max
                    ),
                    "min_consecutive_pass_banks": int(
                        self.activation_criteria.min_consecutive_pass_banks
                    ),
                    "sticky_activation": bool(self.activation_criteria.sticky_activation),
                },
            }
        )


@dataclass
class RuntimeAdaptiveKNNState:
    """Mutable runtime-adaptive state for one dispatch stream."""

    config: RuntimeAdaptiveKNNConfig
    bank_id: str
    _feature_config: AdaptiveKNNConfig = field(init=False, repr=False)
    _bank: OnlineMismatchBank = field(init=False, repr=False)
    bank_is_active: bool = False
    activation_pass_streak: int = 0
    bank_version: int = 0
    global_case_index: int = 0
    _current_bank_labeled_rows: List[Dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _current_bank_decision_supports: List[float] = field(default_factory=list, init=False, repr=False)
    _current_bank_unlabeled_case_count: int = 0

    def __post_init__(self) -> None:
        """Initialize the feature config and empty online bank."""
        self._feature_config = self.config.build_feature_config()
        self._bank = OnlineMismatchBank.empty(self._feature_config.build_online_bank_config())

    @classmethod
    def from_mapping(
        cls,
        config: Optional[Mapping[str, object]],
        *,
        bank_id: str,
    ) -> "RuntimeAdaptiveKNNState":
        """Build runtime state from a policy mapping and bank id."""
        return cls(
            config=RuntimeAdaptiveKNNConfig.from_mapping(config),
            bank_id=str(bank_id),
        )

    def _bank_phase(self) -> str:
        """Return the formal bank phase used for current decisions."""
        return "post_activation" if self.bank_is_active else "pre_activation"

    def _normalize_case_context(
        self,
        *,
        total_gpu: int,
        gpu_need: int,
        avail_gpu: Sequence[int],
        if_real_data: bool,
        cluster_type: str,
        contention_mode: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, object]:
        """Normalize dispatch inputs into the sample fields used by the policy."""
        raw = dict(context or {})
        avail_indices = [int(value) for value in list(avail_gpu)]
        if "background_gpu" in raw:
            background_gpu = [int(value) for value in list(raw.get("background_gpu", []))]
        else:
            avail_set = set(avail_indices)
            background_gpu = [gpu_idx for gpu_idx in range(int(total_gpu)) if gpu_idx not in avail_set]
        case_index = int(self.global_case_index)
        return {
            "cluster_type": str(raw.get("cluster_type", cluster_type)),
            "policy_bucket": str(raw.get("policy_bucket", "runtime")),
            "policy_mode": str(raw.get("policy_mode", self.config.policy_name)),
            "contention_mode": str(raw.get("contention_mode", contention_mode)).strip().lower(),
            "total_gpu": int(raw.get("total_gpu", total_gpu)),
            "if_dynamic": bool(raw.get("if_dynamic", len(avail_indices) < int(total_gpu))),
            "search_if_real_data": bool(raw.get("search_if_real_data", if_real_data)),
            "seed_used": int(raw.get("seed_used", 0)),
            "test_num": int(raw.get("test_num", gpu_need)),
            "repeat_idx": int(raw.get("repeat_idx", case_index)),
            "avail_gpu_count": int(len(avail_indices)),
            "avail_signature": str(raw.get("avail_signature", _index_signature(avail_indices))),
            "background_gpu_count": int(len(background_gpu)),
            "background_signature": str(
                raw.get("background_signature", _index_signature(background_gpu))
            ),
            "occupancy_seed": int(raw.get("occupancy_seed", case_index)),
            "probe_job_id": int(raw.get("probe_job_id", case_index)),
        }

    def build_eha_decision_input(
        self,
        *,
        total_gpu: int,
        gpu_need: int,
        avail_gpu: Sequence[int],
        if_real_data: bool,
        cluster_type: str,
        contention_mode: str,
        eha_combo: Optional[np.ndarray],
        eha_meta: Mapping[str, object],
        eha_search_latency_s: float,
        eha_final_bw: float,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        """Build the EHA-derived runtime decision sample and feature row."""
        sample = self._normalize_case_context(
            total_gpu=total_gpu,
            gpu_need=gpu_need,
            avail_gpu=avail_gpu,
            if_real_data=if_real_data,
            cluster_type=cluster_type,
            contention_mode=contention_mode,
            context=context,
        )
        sample.update(
            {
                "eha_feasible": bool(eha_combo is not None),
                "eha_node_count": int(eha_meta.get("node_count", 0)),
                "eha_min_node_density": int(eha_meta.get("min_node_density", 0)),
                "eha_num_candidates": int(eha_meta.get("num_candidates", 0)),
                "eha_bw_cv": float(eha_meta.get("bw_cv", 0.0)),
                "eha_top5_gap": float(eha_meta.get("top5_gap", 0.0)),
                "eha_best_pred_bw": float(eha_meta.get("best_bw", 0.0)),
                "eha_second_pred_bw": float(
                    (list(eha_meta.get("bw_list", [])) + [0.0, 0.0])[1]
                ),
                "eha_topk_pred_bws_json": json.dumps(
                    [float(value) for value in list(eha_meta.get("bw_list", []))[:5]]
                ),
                "eha_phase2_mode": str(eha_meta.get("phase2_mode", "")),
                "eha_hierarchical_path": bool(eha_meta.get("hierarchical_path", False)),
                "eha_candidate_plan_count": int(eha_meta.get("candidate_plan_count", 0)),
                "eha_estimated_subset_calls": int(eha_meta.get("estimated_subset_calls", 0)),
                "eha_kplus1_probe_count": int(eha_meta.get("kplus1_probe_count", 0)),
                "eha_k_values_json": json.dumps(
                    [int(value) for value in list(eha_meta.get("k_values", []))]
                ),
                "eha_search_latency_s": float(eha_search_latency_s),
                "eha_predict_time_s": 0.0,
                "eha_contention_time_s": 0.0,
                "eha_final_bw": float(eha_final_bw),
                "eha_standalone_bw": float(eha_final_bw),
                "selected_gpu_count_eha": int(gpu_need),
                "eha_combo_signature": _combo_signature(eha_combo),
                # Runtime has not observed the BandPilot result yet, so seed
                # these fields from EHA to keep feature construction total.
                "bandpilot_search_latency_s": 0.0,
                "bandpilot_predict_time_s": 0.0,
                "bandpilot_contention_time_s": 0.0,
                "bandpilot_final_bw": float(eha_final_bw),
                "bandpilot_standalone_bw": float(eha_final_bw),
                "selected_gpu_count_bandpilot": int(gpu_need),
                "bandpilot_combo_signature": _combo_signature(eha_combo),
            }
        )
        feature_row = build_adaptive_knn_feature_rows(
            samples=[sample],
            config=self._feature_config,
        )[0]
        return sample, feature_row

    def decide_case(
        self,
        *,
        sample: Mapping[str, object],
        feature_row: Mapping[str, object],
    ) -> RuntimeAdaptiveDecision:
        """Decide whether the current case should trigger PTS."""
        import time

        decision_start = time.perf_counter()
        online_risk_row = self._bank.evaluate_query(
            query_sample=sample,
            query_feature_row=feature_row,
        )
        shadow_skip, shadow_reason = _evaluate_shadow_mismatch(
            sample=sample,
            online_risk_row=online_risk_row,
            risk_threshold=float(self.config.risk_threshold),
        )
        decision_overhead_ms = 1000.0 * (time.perf_counter() - decision_start)
        self._current_bank_decision_supports.append(
            float(online_risk_row.get("online_mismatch_support", 0.0))
        )

        if self.bank_is_active:
            trigger_pts = not shadow_skip
            trigger_reason = shadow_reason
        else:
            trigger_pts = True
            trigger_reason = "pre_activation_bandpilot_fallback"

        decision = RuntimeAdaptiveDecision(
            trigger_pts=bool(trigger_pts),
            trigger_reason=str(trigger_reason),
            shadow_trigger_pts=bool(not shadow_skip),
            shadow_trigger_reason=str(shadow_reason),
            online_risk=float(online_risk_row.get("online_mismatch_knn_risk", 1.0)),
            support_count=int(online_risk_row.get("online_mismatch_support", 0)),
            support_insufficient=bool(
                online_risk_row.get("support_insufficient", False)
            ),
            online_low_trust=bool(
                online_risk_row.get("online_mismatch_low_trust", False)
            ),
            bank_version=int(self.bank_version),
            bank_phase=str(self._bank_phase()),
            bank_active_before=bool(self.bank_is_active),
            train_size_before=int(self._bank.labeled_size),
            case_index=int(self.global_case_index),
            decision_overhead_ms=float(decision_overhead_ms),
        )
        self.global_case_index += 1
        return decision

    def observe_labeled_case(
        self,
        *,
        sample: Mapping[str, object],
        feature_row: Mapping[str, object],
        decision: RuntimeAdaptiveDecision,
        bandpilot_combo: Optional[np.ndarray],
        bandpilot_final_bw: float,
        bandpilot_search_latency_s: float,
    ) -> Dict[str, object]:
        """Record a dual-observed case for later bank admission and diagnostics."""
        labeled_sample = dict(sample)
        labeled_sample.update(
            {
                "bandpilot_search_latency_s": float(bandpilot_search_latency_s),
                "bandpilot_predict_time_s": 0.0,
                "bandpilot_contention_time_s": 0.0,
                "bandpilot_final_bw": float(bandpilot_final_bw),
                "bandpilot_standalone_bw": float(bandpilot_final_bw),
                "selected_gpu_count_bandpilot": int(sample.get("test_num", 0)),
                "bandpilot_combo_signature": _combo_signature(bandpilot_combo),
            }
        )
        labeled_feature_row = build_adaptive_knn_feature_rows(
            samples=[labeled_sample],
            config=self._feature_config,
        )[0]
        query_row = {
            **labeled_sample,
            **labeled_feature_row,
            "bank_version": int(decision.bank_version),
            "bank_phase": str(decision.bank_phase),
            "train_size_before": int(decision.train_size_before),
            "global_case_index": int(decision.case_index),
            "online_mismatch_knn_risk": float(decision.online_risk),
            "online_mismatch_support": int(decision.support_count),
            "online_mismatch_scope": str(sample.get("cluster_type", "")),
            "online_mismatch_helpful_neighbor_count": 0,
            "online_mismatch_non_helpful_neighbor_count": 0,
            "online_mismatch_conflict_risk": 0.0,
            "online_mismatch_low_trust": bool(decision.online_low_trust),
            "online_mismatch_min_support": int(self.config.min_support),
            "support_insufficient": bool(decision.support_insufficient),
            "decision_overhead_ms": float(decision.decision_overhead_ms),
            "shadow_skip_pts": bool(not decision.shadow_trigger_pts),
            "shadow_trigger_reason": str(decision.shadow_trigger_reason),
            "formal_skip_pts": bool(not decision.trigger_pts),
            "formal_trigger_reason": str(decision.trigger_reason),
        }
        self._current_bank_labeled_rows.append(query_row)
        return query_row

    def record_unlabeled_skip_case(self) -> None:
        """Track a skipped case that cannot be used as a labeled bank row."""
        self._current_bank_unlabeled_case_count += 1

    def finish_bank(self) -> Dict[str, object]:
        """Finalize the current bank, update activation state, and reset buffers."""
        bank_phase = self._bank_phase()
        train_size_before = int(self._bank.labeled_size)

        if self._current_bank_labeled_rows:
            shadow_eval_bank = OnlineMismatchBank.from_rows(
                config=self._bank.config,
                labeled_rows=[*self._bank.labeled_samples, *self._current_bank_labeled_rows],
            )
            shadow_supports: List[float] = []
            shadow_rows: List[Dict[str, object]] = []
            for query_row in self._current_bank_labeled_rows:
                shadow_eval_risk_row = shadow_eval_bank.evaluate_query(
                    query_sample=query_row,
                    query_feature_row=query_row,
                )
                shadow_supports.append(
                    _as_float(shadow_eval_risk_row.get("online_mismatch_support", 0))
                )
                shadow_skip, shadow_reason = _evaluate_shadow_mismatch(
                    sample=query_row,
                    online_risk_row=shadow_eval_risk_row,
                    risk_threshold=float(self.config.risk_threshold),
                )
                shadow_rows.append(
                    {
                        **dict(query_row),
                        "shadow_skip_pts": bool(shadow_skip),
                        "shadow_trigger_reason": str(shadow_reason),
                        "support_insufficient": bool(
                            shadow_eval_risk_row.get("support_insufficient", False)
                        ),
                        "online_mismatch_low_trust": bool(
                            shadow_eval_risk_row.get("online_mismatch_low_trust", False)
                        ),
                    }
                )
            unsafe_skip_count = int(
                sum(
                    1
                    for row in shadow_rows
                    if _as_bool(row.get("shadow_skip_pts", False))
                    and _as_bool(row.get("pts_helpful", False))
                )
            )
            over_trigger_count = int(
                sum(
                    1
                    for row in shadow_rows
                    if (not _as_bool(row.get("shadow_skip_pts", False)))
                    and (not _as_bool(row.get("pts_helpful", False)))
                )
            )
            shadow_unsafe_skip_rate_pct = (
                100.0 * unsafe_skip_count / float(len(shadow_rows)) if shadow_rows else 100.0
            )
            shadow_over_trigger_rate_pct = (
                100.0 * over_trigger_count / float(len(shadow_rows)) if shadow_rows else 100.0
            )
            shadow_support_insufficient_case_count = int(
                sum(1 for row in shadow_rows if _as_bool(row.get("support_insufficient", False)))
            )
            shadow_low_trust_case_count = int(
                sum(
                    1 for row in shadow_rows if _as_bool(row.get("online_mismatch_low_trust", False))
                )
            )
        else:
            shadow_supports = []
            shadow_unsafe_skip_rate_pct = 100.0
            shadow_over_trigger_rate_pct = 100.0
            shadow_support_insufficient_case_count = 0
            shadow_low_trust_case_count = 0

        criteria = self.config.activation_criteria
        bank_meets_activation_gate = (
            bool(self._current_bank_labeled_rows)
            and shadow_unsafe_skip_rate_pct <= float(criteria.unsafe_skip_rate_pct_max)
            and shadow_over_trigger_rate_pct < float(criteria.over_trigger_rate_pct_max)
            and shadow_support_insufficient_case_count
            <= int(criteria.support_insufficient_case_count_max)
        )
        self.activation_pass_streak = (
            self.activation_pass_streak + 1 if bank_meets_activation_gate else 0
        )
        activate_next_bank = (
            (not self.bank_is_active or not criteria.sticky_activation)
            and self.activation_pass_streak >= int(criteria.min_consecutive_pass_banks)
        )
        bank_active_next = bool(self.bank_is_active or activate_next_bank)

        admitted_rows = _build_admission_rows(
            bank_version=int(self.bank_version),
            bank_phase=str(bank_phase),
            query_rows=self._current_bank_labeled_rows,
        )
        if admitted_rows:
            self._bank.append_labeled_rows(admitted_rows)

        summary = {
            "bank_id": str(self.bank_id),
            "bank_version": int(self.bank_version),
            "bank_phase": str(bank_phase),
            "bank_active_before": bool(self.bank_is_active),
            "bank_active_next": bool(bank_active_next),
            "activate_next_bank": bool(activate_next_bank),
            "train_size_before": int(train_size_before),
            "train_size_after": int(self._bank.labeled_size),
            "bank_size_labeled": int(len(self._current_bank_labeled_rows)),
            "bank_size_unlabeled_skips": int(self._current_bank_unlabeled_case_count),
            "admitted_count": int(len(admitted_rows)),
            "shadow_unsafe_skip_rate_pct": float(shadow_unsafe_skip_rate_pct),
            "shadow_over_trigger_rate_pct": float(shadow_over_trigger_rate_pct),
            "shadow_support_insufficient_case_count": int(
                shadow_support_insufficient_case_count
            ),
            "shadow_low_trust_case_count": int(shadow_low_trust_case_count),
            "activation_pass_streak": int(self.activation_pass_streak),
            "mean_decision_support": float(_mean(self._current_bank_decision_supports)),
            "p95_decision_support": float(_percentile(self._current_bank_decision_supports, 0.95)),
            "shadow_mean_support": float(_mean(shadow_supports)),
            "shadow_p95_support": float(_percentile(shadow_supports, 0.95)),
        }

        self.bank_is_active = bool(bank_active_next)
        self.bank_version += 1
        self._current_bank_labeled_rows.clear()
        self._current_bank_decision_supports.clear()
        self._current_bank_unlabeled_case_count = 0
        return summary
