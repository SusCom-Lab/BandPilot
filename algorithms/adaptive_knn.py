"""Compare-time adaptive kNN replay for BandPilot and legacy diagnostics.

This module converts compare-style runtime samples into banked online-kNN
decisions. It is used by `evaluation.compare` to emit compare-compatible
rows without changing the main BandPilot search implementation.

Typical flow:
1. Build confidence features from EHA, PTS, and legacy-BandPilot observations.
2. Replay banked online decisions over the repeat-major sample stream.
3. Convert policy decisions back into rows consumed by evaluation reports.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from algorithms.adaptive_gate_support.confidence import (
    compute_confidence_feature_row,
)
from algorithms.adaptive_gate_support.utils import (
    _as_bool,
    _as_float,
    _as_int,
)
from algorithms.adaptive_gate_support.online_mismatch_knn.online_bank import (
    OnlineBankConfig,
    OnlineMismatchBank,
)
from algorithms.adaptive_gate_support.online_mismatch_knn.policy import (
    ActivationCriteria,
    _build_admission_rows,
    _evaluate_shadow_mismatch,
)
from algorithms.adaptive_gate_support.shared_metrics import (
    derive_case_outcome_fields,
    summarize_case_rows,
)


_MODE_ORDER = {"idle": 0, "common": 1, "intensive": 2}


@dataclass(frozen=True)
class AdaptiveKNNConfig:
    """Configuration for compare-time adaptive kNN replay."""

    top_k: int = 5
    bootstrap_draws: int = 256
    relative_noise_floor: float = 0.02
    bw_improvement_threshold_pct_of_bandpilot: float = 0.5
    bank_partition_mode: str = "repeat"
    fixed_bank_size: Optional[int] = None
    k_neighbors: int = 5
    same_contention_only: bool = True
    min_support: Optional[int] = 5
    low_trust_conflict_risk_threshold: float = 0.30
    risk_threshold: float = 0.15
    activation_criteria: ActivationCriteria = ActivationCriteria(
        unsafe_skip_rate_pct_max=0.0,
        over_trigger_rate_pct_max=50.0,
        support_insufficient_case_count_max=0,
        min_consecutive_pass_banks=2,
        sticky_activation=True,
    )
    algorithm_label: str = "BandPilot"
    include_legacy_adaptive_baseline: bool = False
    legacy_algorithm_label: str = "legacy-BandPilot-KNN"

    @classmethod
    def from_mapping(
        cls,
        config: Optional[Mapping[str, object]],
    ) -> "AdaptiveKNNConfig":
        """Build a config from `single_contention.adaptive_runtime_policy`."""

        raw = dict(config or {})
        activation_cfg = dict(raw.get("activation_criteria", {}))
        return cls(
            top_k=max(1, int(raw.get("top_k", 5))),
            bootstrap_draws=max(1, int(raw.get("bootstrap_draws", 256))),
            relative_noise_floor=max(0.0, float(raw.get("relative_noise_floor", 0.02))),
            bw_improvement_threshold_pct_of_bandpilot=float(
                raw.get("bw_improvement_threshold_pct_of_bandpilot", 0.5)
            ),
            bank_partition_mode=str(raw.get("bank_partition_mode", "repeat")).strip().lower(),
            fixed_bank_size=(
                None
                if raw.get("fixed_bank_size") in ("", None)
                else max(1, int(raw.get("fixed_bank_size", 1)))
            ),
            k_neighbors=max(1, int(raw.get("k_neighbors", 5))),
            same_contention_only=bool(raw.get("same_contention_only", True)),
            min_support=(
                None
                if raw.get("min_support") in ("", None)
                else max(1, int(raw.get("min_support", 1)))
            ),
            low_trust_conflict_risk_threshold=float(
                raw.get("low_trust_conflict_risk_threshold", 0.30)
            ),
            risk_threshold=float(raw.get("risk_threshold", 0.15)),
            activation_criteria=ActivationCriteria(
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
            algorithm_label=(
                str(raw.get("output_algorithm_label", "BandPilot")).strip()
                or "BandPilot"
            ),
            include_legacy_adaptive_baseline=bool(
                raw.get("include_legacy_adaptive_baseline", False)
            ),
            legacy_algorithm_label=(
                str(raw.get("legacy_algorithm_label", "legacy-BandPilot-KNN")).strip()
                or "legacy-BandPilot-KNN"
            ),
        )

    def build_online_bank_config(self) -> OnlineBankConfig:
        """Project compare settings onto the runtime online-bank config."""

        return OnlineBankConfig(
            k_neighbors=int(self.k_neighbors),
            same_contention_only=bool(self.same_contention_only),
            min_support=self.min_support,
            low_trust_conflict_risk_threshold=float(
                self.low_trust_conflict_risk_threshold
            ),
        )


def build_adaptive_knn_feature_rows(
    *,
    samples: Sequence[Mapping[str, object]],
    config: AdaptiveKNNConfig,
) -> List[Dict[str, object]]:
    """Build confidence-feature rows for adaptive kNN replay."""

    return [
        compute_confidence_feature_row(
            sample,
            top_k=int(config.top_k),
            bootstrap_draws=int(config.bootstrap_draws),
            relative_noise_floor=float(config.relative_noise_floor),
            bw_improvement_threshold_pct_of_bandpilot=float(
                config.bw_improvement_threshold_pct_of_bandpilot
            ),
        )
        for sample in samples
    ]


def _ordered_runtime_pairs(
    *,
    samples: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, object]],
) -> List[Tuple[Mapping[str, object], Mapping[str, object]]]:
    """Order runtime samples by repeat, contention mode, and test size."""

    paired = list(zip(samples, feature_rows))
    return sorted(
        paired,
        key=lambda pair: (
            _as_int(pair[0].get("repeat_idx", 0)),
            _MODE_ORDER.get(str(pair[0].get("contention_mode", "")).strip().lower(), 99),
            _as_int(pair[0].get("test_num", 0)),
        ),
    )


def _build_bank_key(
    *,
    sample: Mapping[str, object],
    partition_mode: str,
    fixed_bank_size: Optional[int],
    sorted_index: int,
) -> Tuple[object, ...]:
    """Build the grouping key for the configured online bank partition."""

    if partition_mode == "repeat":
        return (_as_int(sample.get("repeat_idx", 0)),)
    if partition_mode == "repeat_mode":
        return (
            _as_int(sample.get("repeat_idx", 0)),
            _MODE_ORDER.get(str(sample.get("contention_mode", "")).strip().lower(), 99),
        )
    if partition_mode == "fixed_case_count":
        if fixed_bank_size is None or fixed_bank_size <= 0:
            raise ValueError("fixed_case_count requires a positive fixed_bank_size.")
        return (sorted_index // int(fixed_bank_size),)
    raise ValueError(f"Unsupported adaptive_knn bank partition mode: {partition_mode}")


def _build_banks(
    *,
    samples: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, object]],
    config: AdaptiveKNNConfig,
) -> List[List[Tuple[Mapping[str, object], Mapping[str, object]]]]:
    """Group the compare sample stream into online replay banks."""

    ordered_pairs = _ordered_runtime_pairs(samples=samples, feature_rows=feature_rows)
    grouped: Dict[Tuple[object, ...], List[Tuple[Mapping[str, object], Mapping[str, object]]]] = {}
    ordered_keys: List[Tuple[object, ...]] = []
    for sorted_index, (sample, feature_row) in enumerate(ordered_pairs):
        key = _build_bank_key(
            sample=sample,
            partition_mode=str(config.bank_partition_mode),
            fixed_bank_size=config.fixed_bank_size,
            sorted_index=sorted_index,
        )
        if key not in grouped:
            ordered_keys.append(key)
            grouped[key] = []
        grouped[key].append((sample, feature_row))
    return [grouped[key] for key in ordered_keys]


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean or zero for an empty sequence."""

    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    """Return the linear-interpolated percentile for a numeric sequence."""

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


def _build_policy_row(
    *,
    sample: Mapping[str, object],
    feature_row: Mapping[str, object],
    bank_version: int,
    bank_phase: str,
    train_size_before: int,
    global_case_index: int,
    online_risk_row: Mapping[str, object],
    selected_source: str,
    trigger_reason: str,
    decision_overhead_ms: float,
    algorithm_label: str,
) -> Dict[str, object]:
    """Build one `adaptive_knn` policy row.

    The row merges compare-compatible result fields with online decision
    diagnostics so replay outputs can be written directly into
    `Single_contention_*.csv`-style artifacts.
    """

    source_prefix = str(selected_source).strip().lower()
    if source_prefix not in {"eha", "bandpilot", "pts"}:
        raise ValueError(f"Unsupported selected_source: {selected_source}")

    chosen_bw = _as_float(sample.get(f"{source_prefix}_final_bw", 0.0))
    chosen_standalone_bw = _as_float(sample.get(f"{source_prefix}_standalone_bw", 0.0))
    chosen_latency_s = _as_float(sample.get(f"{source_prefix}_search_latency_s", 0.0))
    chosen_predict_time_s = _as_float(sample.get(f"{source_prefix}_predict_time_s", 0.0))
    chosen_contention_time_s = _as_float(
        sample.get(f"{source_prefix}_contention_time_s", 0.0)
    )
    chosen_selected_gpu_count = _as_int(
        sample.get(f"selected_gpu_count_{source_prefix}", 0)
    )
    bandpilot_bw = _as_float(sample.get("bandpilot_final_bw", 0.0))

    row = {
        "algorithm_label": algorithm_label,
        "adaptive_policy_name": "adaptive_knn",
        "cluster_type": str(sample.get("cluster_type", "")),
        "policy_bucket": str(sample.get("policy_bucket", "")),
        "policy_mode": str(sample.get("policy_mode", "")),
        "contention_mode": str(sample.get("contention_mode", "")),
        "total_gpu": _as_int(sample.get("total_gpu", 0)),
        "if_dynamic": _as_bool(sample.get("if_dynamic", False)),
        "search_if_real_data": _as_bool(sample.get("search_if_real_data", False)),
        "seed_used": sample.get("seed_used"),
        "test_num": _as_int(sample.get("test_num", 0)),
        "repeat_idx": _as_int(sample.get("repeat_idx", 0)),
        "avail_gpu_count": _as_int(sample.get("avail_gpu_count", 0)),
        "avail_signature": str(sample.get("avail_signature", "")),
        "background_gpu_count": _as_int(sample.get("background_gpu_count", 0)),
        "background_signature": str(sample.get("background_signature", "")),
        "occupancy_seed": _as_int(sample.get("occupancy_seed", 0)),
        "probe_job_id": _as_int(sample.get("probe_job_id", 0)),
        "max_bw": _as_float(sample.get("max_bw", 0.0)),
        "bw_type": str(sample.get("bw_type", "")),
        "bank_version": bank_version,
        "bank_phase": bank_phase,
        "train_size_before": train_size_before,
        "global_case_index": global_case_index,
        "chosen_source": source_prefix,
        "trigger_reason": trigger_reason,
        "skip_pts": source_prefix == "eha",
        "pts_triggered": source_prefix != "eha",
        "pts_helpful": _as_bool(feature_row.get("pts_helpful", False)),
        "eha_feasible": _as_bool(feature_row.get("eha_feasible", False)),
        "chosen_elapsed_time_s": chosen_latency_s + float(decision_overhead_ms) / 1000.0,
        "chosen_predict_time_s": chosen_predict_time_s,
        "chosen_contention_time_s": chosen_contention_time_s,
        "chosen_final_bw": chosen_bw,
        "chosen_standalone_bw": chosen_standalone_bw,
        "bandpilot_elapsed_time_s": _as_float(sample.get("bandpilot_search_latency_s", 0.0)),
        "eha_elapsed_time_s": _as_float(sample.get("eha_search_latency_s", 0.0)),
        "bandpilot_final_bw": bandpilot_bw,
        "chosen_final_bw_pct_of_bandpilot": (
            100.0 * chosen_bw / bandpilot_bw if bandpilot_bw > 0 else 0.0
        ),
        "chosen_combo_signature": str(sample.get(f"{source_prefix}_combo_signature", "")),
        "bandpilot_combo_signature": str(sample.get("bandpilot_combo_signature", "")),
        "eha_combo_signature": str(sample.get("eha_combo_signature", "")),
        "selected_gpu_count": chosen_selected_gpu_count,
        "pts_gain_pct_of_bandpilot": _as_float(
            feature_row.get("pts_gain_pct_of_bandpilot", 0.0)
        ),
        "z_margin": _as_float(feature_row.get("z_margin", 0.0)),
        "p_stable": _as_float(feature_row.get("p_stable", 0.0)),
        "coverage_score": _as_float(feature_row.get("coverage_score", 0.0)),
        "budget_binding": _as_bool(feature_row.get("budget_binding", False)),
        "decision_overhead_ms": float(decision_overhead_ms),
        "online_mismatch_knn_risk": _as_float(
            online_risk_row.get("online_mismatch_knn_risk", 0.0)
        ),
        "online_mismatch_support": _as_int(
            online_risk_row.get("online_mismatch_support", 0)
        ),
        "online_mismatch_scope": str(online_risk_row.get("online_mismatch_scope", "")),
        "online_mismatch_helpful_neighbor_count": _as_int(
            online_risk_row.get("online_mismatch_helpful_neighbor_count", 0)
        ),
        "online_mismatch_non_helpful_neighbor_count": _as_int(
            online_risk_row.get("online_mismatch_non_helpful_neighbor_count", 0)
        ),
        "online_mismatch_conflict_risk": _as_float(
            online_risk_row.get("online_mismatch_conflict_risk", 0.0)
        ),
        "online_mismatch_low_trust": _as_bool(
            online_risk_row.get("online_mismatch_low_trust", False)
        ),
        "online_mismatch_min_support": _as_int(
            online_risk_row.get("online_mismatch_min_support", 0)
        ),
        "support_insufficient": _as_bool(
            online_risk_row.get("support_insufficient", False)
        ),
    }
    row.update(derive_case_outcome_fields(row))
    return row


def _build_query_row(
    *,
    sample: Mapping[str, object],
    feature_row: Mapping[str, object],
    bank_version: int,
    bank_phase: str,
    train_size_before: int,
    global_case_index: int,
    online_risk_row: Mapping[str, object],
    shadow_skip: bool,
    shadow_reason: str,
    formal_skip: bool,
    formal_reason: str,
    decision_overhead_ms: float,
) -> Dict[str, object]:
    """Build the diagnostic query row for one replay-bank decision.

    Query rows preserve the online risk estimate, formal decision, and shadow
    decision. They are reused for admission accounting, shadow-bank evaluation,
    and blind-pocket diagnostics.
    """

    return {
        **dict(sample),
        **dict(feature_row),
        "bank_version": bank_version,
        "bank_phase": bank_phase,
        "train_size_before": train_size_before,
        "global_case_index": global_case_index,
        "online_mismatch_knn_risk": _as_float(
            online_risk_row.get("online_mismatch_knn_risk", 0.0)
        ),
        "online_mismatch_support": _as_int(
            online_risk_row.get("online_mismatch_support", 0)
        ),
        "online_mismatch_scope": str(online_risk_row.get("online_mismatch_scope", "")),
        "online_mismatch_helpful_neighbor_count": _as_int(
            online_risk_row.get("online_mismatch_helpful_neighbor_count", 0)
        ),
        "online_mismatch_non_helpful_neighbor_count": _as_int(
            online_risk_row.get("online_mismatch_non_helpful_neighbor_count", 0)
        ),
        "online_mismatch_conflict_risk": _as_float(
            online_risk_row.get("online_mismatch_conflict_risk", 0.0)
        ),
        "online_mismatch_low_trust": _as_bool(
            online_risk_row.get("online_mismatch_low_trust", False)
        ),
        "online_mismatch_min_support": _as_int(
            online_risk_row.get("online_mismatch_min_support", 0)
        ),
        "support_insufficient": _as_bool(
            online_risk_row.get("support_insufficient", False)
        ),
        "decision_overhead_ms": float(decision_overhead_ms),
        "shadow_skip_pts": bool(shadow_skip),
        "shadow_trigger_reason": shadow_reason,
        "formal_skip_pts": bool(formal_skip),
        "formal_trigger_reason": formal_reason,
    }


def _build_activation_summary(
    *,
    activation_timeline_rows: Sequence[Mapping[str, object]],
    policy_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Summarize when the replay policy activates and how post-activation behaves."""

    activation_signal_row = next(
        (
            row
            for row in activation_timeline_rows
            if _as_bool(row.get("activate_next_bank", False))
        ),
        None,
    )
    post_rows = [
        row
        for row in policy_rows
        if str(row.get("bank_phase", "")) == "post_activation"
    ]
    post_summary = summarize_case_rows(post_rows) if post_rows else None
    return {
        "activated": activation_signal_row is not None,
        "activation_signal_bank_version": _as_int(
            activation_signal_row.get("bank_version", -1) if activation_signal_row else -1
        ),
        "activation_bank_version": (
            _as_int(activation_signal_row.get("bank_version", -1)) + 1
            if activation_signal_row is not None
            else -1
        ),
        "activation_case_index": (
            _as_int(activation_signal_row.get("global_case_end_index", -1)) + 1
            if activation_signal_row is not None
            else -1
        ),
        "warmup_case_count": (
            _as_int(activation_signal_row.get("global_case_end_index", -1)) + 1
            if activation_signal_row is not None
            else 0
        ),
        "post_activation_case_count": int(post_summary.get("sample_count", 0))
        if post_summary
        else 0,
        "post_activation_mean_search_latency_ms": _as_float(
            post_summary.get("mean_search_latency_ms", 0.0) if post_summary else 0.0
        ),
        "post_activation_unsafe_skip_rate_pct": _as_float(
            post_summary.get("unsafe_skip_rate_pct", 0.0) if post_summary else 0.0
        ),
        "post_activation_over_trigger_rate_pct": _as_float(
            post_summary.get("over_trigger_rate_pct", 0.0) if post_summary else 0.0
        ),
        "post_activation_mean_final_bw_pct_of_bandpilot": _as_float(
            post_summary.get("mean_final_bw_pct_of_bandpilot", 0.0)
            if post_summary
            else 0.0
        ),
    }


def run_adaptive_knn_replay(
    *,
    samples: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, object]],
    config: AdaptiveKNNConfig,
) -> Dict[str, object]:
    """Replay compare-style runtime samples with the adaptive-kNN policy.

    The formal path uses only the already-admitted bank before activation and
    falls back to BandPilot during warmup. The shadow path evaluates each bank as
    if its current queries were also labeled, which estimates whether the next
    bank is safe to activate.
    """

    if len(samples) != len(feature_rows):
        raise ValueError("samples and feature_rows must have the same length.")

    bank = OnlineMismatchBank.empty(config.build_online_bank_config())
    banks = _build_banks(samples=samples, feature_rows=feature_rows, config=config)

    policy_rows: List[Dict[str, object]] = []
    query_rows: List[Dict[str, object]] = []
    labeled_bank_rows: List[Dict[str, object]] = []
    activation_timeline_rows: List[Dict[str, object]] = []

    bank_is_active = False
    activation_pass_streak = 0
    global_case_cursor = 0

    for bank_version, bank_pairs in enumerate(banks):
        bank_phase = "post_activation" if bank_is_active else "pre_activation"
        train_size_before = bank.labeled_size
        bank_query_rows: List[Dict[str, object]] = []
        bank_decision_supports: List[float] = []

        # Evaluate queries against the bank that was available before this batch.
        for sample, feature_row in bank_pairs:
            decision_start = time.perf_counter()
            online_risk_row = bank.evaluate_query(
                query_sample=sample,
                query_feature_row=feature_row,
            )
            shadow_skip, shadow_reason = _evaluate_shadow_mismatch(
                sample=sample,
                online_risk_row=online_risk_row,
                risk_threshold=float(config.risk_threshold),
            )
            decision_overhead_ms = 1000.0 * (time.perf_counter() - decision_start)
            bank_decision_supports.append(
                _as_float(online_risk_row.get("online_mismatch_support", 0))
            )

            formal_skip = shadow_skip if bank_is_active else False
            formal_reason = (
                shadow_reason if bank_is_active else "pre_activation_bandpilot_fallback"
            )
            selected_source = "eha" if formal_skip else "bandpilot"

            policy_rows.append(
                _build_policy_row(
                    sample=sample,
                    feature_row=feature_row,
                    bank_version=bank_version,
                    bank_phase=bank_phase,
                    train_size_before=train_size_before,
                    global_case_index=global_case_cursor,
                    online_risk_row=online_risk_row,
                    selected_source=selected_source,
                    trigger_reason=formal_reason,
                    decision_overhead_ms=decision_overhead_ms,
                    algorithm_label=str(config.algorithm_label),
                )
            )

            bank_query_rows.append(
                _build_query_row(
                    sample=sample,
                    feature_row=feature_row,
                    bank_version=bank_version,
                    bank_phase=bank_phase,
                    train_size_before=train_size_before,
                    global_case_index=global_case_cursor,
                    online_risk_row=online_risk_row,
                    shadow_skip=shadow_skip,
                    shadow_reason=shadow_reason,
                    formal_skip=formal_skip,
                    formal_reason=formal_reason,
                    decision_overhead_ms=decision_overhead_ms,
                )
            )
            global_case_cursor += 1

        # Shadow evaluation estimates the next bank using the current batch as
        # additional labeled support.
        shadow_eval_start = time.perf_counter()
        shadow_eval_bank = OnlineMismatchBank.from_rows(
            config=bank.config,
            labeled_rows=[*bank.labeled_samples, *bank_query_rows],
        )
        shadow_supports: List[float] = []
        shadow_rows: List[Dict[str, object]] = []
        for query_row in bank_query_rows:
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
                risk_threshold=float(config.risk_threshold),
            )
            shadow_rows.append(
                _build_policy_row(
                    sample=query_row,
                    feature_row=query_row,
                    bank_version=bank_version,
                    bank_phase=bank_phase,
                    train_size_before=train_size_before,
                    global_case_index=_as_int(query_row.get("global_case_index", 0)),
                    online_risk_row=shadow_eval_risk_row,
                    selected_source="eha" if shadow_skip else "bandpilot",
                    trigger_reason=shadow_reason,
                    decision_overhead_ms=0.0,
                    algorithm_label="shadow_adaptive_knn",
                )
            )
        shadow_eval_time_ms = 1000.0 * (time.perf_counter() - shadow_eval_start)

        shadow_summary = summarize_case_rows(shadow_rows)
        shadow_unsafe_skip_rate_pct = _as_float(
            shadow_summary.get("unsafe_skip_rate_pct", 100.0)
        )
        shadow_over_trigger_rate_pct = _as_float(
            shadow_summary.get("over_trigger_rate_pct", 100.0)
        )
        shadow_support_insufficient_case_count = int(
            sum(1 for row in shadow_rows if _as_bool(row.get("support_insufficient", False)))
        )
        shadow_low_trust_case_count = int(
            sum(
                1
                for row in shadow_rows
                if _as_bool(row.get("online_mismatch_low_trust", False))
            )
        )

        bank_meets_activation_gate = (
            shadow_unsafe_skip_rate_pct
            <= float(config.activation_criteria.unsafe_skip_rate_pct_max)
            and shadow_over_trigger_rate_pct
            < float(config.activation_criteria.over_trigger_rate_pct_max)
            and shadow_support_insufficient_case_count
            <= int(config.activation_criteria.support_insufficient_case_count_max)
        )
        activation_pass_streak = (
            activation_pass_streak + 1 if bank_meets_activation_gate else 0
        )
        activate_next_bank = (
            (not bank_is_active or not config.activation_criteria.sticky_activation)
            and activation_pass_streak
            >= int(config.activation_criteria.min_consecutive_pass_banks)
        )
        bank_active_next = bank_is_active or activate_next_bank

        # Admission policy:
        # - pre-activation: admit BandPilot-labeled cases.
        # - post-activation: admit dual-observed cases.
        admitted_rows = _build_admission_rows(
            bank_version=bank_version,
            bank_phase=bank_phase,
            query_rows=bank_query_rows,
        )
        update_start = time.perf_counter()
        bank.append_labeled_rows(admitted_rows)
        bank_update_time_ms = 1000.0 * (time.perf_counter() - update_start)

        activation_timeline_rows.append(
            {
                "bank_version": bank_version,
                "bank_phase": bank_phase,
                "bank_size": len(bank_pairs),
                "global_case_start_index": global_case_cursor - len(bank_pairs),
                "global_case_end_index": global_case_cursor - 1,
                "train_size_before": train_size_before,
                "admitted_count": len(admitted_rows),
                "train_size_after": bank.labeled_size,
                "bank_active_before": bank_is_active,
                "bank_meets_activation_gate": bank_meets_activation_gate,
                "activation_pass_streak": activation_pass_streak,
                "activate_next_bank": activate_next_bank,
                "bank_active_next": bank_active_next,
                "shadow_unsafe_skip_rate_pct": shadow_unsafe_skip_rate_pct,
                "shadow_over_trigger_rate_pct": shadow_over_trigger_rate_pct,
                "shadow_support_insufficient_case_count": shadow_support_insufficient_case_count,
                "shadow_low_trust_case_count": shadow_low_trust_case_count,
                "shadow_mean_search_latency_ms": _as_float(
                    shadow_summary.get("mean_search_latency_ms", 0.0)
                ),
                "shadow_mean_support": _mean(shadow_supports),
                "shadow_p95_support": _percentile(shadow_supports, 0.95),
                "mean_decision_support": _mean(bank_decision_supports),
                "p95_decision_support": _percentile(bank_decision_supports, 0.95),
                "shadow_eval_time_ms": shadow_eval_time_ms,
                "bank_update_time_ms": bank_update_time_ms,
            }
        )

        query_rows.extend(bank_query_rows)
        labeled_bank_rows.extend(admitted_rows)
        bank_is_active = bank_active_next

    return {
        "policy_rows": policy_rows,
        "query_rows": query_rows,
        "labeled_bank_rows": labeled_bank_rows,
        "activation_timeline_rows": activation_timeline_rows,
        "activation_summary": _build_activation_summary(
            activation_timeline_rows=activation_timeline_rows,
            policy_rows=policy_rows,
        ),
    }


def build_compare_records_from_replay(
    *,
    policy_rows: Sequence[Mapping[str, object]],
    config: AdaptiveKNNConfig,
) -> List[Dict[str, object]]:
    """Convert replay policy rows into compare-compatible result records."""

    records: List[Dict[str, object]] = []
    for row in policy_rows:
        max_bw = _as_float(row.get("max_bw", 0.0))
        chosen_final_bw = _as_float(row.get("chosen_final_bw", 0.0))
        chosen_standalone_bw = _as_float(row.get("chosen_standalone_bw", 0.0))
        record = {
            "test_num": _as_int(row.get("test_num", 0)),
            "repeat_idx": _as_int(row.get("repeat_idx", 0)),
            "total_gpu": _as_int(row.get("total_gpu", 0)),
            "bw_type": str(row.get("bw_type", "")),
            "cluster_type": str(row.get("cluster_type", "")),
            "if_dynamic": _as_bool(row.get("if_dynamic", False)),
            "seed_used": row.get("seed_used"),
            "avail_gpu_count": _as_int(row.get("avail_gpu_count", 0)),
            "avail_signature": str(row.get("avail_signature", "")),
            "max_bw": max_bw,
            "contention_mode": str(row.get("contention_mode", "")),
            "search_if_real_data_global": _as_bool(row.get("search_if_real_data", False)),
            "background_gpu_count": _as_int(row.get("background_gpu_count", 0)),
            "background_signature": str(row.get("background_signature", "")),
            "occupancy_seed": _as_int(row.get("occupancy_seed", 0)),
            "probe_job_id": _as_int(row.get("probe_job_id", 0)),
            "algorithm": str(config.algorithm_label),
            "search_if_real_data_effective": _as_bool(
                row.get("search_if_real_data", False)
            ),
            "final_bw": chosen_final_bw,
            "standalone_bw": chosen_standalone_bw,
            "final_utilization": (
                100.0 * chosen_final_bw / max_bw if max_bw > 0 else 0.0
            ),
            "standalone_utilization": (
                100.0 * chosen_standalone_bw / max_bw if max_bw > 0 else 0.0
            ),
            "elapsed_time": _as_float(row.get("chosen_elapsed_time_s", 0.0)),
            "predict_time": _as_float(row.get("chosen_predict_time_s", 0.0)),
            "contention_time": _as_float(row.get("chosen_contention_time_s", 0.0)),
            "selected_gpu_count": _as_int(row.get("selected_gpu_count", 0)),
            "combo_signature": str(row.get("chosen_combo_signature", "")),
            "contention_job_id": _as_int(row.get("probe_job_id", 0)),
            "adaptive_policy_name": "adaptive_knn",
            "adaptive_trigger_reason": str(row.get("trigger_reason", "")),
            "adaptive_bank_version": _as_int(row.get("bank_version", -1)),
            "adaptive_bank_phase": str(row.get("bank_phase", "")),
            "adaptive_train_size_before": _as_int(row.get("train_size_before", 0)),
            "adaptive_online_risk": _as_float(
                row.get("online_mismatch_knn_risk", 0.0)
            ),
            "adaptive_online_support": _as_int(
                row.get("online_mismatch_support", 0)
            ),
            "adaptive_online_scope": str(row.get("online_mismatch_scope", "")),
            "adaptive_online_conflict_risk": _as_float(
                row.get("online_mismatch_conflict_risk", 0.0)
            ),
            "adaptive_online_low_trust": _as_bool(
                row.get("online_mismatch_low_trust", False)
            ),
            "adaptive_support_insufficient": _as_bool(
                row.get("support_insufficient", False)
            ),
            "adaptive_decision_overhead_ms": _as_float(
                row.get("decision_overhead_ms", 0.0)
            ),
        }
        records.append(record)
    return records
