"""Shared outcome metrics for BandPilot adaptive replay rows.

The metrics classify two policy errors:
`unsafe_skip`, where EHA skips PTS even though PTS was helpful, and
`over_trigger`, where the policy runs PTS although EHA was already sufficient.
These summaries are used by compare replay, runtime-bank diagnostics, and unit
tests without depending on historical calibration scripts.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from algorithms.adaptive_gate_support.utils import (
    _as_bool,
    _as_float,
    _as_int,
    _mean,
    _percentile,
)


def _latency_ms_from_row(row: Mapping[str, object], *, prefix: str) -> float:
    """Read a row latency field and normalize it to milliseconds."""
    if f"{prefix}_search_latency_ms" in row:
        return float(row[f"{prefix}_search_latency_ms"])
    if f"{prefix}_elapsed_time_s" in row:
        return 1000.0 * float(row[f"{prefix}_elapsed_time_s"])
    if f"{prefix}_search_latency_s" in row:
        return 1000.0 * float(row[f"{prefix}_search_latency_s"])
    return 0.0


def derive_case_outcome_fields(
    row: Mapping[str, object],
    *,
    skip_key: str = "skip_pts",
    helpful_key: str = "pts_helpful",
    feasible_key: str = "eha_feasible",
) -> Dict[str, object]:
    """Derive case-level safety and overhead outcomes from one policy row."""

    skip_pts = _as_bool(row.get(skip_key, False))
    pts_helpful = _as_bool(row.get(helpful_key, False))
    eha_feasible = _as_bool(row.get(feasible_key, True))

    unsafe_skip = bool(skip_pts and pts_helpful)
    over_trigger = bool((not skip_pts) and eha_feasible and (not pts_helpful))

    chosen_bw_pct = _as_float(row.get("chosen_final_bw_pct_of_bandpilot", 0.0))
    pts_gain_pct = _as_float(row.get("pts_gain_pct_of_bandpilot", 0.0))
    chosen_latency_ms = _latency_ms_from_row(row, prefix="chosen")
    bandpilot_latency_ms = _latency_ms_from_row(row, prefix="bandpilot")
    eha_latency_ms = _latency_ms_from_row(row, prefix="eha")

    return {
        "unsafe_skip": unsafe_skip,
        "over_trigger": over_trigger,
        "false_safe": unsafe_skip,
        "oracle_should_skip": bool(eha_feasible and (not pts_helpful)),
        "unsafe_skip_bw_regret_pct_points": (
            max(0.0, 100.0 - chosen_bw_pct) if unsafe_skip else 0.0
        ),
        "unsafe_skip_pts_gain_pct": pts_gain_pct if unsafe_skip else 0.0,
        "over_trigger_extra_search_ms": (
            max(0.0, bandpilot_latency_ms - eha_latency_ms) if over_trigger else 0.0
        ),
        "chosen_search_latency_ms": chosen_latency_ms,
        "bandpilot_search_latency_ms": bandpilot_latency_ms,
        "eha_search_latency_ms": eha_latency_ms,
    }


def summarize_case_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    passthrough_keys: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Summarize safety, bandwidth, and latency metrics for a row group."""
    if not rows:
        raise ValueError("cannot summarize empty case rows")

    derived_rows = [derive_case_outcome_fields(row) for row in rows]
    chosen_latencies = [float(item["chosen_search_latency_ms"]) for item in derived_rows]
    final_bw_pct = [float(row.get("chosen_final_bw_pct_of_bandpilot", 0.0)) for row in rows]
    trigger_flags = [100.0 if (not _as_bool(row.get("skip_pts", False))) else 0.0 for row in rows]
    skip_flags = [100.0 if _as_bool(row.get("skip_pts", False)) else 0.0 for row in rows]
    unsafe_skip_flags = [100.0 if bool(item["unsafe_skip"]) else 0.0 for item in derived_rows]
    over_trigger_flags = [100.0 if bool(item["over_trigger"]) else 0.0 for item in derived_rows]
    helpful_flags = [100.0 if _as_bool(row.get("pts_helpful", False)) else 0.0 for row in rows]
    same_combo_flags = [
        100.0
        if str(row.get("chosen_combo_signature", "")) == str(row.get("bandpilot_combo_signature", ""))
        else 0.0
        for row in rows
        if "chosen_combo_signature" in row or "bandpilot_combo_signature" in row
    ]

    unsafe_skip_bw_regrets = [
        float(item["unsafe_skip_bw_regret_pct_points"])
        for item in derived_rows
        if bool(item["unsafe_skip"])
    ]
    unsafe_skip_pts_gains = [
        float(item["unsafe_skip_pts_gain_pct"])
        for item in derived_rows
        if bool(item["unsafe_skip"])
    ]
    over_trigger_extra_latencies = [
        float(item["over_trigger_extra_search_ms"])
        for item in derived_rows
        if bool(item["over_trigger"])
    ]

    summary: Dict[str, object] = {
        "sample_count": len(rows),
        "mean_search_latency_ms": _mean(chosen_latencies),
        "p95_search_latency_ms": _percentile(chosen_latencies, 0.95),
        "mean_final_bw_pct_of_bandpilot": _mean(final_bw_pct),
        "skip_rate_pct": _mean(skip_flags),
        "trigger_rate_pct": _mean(trigger_flags),
        "false_skip_rate_pct": _mean(unsafe_skip_flags),
        "unsafe_skip_rate_pct": _mean(unsafe_skip_flags),
        "over_trigger_rate_pct": _mean(over_trigger_flags),
        "helpful_case_rate_pct": _mean(helpful_flags),
        "false_safe_case_count": int(sum(1 for item in derived_rows if bool(item["unsafe_skip"]))),
        "unsafe_skip_case_count": int(sum(1 for item in derived_rows if bool(item["unsafe_skip"]))),
        "over_trigger_case_count": int(sum(1 for item in derived_rows if bool(item["over_trigger"]))),
        "unsafe_skip_mean_bw_regret_pct_points": _mean(unsafe_skip_bw_regrets),
        "unsafe_skip_mean_pts_gain_pct": _mean(unsafe_skip_pts_gains),
        "over_trigger_mean_extra_search_ms": _mean(over_trigger_extra_latencies),
        "over_trigger_p95_extra_search_ms": _percentile(over_trigger_extra_latencies, 0.95),
        "bw_drop_gt_1pct_case_count": 0,
    }
    if same_combo_flags:
        summary["same_combo_pct_vs_bandpilot"] = _mean(same_combo_flags)

    for row in rows:
        if _as_float(row.get("chosen_final_bw_pct_of_bandpilot", 100.0)) < 99.0:
            summary["bw_drop_gt_1pct_case_count"] = summary.get("bw_drop_gt_1pct_case_count", 0) + 1

    if passthrough_keys:
        first_row = rows[0]
        for key in passthrough_keys:
            if key in first_row:
                summary[key] = first_row[key]
    return summary


def build_cluster_mode_k_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    include_case_group: bool = True,
    passthrough_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Build summaries grouped by cluster, policy, contention mode, and k."""
    grouped: Dict[tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            row.get("cluster_type", ""),
            row.get("policy_label", ""),
            row.get("policy_id", ""),
            row.get("case_group", "") if include_case_group else "",
            row.get("contention_mode", ""),
            _as_int(row.get("test_num", 0)),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (cluster_type, policy_label, policy_id, case_group, contention_mode, test_num), group_rows in sorted(
        grouped.items(),
        key=lambda item: item[0],
    ):
        summary = summarize_case_rows(group_rows, passthrough_keys=passthrough_keys)
        summary.update(
            {
                "cluster_type": cluster_type,
                "policy_label": policy_label,
                "policy_id": policy_id,
                "contention_mode": contention_mode,
                "test_num": test_num,
            }
        )
        if include_case_group:
            summary["case_group"] = case_group
        summary_rows.append(summary)
    return summary_rows


def partition_error_case_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Partition rows into unsafe-skip and over-trigger case lists."""
    unsafe_skip_rows: List[Dict[str, object]] = []
    over_trigger_rows: List[Dict[str, object]] = []
    for row in rows:
        enriched: Dict[str, object] = dict(row)
        enriched.update(derive_case_outcome_fields(row))
        if bool(enriched["unsafe_skip"]):
            unsafe_skip_rows.append(enriched)
        if bool(enriched["over_trigger"]):
            over_trigger_rows.append(enriched)
    return unsafe_skip_rows, over_trigger_rows
