"""Build reviewer-facing scalability reports from raw benchmark artifacts.

The builder audits expected rows, records complete/partial/missing status, and
writes summary CSV/TEX plus a manifest that states whether the regenerated
artifact is full or partial evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from evaluation.scalability import FULL_CONFIG_PATH, WELL_SHOW_REPORT_DIR


# Canonical public benchmark order. Legacy labels remain accepted through aliases.
ALGORITHM_ORDER: List[str] = ["EHA", "PTS", "BandPilot"]
ALGORITHM_DISPLAY_ALIASES: Dict[str, str] = {
    "EHA": "EHA",
    "EHA-only": "EHA",
    "PTS": "PTS",
    "HU-" + "PTS": "PTS",
    "HU-" + "PTS-only": "PTS",
    "BandPilot": "BandPilot",
    "HU-" + "Adaptive": "BandPilot",
    "HU-" + "BandPilot": "BandPilot",
}


# Public figure style for the three displayed algorithms.
ALGORITHM_COLORS: Dict[str, str] = {
    "EHA": "#4c78a8",
    "PTS": "#f58518",
    "BandPilot": "#54a24b",
}

ALGORITHM_MARKERS: Dict[str, str] = {
    "EHA": "o",
    "PTS": "s",
    "BandPilot": "D",
}

ALGORITHM_LINESTYLES: Dict[str, str] = {
    "EHA": "-",
    "PTS": "--",
    "BandPilot": "-",
}


# Benchmark tiers and display names used by audit tables and figures.
TIER_ORDER: List[str] = ["tier1", "tier2", "predictor_profile", "tier4"]
TIER_DISPLAY_NAMES: Dict[str, str] = {
    "tier1": "Tier 1 Real-Dispatch",
    "tier2": "Tier 2 Scaled-Search",
    "predictor_profile": "Tier 3 Predictor-Profile",
    "tier4": "Tier 4 Latency-Synthesis",
}

TIER_SHORT_NAMES: Dict[str, str] = {
    "tier1": "Real32",
    "tier2": "Scaled",
    "predictor_profile": "Predictor",
    "tier4": "Synth",
}


# Completeness status ordering used by the heatmap.
STATUS_ORDER: List[str] = ["missing", "partial", "complete", "overflow"]
STATUS_TO_SCORE: Dict[str, float] = {
    "missing": 0.0,
    "partial": 0.55,
    "complete": 1.0,
    "overflow": 0.2,
}

STATUS_TO_LABEL: Dict[str, str] = {
    "missing": "Missing",
    "partial": "Partial",
    "complete": "Complete",
    "overflow": "Overflow",
}


# Sanitize benchmark cluster names before embedding them in per-cluster CSV paths.
CLUSTER_TAG_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class BenchmarkSpec:
    """Benchmark dimensions required by the scalability report builder.

    The spec records benchmark sweeps, representative public slices, expected
    row counts, and evidence-tier settings so the report can audit raw artifacts
    before generating reviewer-facing summaries.
    """

    cluster_total_gpu: int
    cluster_types: List[str]
    algorithms: List[str]
    public_algorithms: List[str]
    representative_contention_mode: str
    representative_avail_ratio: float
    representative_inter_pod_factor: float
    representative_k_values: List[int]
    tier1_k_values: List[int]
    tier1_contention_modes: List[str]
    tier1_repeat_num: int
    tier2_gpu_counts: List[int]
    tier2_k_values: List[int]
    tier2_avail_ratios: List[float]
    tier2_contention_modes: List[str]
    tier2_inter_pod_factors: List[float]
    tier2_repeat_num: int
    tier2_public_repeat_num: Optional[int]
    predictor_node_counts: List[int]
    predictor_inference_repeats: int
    tier4_target_gpu_counts: List[int]


def parse_args() -> argparse.Namespace:
    """Parse the report-builder CLI.

    `strict` mode rejects incomplete artifacts. `partial` mode allows the named
    partial clusters while preserving their incomplete status in the report.
    """

    parser = argparse.ArgumentParser(description="Build scalability well-show partial reports")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory that stores scalability benchmark raw artifact CSVs and search_overhead.log",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=WELL_SHOW_REPORT_DIR,
        help="Directory that stores generated reports and figures",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=FULL_CONFIG_PATH,
        help="Benchmark config used to infer expected row counts",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "partial"],
        default="strict",
        help="Whether incomplete clusters are rejected (`strict`) or explicitly marked (`partial`)",
    )
    parser.add_argument(
        "--partial-clusters",
        type=str,
        nargs="*",
        default=[],
        help="Clusters that are allowed to remain incomplete in `partial` mode",
    )
    return parser.parse_args()


def _default_spec() -> BenchmarkSpec:
    """Return defaults that mirror `evaluation/scalability/configs/full.yaml`."""

    return BenchmarkSpec(
        cluster_total_gpu=32,
        cluster_types=["H100_26H100_27H100_28H100_29", "Het-4Mix"],
        algorithms=list(ALGORITHM_ORDER),
        public_algorithms=list(ALGORITHM_ORDER),
        representative_contention_mode="common",
        representative_avail_ratio=0.7,
        representative_inter_pod_factor=0.7,
        representative_k_values=[16, 64],
        tier1_k_values=[4, 8, 12, 16, 20, 24, 28],
        tier1_contention_modes=["idle", "common", "intensive"],
        tier1_repeat_num=50,
        tier2_gpu_counts=[64, 128, 256, 512, 1024],
        tier2_k_values=[8, 16, 32, 64],
        tier2_avail_ratios=[0.5, 0.7, 0.9],
        tier2_contention_modes=["common", "intensive"],
        tier2_inter_pod_factors=[0.5, 0.7],
        tier2_repeat_num=10,
        tier2_public_repeat_num=20,
        predictor_node_counts=[4, 8, 16, 32, 64, 128, 256, 512],
        predictor_inference_repeats=200,
        tier4_target_gpu_counts=[512, 1024, 2048, 4096],
    )


def _sanitize_cluster_tag(cluster_type: str) -> str:
    """Convert a cluster type into a filesystem-safe benchmark tag."""

    return CLUSTER_TAG_RE.sub("_", str(cluster_type)).strip("_")


def _load_benchmark_spec(config_path: Path) -> BenchmarkSpec:
    """Load benchmark dimensions needed for expected-row auditing.

    Missing config files fall back to the built-in full benchmark spec so older
    artifacts can still be audited and reported.
    """

    default = _default_spec()
    if not config_path.exists():
        return default

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cluster_cfg = dict(raw.get("cluster", {}))
    eval_cfg = dict(raw.get("evaluation", {}))
    benchmark_cfg = dict(eval_cfg.get("scalability_benchmark", {}))
    public_cfg = dict(benchmark_cfg.get("public_view", {}))
    tier1_cfg = dict(benchmark_cfg.get("real_dispatch", {}))
    tier2_cfg = dict(benchmark_cfg.get("scaled_search", {}))
    predictor_cfg = dict(benchmark_cfg.get("predictor_profile", {}))
    tier4_cfg = dict(benchmark_cfg.get("latency_synthesis", {}))

    return BenchmarkSpec(
        cluster_total_gpu=int(cluster_cfg.get("total_gpu", default.cluster_total_gpu)),
        cluster_types=list(cluster_cfg.get("cluster_types", default.cluster_types)),
        algorithms=[
            ALGORITHM_DISPLAY_ALIASES.get(str(value), str(value))
            for value in benchmark_cfg.get("algorithms", default.algorithms)
        ],
        public_algorithms=[
            ALGORITHM_DISPLAY_ALIASES.get(str(value), str(value))
            for value in public_cfg.get("public_algorithms", default.public_algorithms)
        ],
        representative_contention_mode=str(
            public_cfg.get("representative_contention_mode", default.representative_contention_mode)
        ),
        representative_avail_ratio=float(
            public_cfg.get("representative_avail_ratio", default.representative_avail_ratio)
        ),
        representative_inter_pod_factor=float(
            public_cfg.get("representative_inter_pod_factor", default.representative_inter_pod_factor)
        ),
        representative_k_values=[int(value) for value in public_cfg.get("representative_k_values", default.representative_k_values)],
        tier1_k_values=[int(value) for value in tier1_cfg.get("k_values", default.tier1_k_values)],
        tier1_contention_modes=list(tier1_cfg.get("contention_modes", default.tier1_contention_modes)),
        tier1_repeat_num=int(tier1_cfg.get("repeat_num", default.tier1_repeat_num)),
        tier2_gpu_counts=[int(value) for value in tier2_cfg.get("gpu_counts", default.tier2_gpu_counts)],
        tier2_k_values=[int(value) for value in tier2_cfg.get("k_values", default.tier2_k_values)],
        tier2_avail_ratios=[float(value) for value in tier2_cfg.get("avail_ratios", default.tier2_avail_ratios)],
        tier2_contention_modes=list(tier2_cfg.get("contention_modes", default.tier2_contention_modes)),
        tier2_inter_pod_factors=[float(value) for value in tier2_cfg.get("inter_pod_factors", default.tier2_inter_pod_factors)],
        tier2_repeat_num=int(tier2_cfg.get("repeat_num", default.tier2_repeat_num)),
        tier2_public_repeat_num=(
            None
            if tier2_cfg.get("repeat_num_public_slice", default.tier2_public_repeat_num) is None
            else int(tier2_cfg.get("repeat_num_public_slice", default.tier2_public_repeat_num))
        ),
        predictor_node_counts=[int(value) for value in predictor_cfg.get("inference_node_counts", default.predictor_node_counts)],
        predictor_inference_repeats=int(
            predictor_cfg.get("inference_repeats", default.predictor_inference_repeats)
        ),
        tier4_target_gpu_counts=[int(value) for value in tier4_cfg.get("target_gpu_counts", default.tier4_target_gpu_counts)],
    )


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if present and normalize historical algorithm labels."""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "algorithm" in frame.columns:
        frame["algorithm"] = frame["algorithm"].apply(
            lambda value: ALGORITHM_DISPLAY_ALIASES.get(str(value), str(value))
        )
    if "algorithm_display_name" in frame.columns:
        frame["algorithm_display_name"] = frame["algorithm_display_name"].apply(
            lambda value: ALGORITHM_DISPLAY_ALIASES.get(str(value), str(value))
        )
    if "selected_backend_display_name" in frame.columns:
        frame["selected_backend_display_name"] = frame["selected_backend_display_name"].apply(
            lambda value: ALGORITHM_DISPLAY_ALIASES.get(str(value), str(value))
        )
    return frame


def _resolve_artifact_paths(artifact_dir: Path, cluster_type: str) -> Dict[str, Path]:
    """Resolve all expected raw artifact paths for one cluster."""

    cluster_tag = _sanitize_cluster_tag(cluster_type)
    return {
        "tier1": artifact_dir / f"tier1_{cluster_tag}.csv",
        "tier2": artifact_dir / f"scaled_search_{cluster_tag}.csv",
        "predictor_profile": artifact_dir / f"predictor_latency_profile_{cluster_tag}.csv",
        "tier4": artifact_dir / f"synthesized_dispatch_latency_{cluster_tag}.csv",
    }


def _is_public_slice(spec: BenchmarkSpec, avail_ratio: float, contention_mode: str, inter_pod_factor: float, k: int) -> bool:
    """Return whether a Tier 2 case belongs to the reviewer-facing slice."""

    return (
        str(contention_mode) == str(spec.representative_contention_mode)
        and math.isclose(float(avail_ratio), float(spec.representative_avail_ratio), abs_tol=1e-8)
        and math.isclose(float(inter_pod_factor), float(spec.representative_inter_pod_factor), abs_tol=1e-8)
        and int(k) in set(int(value) for value in spec.representative_k_values)
    )


def _build_tier2_expected_repeat_map(spec: BenchmarkSpec) -> Tuple[int, Dict[int, int]]:
    """Compute expected Tier 2 rows in total and by scale.

    This mirrors `benchmark.py::_build_tier2_group_specs()`: each scenario uses
    `target_avail = max(max(k_values), round(total_gpu * avail_ratio))`, keeps
    feasible `k <= target_avail`, and expands representative public groups to
    `repeat_num_public_slice` when configured.
    """

    expected_total_rows = 0
    expected_rows_by_scale: Dict[int, int] = {int(scale): 0 for scale in spec.tier2_gpu_counts}
    max_k = max(int(value) for value in spec.tier2_k_values)
    algorithm_count = len(spec.algorithms)

    for total_gpu in spec.tier2_gpu_counts:
        for inter_pod_factor in spec.tier2_inter_pod_factors:
            for avail_ratio in spec.tier2_avail_ratios:
                target_avail = max(max_k, int(round(int(total_gpu) * float(avail_ratio))))
                for contention_mode in spec.tier2_contention_modes:
                    feasible_ks = [
                        int(value) for value in spec.tier2_k_values if int(value) <= int(target_avail)
                    ]
                    if not feasible_ks:
                        continue
                    repeat_num = int(spec.tier2_repeat_num)
                    has_public_slice = any(
                        _is_public_slice(
                            spec=spec,
                            avail_ratio=float(avail_ratio),
                            contention_mode=str(contention_mode),
                            inter_pod_factor=float(inter_pod_factor),
                            k=int(k),
                        )
                        for k in feasible_ks
                    )
                    if spec.tier2_public_repeat_num is not None and has_public_slice:
                        repeat_num = max(int(repeat_num), int(spec.tier2_public_repeat_num))
                    row_count = int(repeat_num) * len(feasible_ks) * int(algorithm_count)
                    expected_total_rows += row_count
                    expected_rows_by_scale[int(total_gpu)] += row_count

    return expected_total_rows, expected_rows_by_scale


def _build_expected_row_map(spec: BenchmarkSpec) -> Dict[str, int]:
    """Build expected row counts for every audited artifact tier."""

    tier2_total_rows, tier2_rows_by_scale = _build_tier2_expected_repeat_map(spec)
    nearest_scale_rows: Dict[int, int] = {}
    available_scales = sorted(int(scale) for scale in spec.tier2_gpu_counts)

    # Tier 4 reuses the nearest observed Tier 2 scale for each target GPU count.
    for target_gpu in spec.tier4_target_gpu_counts:
        nearest_scale = min(available_scales, key=lambda value: abs(int(value) - int(target_gpu)))
        nearest_scale_rows[int(target_gpu)] = int(tier2_rows_by_scale[int(nearest_scale)])

    return {
        "tier1": (
            len(spec.tier1_k_values)
            * len(spec.tier1_contention_modes)
            * int(spec.tier1_repeat_num)
            * len(spec.algorithms)
        ),
        "tier2": int(tier2_total_rows),
        "predictor_profile": len(spec.predictor_node_counts),
        "tier4": int(sum(nearest_scale_rows.values())),
    }


def _build_tier4_reference_scale_map(spec: BenchmarkSpec) -> Dict[int, int]:
    """Map each Tier 4 target GPU count to the nearest Tier 2 reference scale."""

    available_scales = sorted(int(scale) for scale in spec.tier2_gpu_counts)
    reference_map: Dict[int, int] = {}
    for target_gpu in spec.tier4_target_gpu_counts:
        reference_map[int(target_gpu)] = min(
            available_scales,
            key=lambda value: abs(int(value) - int(target_gpu)),
        )
    return reference_map


def _format_scalar_literal(value: Any) -> str:
    """Format one scalar for compact report literals."""

    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
        return f"{numeric:.1f}"
    return str(value)


def _format_sequence_literal(values: Sequence[Any]) -> str:
    """Format a sequence as a compact `{a, b, c}` report literal."""

    return "{" + ", ".join(_format_scalar_literal(value) for value in values) + "}"


def _format_seconds(value: Any) -> str:
    """Format a duration with precision matched to its magnitude."""

    numeric = float(value)
    if numeric >= 10.0:
        return f"{numeric:.2f}s"
    if numeric >= 1.0:
        return f"{numeric:.3f}s"
    if numeric >= 0.1:
        return f"{numeric:.3f}s"
    return f"{numeric:.4f}s"


def _extract_repeat_hist_from_note(note: str) -> str:
    """Extract the `repeat_hist=...` fragment from an audit note."""

    if not note:
        return ""
    for part in str(note).split("; "):
        if part.startswith("repeat_hist="):
            return part.split("=", 1)[1]
    return ""


def _status_from_counts(actual_rows: int, expected_rows: int) -> str:
    """Classify actual versus expected row counts."""

    if int(actual_rows) <= 0:
        return "missing"
    if int(actual_rows) == int(expected_rows):
        return "complete"
    if int(actual_rows) < int(expected_rows):
        return "partial"
    return "overflow"


def _summarize_repeat_histogram(df: pd.DataFrame) -> str:
    """Summarize scenario-repeat multiplicities for Tier 2 and Tier 4."""

    if df.empty:
        return ""
    required_columns = ["total_gpu", "k", "avail_ratio", "contention_mode", "inter_pod_factor", "algorithm"]
    if any(column not in df.columns for column in required_columns):
        return ""

    grouped = (
        df.groupby(required_columns, dropna=False)
        .size()
        .sort_values()
    )
    repeat_hist = grouped.value_counts().sort_index()
    return ", ".join(f"{int(repeat)}x{int(count)}" for repeat, count in repeat_hist.items())


def _summarize_scale_rows(df: pd.DataFrame) -> str:
    """Summarize per-scale row counts for traceability."""

    if df.empty or "total_gpu" not in df.columns:
        return ""
    counts = (
        df.groupby("total_gpu")
        .size()
        .sort_index()
    )
    return ", ".join(f"{int(scale)}:{int(count)}" for scale, count in counts.items())


def build_artifact_audit(artifact_dir: Path, spec: BenchmarkSpec) -> pd.DataFrame:
    """Build the per-cluster, per-tier artifact audit table."""

    expected_rows = _build_expected_row_map(spec)
    records: List[Dict[str, Any]] = []

    for cluster_type in spec.cluster_types:
        paths = _resolve_artifact_paths(artifact_dir, cluster_type)
        for tier_name in TIER_ORDER:
            path = paths[tier_name]
            frame = _safe_read_csv(path)
            actual_rows = int(len(frame))
            expected = int(expected_rows[tier_name])
            note = ""

            # Add tier-specific notes that explain why a row count is partial.
            if tier_name in {"tier2", "tier4"} and not frame.empty:
                repeat_note = _summarize_repeat_histogram(frame)
                scale_note = _summarize_scale_rows(frame)
                note = "; ".join(part for part in [f"repeat_hist={repeat_note}" if repeat_note else "", f"scale_rows={scale_note}" if scale_note else ""] if part)
            elif tier_name == "predictor_profile" and not frame.empty and "node_count" in frame.columns:
                note = "node_counts=" + ",".join(str(int(value)) for value in sorted(frame["node_count"].astype(int).tolist()))

            records.append(
                {
                    "cluster_type": cluster_type,
                    "tier": tier_name,
                    "tier_display_name": TIER_DISPLAY_NAMES[tier_name],
                    "artifact_path": str(path),
                    "exists": bool(path.exists()),
                    "actual_rows": actual_rows,
                    "expected_rows": expected,
                    "coverage_pct": float(actual_rows / expected * 100.0) if expected > 0 else float("nan"),
                    "status": _status_from_counts(actual_rows, expected),
                    "note": note,
                }
            )

    return pd.DataFrame(records)


def validate_artifact_audit(audit_df: pd.DataFrame, mode: str, partial_clusters: Sequence[str]) -> None:
    """Validate artifact completeness under strict or partial-report policy."""

    incomplete = audit_df[audit_df["status"] != "complete"].copy()
    if incomplete.empty:
        return

    if str(mode) == "strict":
        details = [
            f"{row.cluster_type}::{row.tier}={row.status} ({row.actual_rows}/{row.expected_rows})"
            for row in incomplete.itertuples(index=False)
        ]
        raise ValueError("strict mode found incomplete artifacts: " + "; ".join(details))

    allowed_clusters = {str(cluster) for cluster in partial_clusters}
    unexpected = incomplete[~incomplete["cluster_type"].isin(allowed_clusters)]
    if not unexpected.empty:
        details = [
            f"{row.cluster_type}::{row.tier}={row.status} ({row.actual_rows}/{row.expected_rows})"
            for row in unexpected.itertuples(index=False)
        ]
        raise ValueError("partial mode found unexpected incomplete clusters: " + "; ".join(details))


def _resolve_report_paths(artifact_dir: Path, report_dir: Path) -> Dict[str, Path]:
    """Resolve run-specific report, latest-copy, figure, and CSV paths."""

    run_tag = artifact_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures" / run_tag
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "report_dir": report_dir,
        "figure_dir": figure_dir,
        "report_path": report_dir / f"{run_tag}_well_show.md",
        "latest_report_path": report_dir / "latest_report.md",
        "audit_csv_path": report_dir / "artifact_audit.csv",
        "tier1_summary_path": report_dir / "tier1_highlights.csv",
        "tier2_summary_path": report_dir / "tier2_representative_summary.csv",
        "tier4_summary_path": report_dir / "tier4_representative_summary.csv",
        "manifest_path": report_dir / "report_manifest.json",
    }


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without optional dependencies.

    Avoiding `tabulate` keeps the report builder lightweight. The generated
    table is deterministic and sufficient for reviewer-facing summaries.
    """

    if frame.empty:
        return "_No rows available._"

    def _format_cell(value: Any) -> str:
        # Treat missing values and NaN as empty Markdown cells.
        if pd.isna(value):
            return ""

        # Keep booleans lower-case so CSV and Markdown representations match.
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"

        # Use compact numeric formatting without hiding meaningful precision.
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if math.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
                return str(int(round(numeric)))
            if abs(numeric) >= 100.0:
                return f"{numeric:.1f}"
            if abs(numeric) >= 1.0:
                return f"{numeric:.3f}"
            return f"{numeric:.4f}"

        # Escape pipes so arbitrary string values remain valid Markdown cells.
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns.tolist()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows: List[str] = [header, separator]

    for _, row in frame.iterrows():
        rendered = [_format_cell(row[column]) for column in frame.columns]
        rows.append("| " + " | ".join(rendered) + " |")

    return "\n".join(rows)


def _percentile_or_nan(series: pd.Series, percentile: float) -> float:
    """Return a percentile, or NaN for an empty numeric series."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean.to_numpy(dtype=float), percentile))


def _mean_or_nan(series: pd.Series) -> float:
    """Return a mean, or NaN for an empty numeric series."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def _mean_percent_or_nan(series: pd.Series) -> float:
    """Return the mean of a 0..1 ratio as a percentage, or NaN."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean() * 100.0)


def _concat_cluster_frames(artifact_dir: Path, spec: BenchmarkSpec, tier_name: str) -> pd.DataFrame:
    """Load and concatenate all cluster CSVs for one tier."""

    frames: List[pd.DataFrame] = []
    for cluster_type in spec.cluster_types:
        path = _resolve_artifact_paths(artifact_dir, cluster_type)[tier_name]
        frame = _safe_read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _select_diagnostic_algorithm(spec: BenchmarkSpec) -> str:
    """Select the algorithm used for adaptive diagnostics.

    Public comparisons follow `spec.public_algorithms`, while usage, breakdown,
    and adaptive-specific summaries prefer the main adaptive algorithm displayed
    as `BandPilot`.
    """

    preferred = "BandPilot"
    public_algorithms = [str(value) for value in spec.public_algorithms]
    all_algorithms = [str(value) for value in spec.algorithms]
    if preferred in public_algorithms or preferred in all_algorithms:
        return preferred
    for candidate in public_algorithms + all_algorithms + list(ALGORITHM_ORDER):
        normalized = ALGORITHM_DISPLAY_ALIASES.get(str(candidate), str(candidate))
        if normalized in ALGORITHM_ORDER:
            return str(normalized)
    return preferred


def _determine_report_scope(audit_df: pd.DataFrame) -> str:
    """Return `full` only when every audited artifact is complete."""

    if audit_df.empty:
        return "partial"
    if bool((audit_df["status"] == "complete").all()):
        return "full"
    return "partial"


def _build_tier1_highlights(
    artifact_dir: Path,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> pd.DataFrame:
    """Build the Tier 1 measured summary for reviewer-facing diagnostics.

    The summary uses representative k values that are also present in the Tier 1
    measured sweep.
    """

    tier1_df = _concat_cluster_frames(artifact_dir, spec, "tier1")
    if tier1_df.empty:
        return pd.DataFrame()

    tier1_k_set = {int(value) for value in spec.tier1_k_values}
    highlight_ks = [
        int(value) for value in spec.representative_k_values if int(value) in tier1_k_set
    ]
    if not highlight_ks:
        highlight_ks = [min(tier1_k_set)]

    subset = tier1_df[
        (tier1_df["algorithm"] == diagnostic_algorithm)
        & (tier1_df["k"].astype(int).isin(highlight_ks))
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    summary = (
        subset.groupby(["cluster_type", "contention_mode", "k"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            latency_p50_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 50)),
            latency_p95_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 95)),
            predictor_time_mean_s=("predictor_time_s", _mean_or_nan),
            non_predictor_mean_s=("non_predictor_search_time_s", _mean_or_nan),
            hu_pts_usage_rate_pct=("hu_pts_usage_rate", _mean_percent_or_nan),
        )
        .sort_values(["cluster_type", "contention_mode", "k"])
        .reset_index(drop=True)
    )
    summary.insert(1, "algorithm", diagnostic_algorithm)
    summary.insert(3, "evidence", "measured")
    return summary


def _filter_representative_context(df: pd.DataFrame, spec: BenchmarkSpec) -> pd.DataFrame:
    """Filter a dataframe to the configured representative context."""

    if df.empty:
        return df.copy()
    subset = df.copy()
    if "contention_mode" in subset.columns:
        subset = subset[subset["contention_mode"] == spec.representative_contention_mode]
    if "avail_ratio" in subset.columns:
        subset = subset[np.isclose(subset["avail_ratio"].astype(float), spec.representative_avail_ratio, atol=1e-8)]
    if "inter_pod_factor" in subset.columns:
        subset = subset[np.isclose(subset["inter_pod_factor"].astype(float), spec.representative_inter_pod_factor, atol=1e-8)]
    if "k" in subset.columns:
        subset = subset[subset["k"].astype(int).isin(spec.representative_k_values)]
    return subset.copy()


def _exclude_exact_fit_boundary_points(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact-fit `total_gpu == k` rows from reviewer-facing display.

    Raw benchmark artifacts and completeness audit keep those rows. The display
    view removes them so scaled-search figures emphasize the non-trivial scale-up
    regime.
    """

    if df.empty or "total_gpu" not in df.columns or "k" not in df.columns:
        return df.copy()

    subset = df.copy()
    total_gpu = pd.to_numeric(subset["total_gpu"], errors="coerce")
    request_k = pd.to_numeric(subset["k"], errors="coerce")
    exact_fit_mask = total_gpu.notna() & request_k.notna() & np.isclose(
        total_gpu.to_numpy(dtype=float),
        request_k.to_numpy(dtype=float),
        atol=1e-8,
    )
    return subset.loc[~exact_fit_mask].copy()


def _filter_representative_display_context(df: pd.DataFrame, spec: BenchmarkSpec) -> pd.DataFrame:
    """Filter to representative display rows.

    The raw protocol keeps all feasible `k <= target_avail` cases. The display
    protocol additionally removes exact-fit rows so plots focus on scale-up.
    """

    subset = _filter_representative_context(df, spec)
    if subset.empty:
        return subset.copy()
    return _exclude_exact_fit_boundary_points(subset)


def _filter_representative_slice(df: pd.DataFrame, spec: BenchmarkSpec, primary_algorithm: str) -> pd.DataFrame:
    """Filter to the representative slice for one primary algorithm."""

    subset = _filter_representative_context(df, spec)
    if subset.empty:
        return subset.copy()
    return subset[subset["algorithm"] == primary_algorithm].copy()


def _filter_representative_display_slice(
    df: pd.DataFrame,
    spec: BenchmarkSpec,
    primary_algorithm: str,
) -> pd.DataFrame:
    """Filter to representative display rows for one primary algorithm."""

    subset = _filter_representative_display_context(df, spec)
    if subset.empty:
        return subset.copy()
    return subset[subset["algorithm"] == primary_algorithm].copy()


def _build_tier2_representative_summary(
    artifact_dir: Path,
    spec: BenchmarkSpec,
    audit_df: pd.DataFrame,
    diagnostic_algorithm: str,
) -> pd.DataFrame:
    """Build the representative Tier 2 scaled-trace summary."""

    tier2_df = _concat_cluster_frames(artifact_dir, spec, "tier2")
    subset = _filter_representative_display_slice(tier2_df, spec, diagnostic_algorithm)
    if subset.empty:
        return pd.DataFrame()

    summary = (
        subset.groupby(["cluster_type", "total_gpu", "k"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            latency_p50_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 50)),
            latency_p95_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 95)),
            predictor_calls_p50=("predictor_calls", lambda series: _percentile_or_nan(series, 50)),
            predictor_time_p50_s=("predictor_time_s", lambda series: _percentile_or_nan(series, 50)),
            non_predictor_p50_s=("non_predictor_search_time_s", lambda series: _percentile_or_nan(series, 50)),
            hu_pts_usage_rate_pct=("hu_pts_usage_rate", _mean_percent_or_nan),
        )
        .sort_values(["cluster_type", "total_gpu", "k"])
        .reset_index(drop=True)
    )

    tier2_status = audit_df[audit_df["tier"] == "tier2"][["cluster_type", "status", "coverage_pct"]].rename(
        columns={"status": "tier2_status", "coverage_pct": "tier2_coverage_pct"}
    )
    summary = summary.merge(tier2_status, on="cluster_type", how="left")
    summary.insert(1, "algorithm", diagnostic_algorithm)
    summary.insert(3, "evidence", "simulated")
    return summary


def _build_tier4_representative_summary(
    artifact_dir: Path,
    spec: BenchmarkSpec,
    audit_df: pd.DataFrame,
    diagnostic_algorithm: str,
) -> pd.DataFrame:
    """Build the representative Tier 4 synthesized-latency summary."""

    tier4_df = _concat_cluster_frames(artifact_dir, spec, "tier4")
    subset = _filter_representative_display_slice(tier4_df, spec, diagnostic_algorithm)
    if subset.empty:
        return pd.DataFrame()

    predictor_component = (
        pd.to_numeric(subset.get("scaled_predictor_calls", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        * pd.to_numeric(subset.get("predictor_e2e_p50_ms", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        / 1000.0
    )
    subset = subset.copy()
    subset["predictor_component_p50_s"] = predictor_component

    summary = (
        subset.groupby(["cluster_type", "total_gpu", "k"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            latency_p50_s=("synthesized_wall_time_p50_s", lambda series: _percentile_or_nan(series, 50)),
            latency_p95_s=("synthesized_wall_time_p95_s", lambda series: _percentile_or_nan(series, 95)),
            predictor_calls_p50=("scaled_predictor_calls", lambda series: _percentile_or_nan(series, 50)),
            predictor_time_p50_s=("predictor_component_p50_s", lambda series: _percentile_or_nan(series, 50)),
            non_predictor_p50_s=("scaled_non_predictor_search_time_s", lambda series: _percentile_or_nan(series, 50)),
            hu_pts_usage_rate_pct=("hu_pts_usage_rate", _mean_percent_or_nan),
        )
        .sort_values(["cluster_type", "total_gpu", "k"])
        .reset_index(drop=True)
    )

    tier4_status = audit_df[audit_df["tier"] == "tier4"][["cluster_type", "status", "coverage_pct"]].rename(
        columns={"status": "tier4_status", "coverage_pct": "tier4_coverage_pct"}
    )
    summary = summary.merge(tier4_status, on="cluster_type", how="left")
    summary.insert(1, "algorithm", diagnostic_algorithm)
    summary.insert(3, "evidence", "synthesized")
    return summary


def _build_tier1_algorithm_snapshot(artifact_dir: Path, spec: BenchmarkSpec) -> pd.DataFrame:
    """Build compact cross-algorithm Tier 1 measured snapshots."""

    tier1_df = _concat_cluster_frames(artifact_dir, spec, "tier1")
    if tier1_df.empty:
        return pd.DataFrame()

    small_k = 8 if 8 in set(int(value) for value in spec.tier1_k_values) else min(int(value) for value in spec.tier1_k_values)
    large_k = max(int(value) for value in spec.tier1_k_values)

    # Include small-k rows across modes plus the largest representative common-mode row.
    small_subset = tier1_df[tier1_df["k"].astype(int) == int(small_k)].copy()
    large_subset = tier1_df[
        (tier1_df["contention_mode"] == spec.representative_contention_mode)
        & (tier1_df["k"].astype(int) == int(large_k))
    ].copy()
    subset = pd.concat([small_subset, large_subset], ignore_index=True, sort=False)
    if subset.empty:
        return pd.DataFrame()

    summary = (
        subset.groupby(["cluster_type", "algorithm", "contention_mode", "k"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            latency_p50_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 50)),
            latency_p95_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 95)),
            predictor_time_mean_s=("predictor_time_s", _mean_or_nan),
            non_predictor_mean_s=("non_predictor_search_time_s", _mean_or_nan),
            hu_pts_usage_rate_pct=("hu_pts_usage_rate", _mean_percent_or_nan),
        )
        .reset_index(drop=True)
    )
    summary.insert(4, "evidence", "measured")
    summary["_contention_rank"] = summary["contention_mode"].map(
        {mode: idx for idx, mode in enumerate(spec.tier1_contention_modes)}
    )
    summary["_algorithm_rank"] = summary["algorithm"].map(
        {algorithm: idx for idx, algorithm in enumerate(ALGORITHM_ORDER)}
    )
    summary = summary.sort_values(
        ["cluster_type", "k", "_contention_rank", "_algorithm_rank"]
    ).drop(columns=["_contention_rank", "_algorithm_rank"])
    return summary.reset_index(drop=True)


def _build_tier2_algorithm_snapshot(
    artifact_dir: Path,
    spec: BenchmarkSpec,
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build cross-algorithm Tier 2 representative snapshots."""

    tier2_df = _concat_cluster_frames(artifact_dir, spec, "tier2")
    subset = _filter_representative_display_context(tier2_df, spec)
    if subset.empty:
        return pd.DataFrame()

    preferred_scales = [128, 512, 1024]
    available_scales = {int(value) for value in subset["total_gpu"].astype(int).unique().tolist()}
    selected_scales = [scale for scale in preferred_scales if scale in available_scales]
    if not selected_scales:
        selected_scales = sorted(available_scales)[: min(3, len(available_scales))]

    subset = subset[subset["total_gpu"].astype(int).isin(selected_scales)].copy()
    summary = (
        subset.groupby(["cluster_type", "algorithm", "total_gpu", "k"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            latency_p50_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 50)),
            latency_p95_s=("measured_wall_time_s", lambda series: _percentile_or_nan(series, 95)),
            predictor_calls_p50=("predictor_calls", lambda series: _percentile_or_nan(series, 50)),
            predictor_time_p50_s=("predictor_time_s", lambda series: _percentile_or_nan(series, 50)),
            non_predictor_p50_s=("non_predictor_search_time_s", lambda series: _percentile_or_nan(series, 50)),
            hu_pts_usage_rate_pct=("hu_pts_usage_rate", _mean_percent_or_nan),
        )
        .reset_index(drop=True)
    )

    status_view = audit_df[audit_df["tier"] == "tier2"][["cluster_type", "status", "coverage_pct"]].rename(
        columns={"status": "tier2_status", "coverage_pct": "tier2_coverage_pct"}
    )
    summary = summary.merge(status_view, on="cluster_type", how="left")
    summary.insert(4, "evidence", "simulated")
    summary["_algorithm_rank"] = summary["algorithm"].map(
        {algorithm: idx for idx, algorithm in enumerate(ALGORITHM_ORDER)}
    )
    summary = summary.sort_values(
        ["cluster_type", "total_gpu", "k", "_algorithm_rank"]
    ).drop(columns=["_algorithm_rank"])
    return summary.reset_index(drop=True)


def _build_tier4_algorithm_snapshot(
    artifact_dir: Path,
    spec: BenchmarkSpec,
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build cross-algorithm Tier 4 representative snapshots."""

    tier4_df = _concat_cluster_frames(artifact_dir, spec, "tier4")
    subset = _filter_representative_display_context(tier4_df, spec)
    if subset.empty:
        return pd.DataFrame()

    preferred_scales = [512, 1024, 4096]
    available_scales = {int(value) for value in subset["total_gpu"].astype(int).unique().tolist()}
    selected_scales = [scale for scale in preferred_scales if scale in available_scales]
    if not selected_scales:
        selected_scales = sorted(available_scales)[: min(3, len(available_scales))]

    subset = subset[subset["total_gpu"].astype(int).isin(selected_scales)].copy()
    subset = subset.copy()
    subset["predictor_component_p50_s"] = (
        pd.to_numeric(subset.get("scaled_predictor_calls", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        * pd.to_numeric(subset.get("predictor_e2e_p50_ms", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        / 1000.0
    )

    summary = (
        subset.groupby(["cluster_type", "algorithm", "total_gpu", "k"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            latency_p50_s=("synthesized_wall_time_p50_s", lambda series: _percentile_or_nan(series, 50)),
            latency_p95_s=("synthesized_wall_time_p95_s", lambda series: _percentile_or_nan(series, 95)),
            predictor_calls_p50=("scaled_predictor_calls", lambda series: _percentile_or_nan(series, 50)),
            predictor_time_p50_s=("predictor_component_p50_s", lambda series: _percentile_or_nan(series, 50)),
            non_predictor_p50_s=("scaled_non_predictor_search_time_s", lambda series: _percentile_or_nan(series, 50)),
            hu_pts_usage_rate_pct=("hu_pts_usage_rate", _mean_percent_or_nan),
        )
        .reset_index(drop=True)
    )

    status_view = audit_df[audit_df["tier"] == "tier4"][["cluster_type", "status", "coverage_pct"]].rename(
        columns={"status": "tier4_status", "coverage_pct": "tier4_coverage_pct"}
    )
    summary = summary.merge(status_view, on="cluster_type", how="left")
    summary.insert(4, "evidence", "synthesized")
    summary["_algorithm_rank"] = summary["algorithm"].map(
        {algorithm: idx for idx, algorithm in enumerate(ALGORITHM_ORDER)}
    )
    summary = summary.sort_values(
        ["cluster_type", "total_gpu", "k", "_algorithm_rank"]
    ).drop(columns=["_algorithm_rank"])
    return summary.reset_index(drop=True)


def _extract_log_window(log_path: Path) -> Dict[str, str]:
    """Extract benchmark log path and first/last timestamp."""

    if not log_path.exists():
        return {"log_path": str(log_path), "start_time": "", "end_time": ""}

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"log_path": str(log_path), "start_time": "", "end_time": ""}

    def _extract_timestamp(line: str) -> str:
        return line.split(" | ", 1)[0].strip()

    return {
        "log_path": str(log_path),
        "start_time": _extract_timestamp(lines[0]),
        "end_time": _extract_timestamp(lines[-1]),
    }


def _find_preferred_cluster(spec: BenchmarkSpec, keyword: str) -> str:
    """Find the first configured cluster whose name contains a keyword."""

    for cluster_type in spec.cluster_types:
        if keyword.lower() in str(cluster_type).lower():
            return str(cluster_type)
    return str(spec.cluster_types[0]) if spec.cluster_types else ""


def _lookup_summary_row(frame: pd.DataFrame, **filters: Any) -> Optional[pd.Series]:
    """Look up one summary row by filters, returning None when absent."""

    if frame.empty:
        return None
    subset = frame.copy()
    for column, value in filters.items():
        if column not in subset.columns:
            return None
        if isinstance(value, float):
            subset = subset[np.isclose(subset[column].astype(float), value, atol=1e-8)]
        else:
            subset = subset[subset[column] == value]
    if subset.empty:
        return None
    return subset.iloc[0]


def _describe_algorithm_latencies(frame: pd.DataFrame, **filters: Any) -> str:
    """Describe p50/p95 latency values for all public algorithms."""

    descriptions: List[str] = []
    for algorithm in ALGORITHM_ORDER:
        row = _lookup_summary_row(frame, algorithm=algorithm, **filters)
        if row is None:
            continue
        descriptions.append(
            f"`{algorithm}` p50/p95={_format_seconds(row['latency_p50_s'])}/{_format_seconds(row['latency_p95_s'])}"
        )
    return ",".join(descriptions)


def _describe_usage_regime(usage_min: float, usage_max: float) -> str:
    """Describe the PTS usage regime for the representative slice."""

    if math.isnan(float(usage_min)) or math.isnan(float(usage_max)):
        return "the raw summary does not contain usage-rate information."
    if float(usage_max) <= 30.0:
        return "most representative cases stay EHA-like, with PTS used only for tail cases."
    if float(usage_min) >= 70.0:
        return "most representative cases use the PTS backend, so adaptive savings are limited."
    return "the representative slice is mixed: some cases skip PTS and some use the PTS backend."


def _subset_frame_by_cluster(frame: pd.DataFrame, cluster_type: str) -> pd.DataFrame:
    """Return one cluster's snapshot rows, or an empty DataFrame."""

    if frame.empty or "cluster_type" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["cluster_type"] == cluster_type].reset_index(drop=True)


def _build_experiment_design_lines(spec: BenchmarkSpec) -> List[str]:
    """Describe benchmark design, evidence tiers, and sweep settings."""

    expected_rows = _build_expected_row_map(spec)
    _, tier2_rows_by_scale = _build_tier2_expected_repeat_map(spec)
    tier4_reference_map = _build_tier4_reference_scale_map(spec)
    tier2_scale_text = ", ".join(
        f"{int(scale)}:{int(rows)}"
        for scale, rows in sorted(tier2_rows_by_scale.items())
    )
    tier4_scale_text = ", ".join(
        f"{target}<-{reference}"
        for target, reference in sorted(tier4_reference_map.items())
    )

    return [
        f"- Scope: `cluster_type={_format_sequence_literal(spec.cluster_types)}`, `algorithm={_format_sequence_literal(spec.algorithms)}`, Tier 1 `total_gpu={spec.cluster_total_gpu}`.",
        f"- Tier 1 (`measured`): `k={_format_sequence_literal(spec.tier1_k_values)}`, `contention_mode={_format_sequence_literal(spec.tier1_contention_modes)}`, `{spec.tier1_repeat_num}` repeats per `(cluster, algorithm, mode, k)`; expected rows per cluster are `{expected_rows['tier1']}`.",
        f"- Tier 2 (`simulated`): `total_gpu={_format_sequence_literal(spec.tier2_gpu_counts)}`, `k={_format_sequence_literal(spec.tier2_k_values)}`, `avail_ratio={_format_sequence_literal(spec.tier2_avail_ratios)}`, `contention_mode={_format_sequence_literal(spec.tier2_contention_modes)}`, `inter_pod_factor={_format_sequence_literal(spec.tier2_inter_pod_factors)}`, `{spec.tier2_repeat_num}` repeats per scenario.",
        f"- Tier 2 representative public group: `mode={spec.representative_contention_mode}`, `avail={spec.representative_avail_ratio:.1f}`, `factor={spec.representative_inter_pod_factor:.1f}`, `k={_format_sequence_literal(spec.representative_k_values)}`, `{spec.tier2_public_repeat_num}` repeats, scale set `{tier2_scale_text}`.",
        "- Tier 2 raw benchmark enforces `target_avail = max(max(k_values), round(total_gpu x avail_ratio))` and includes only feasible `k <= target_avail` cases.",
        "- Reviewer-facing representative display keeps the raw protocol but excludes exact-fit `total_gpu == k` cases from the scaled-search scale-up view.",
        f"- Predictor profile: `node_count={_format_sequence_literal(spec.predictor_node_counts)}`, `{spec.predictor_inference_repeats}` microbenchmark repeats, expected rows `{expected_rows['predictor_profile']}`.",
        f"- Tier 4 (`synthesized`): `total_gpu={_format_sequence_literal(spec.tier4_target_gpu_counts)}`; bounds combine Tier 2 scale traces and predictor microbenchmarks over scale set `{tier4_scale_text}`.",
        f"- Public algorithms for Tier 2 / Tier 4 are `{_format_sequence_literal(spec.public_algorithms)}`; BandPilot diagnostics remain available for audit.",
    ]


def _build_metric_definition_lines() -> List[str]:
    """Describe latency metrics and predictor/non-predictor decomposition."""

    return [
        "- `measured_wall_time_s`: end-to-end dispatch wall-clock latency.",
        "- `predictor_time_s`: time spent in bandwidth or contention predictor calls.",
        "- `non_predictor_search_time_s = measured_wall_time_s - predictor_time_s`: EHA search, PTS refine, adaptive trigger or bank lookup, and other scheduling bookkeeping.",
        "- Predictor/non-predictor stacked bars report component-level p50 values.",
        "- Tier 4 bounds combine scaled trace counts with predictor microbenchmarks and remain `synthesized`, not measured deployment results.",
    ]


def _build_completeness_interpretation_lines(audit_df: pd.DataFrame, spec: BenchmarkSpec) -> List[str]:
    """Explain whether the artifact set is a full or partial run."""

    lines: List[str] = []
    scope = _determine_report_scope(audit_df)
    per_cluster = (
        audit_df.groupby("cluster_type", as_index=False)["status"]
        .apply(lambda series: int((series == "complete").sum()))
        .rename(columns={"status": "complete_tier_count"})
    )
    if scope == "full":
        cluster_text = ",".join(
            f"`{row.cluster_type}` = `{int(row.complete_tier_count)}/{len(TIER_ORDER)}`"
            for row in per_cluster.itertuples(index=False)
        )
        lines.append(
            f"- Artifact completeness: every cluster has all `4/4` tiers complete ({cluster_text}); this is a full reviewer-facing readout."
        )
        lines.append(
            "- Interpretation should still keep measured, simulated, and synthesized evidence tiers separate."
        )
        return lines

    h100_cluster = _find_preferred_cluster(spec, "h100")
    incomplete_rows = audit_df[audit_df["status"] != "complete"].copy()
    if not incomplete_rows.empty:
        row = incomplete_rows.iloc[0]
        repeat_hist = _extract_repeat_hist_from_note(str(row.get("note", "")))
        repeat_text = f", repeat histogram `{repeat_hist}`" if repeat_hist else ""
        lines.append(
            f"- Artifact completeness: `{h100_cluster}` is complete, while `{row['cluster_type']}` has `{row['tier_display_name']}` marked `{row['status']}` (`{int(row['actual_rows'])}/{int(row['expected_rows'])}`, `{float(row['coverage_pct']):.1f}%`){repeat_text}."
        )
        lines.append(
            "- Treat this as a partial or provisional readout until the incomplete tier is regenerated."
        )

    return lines


def _build_tier1_interpretation_lines(
    tier1_snapshot: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> List[str]:
    """Build interpretation bullets for 32-GPU measured latency."""

    if tier1_snapshot.empty:
        return ["- Tier 1 raw CSV is missing, so no 32-GPU measured interpretation is available."]

    lines: List[str] = []
    small_k = 8 if 8 in set(int(value) for value in spec.tier1_k_values) else min(int(value) for value in spec.tier1_k_values)
    large_k = max(int(value) for value in spec.tier1_k_values)
    h100_cluster = _find_preferred_cluster(spec, "h100")
    het_cluster = next((cluster for cluster in spec.cluster_types if cluster != h100_cluster), h100_cluster)

    h100_small = _describe_algorithm_latencies(
        tier1_snapshot,
        cluster_type=h100_cluster,
        contention_mode=spec.representative_contention_mode,
        k=small_k,
    )
    h100_large = _describe_algorithm_latencies(
        tier1_snapshot,
        cluster_type=h100_cluster,
        contention_mode=spec.representative_contention_mode,
        k=large_k,
    )
    het_small = _describe_algorithm_latencies(
        tier1_snapshot,
        cluster_type=het_cluster,
        contention_mode=spec.representative_contention_mode,
        k=small_k,
    )
    het_large = _describe_algorithm_latencies(
        tier1_snapshot,
        cluster_type=het_cluster,
        contention_mode=spec.representative_contention_mode,
        k=large_k,
    )
    representative_measured_ks = [
        int(value) for value in spec.representative_k_values if int(value) in set(int(v) for v in spec.tier1_k_values)
    ]
    representative_text = _format_sequence_literal(representative_measured_ks) if representative_measured_ks else "{}"

    lines.append(
        f"- Tier 1 measured sweep uses `k={_format_sequence_literal(spec.tier1_k_values)}`, `mode={_format_sequence_literal(spec.tier1_contention_modes)}`, and `{spec.tier1_repeat_num}` repeats. Diagnostic snapshots use `k={small_k}` and `k={large_k}`; the public measured slice uses `k={representative_text}`."
    )
    if h100_small and h100_large:
        lines.append(
            f"- `{h100_cluster}` measured common-mode snapshots: `k={small_k}` -> {h100_small}; `k={large_k}` -> {h100_large}."
        )
    if het_small and het_large:
        lines.append(
            f"- `{het_cluster}` measured common-mode snapshots: `k={small_k}` -> {het_small}; `k={large_k}` -> {het_large}."
        )

    small_row = _lookup_summary_row(
        tier1_snapshot,
        cluster_type=h100_cluster,
        algorithm=diagnostic_algorithm,
        contention_mode=spec.representative_contention_mode,
        k=small_k,
    )
    large_row = _lookup_summary_row(
        tier1_snapshot,
        cluster_type=h100_cluster,
        algorithm=diagnostic_algorithm,
        contention_mode=spec.representative_contention_mode,
        k=large_k,
    )
    if small_row is not None and large_row is not None:
        lines.append(
            f"- For `{diagnostic_algorithm}` on `{h100_cluster}`, `common, k={small_k}` decomposes into `predictor={_format_seconds(small_row['predictor_time_mean_s'])}` and `non-predictor={_format_seconds(small_row['non_predictor_mean_s'])}`; `k={large_k}` decomposes into `predictor={_format_seconds(large_row['predictor_time_mean_s'])}` and `non-predictor={_format_seconds(large_row['non_predictor_mean_s'])}`."
        )

    return lines


def _build_h100_scaled_interpretation_lines(
    tier2_snapshot: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> List[str]:
    """Build interpretation bullets for the H100 representative Tier 2 slice."""

    if tier2_snapshot.empty:
        return ["- H100 representative Tier 2 raw CSV is missing, so scaled-trace interpretation is unavailable."]

    lines: List[str] = []
    h100_cluster = _find_preferred_cluster(spec, "h100")
    k_small = min(int(value) for value in spec.representative_k_values)
    k_large = max(int(value) for value in spec.representative_k_values)

    h100_frame = tier2_snapshot[tier2_snapshot["cluster_type"] == h100_cluster].copy()
    if h100_frame.empty:
        return ["- H100 representative Tier 2 snapshot is empty."]

    small_1024 = _describe_algorithm_latencies(h100_frame, cluster_type=h100_cluster, total_gpu=1024, k=k_small)
    large_128 = _describe_algorithm_latencies(h100_frame, cluster_type=h100_cluster, total_gpu=128, k=k_large)
    large_512 = _describe_algorithm_latencies(h100_frame, cluster_type=h100_cluster, total_gpu=512, k=k_large)
    large_1024 = _describe_algorithm_latencies(h100_frame, cluster_type=h100_cluster, total_gpu=1024, k=k_large)

    lines.append(
        "- The H100 representative slice isolates scale-up behavior for `EHA`, `PTS`, and `BandPilot`."
    )
    lines.append(
        f"- The representative display excludes the exact-fit `64 GPU, k={k_large}` case from the scaled-search view; `128+ GPU` cases represent the scaled-search regime."
    )
    if small_1024:
        adaptive_1024_small = _lookup_summary_row(
            h100_frame,
            cluster_type=h100_cluster,
            algorithm=diagnostic_algorithm,
            total_gpu=1024,
            k=k_small,
        )
        if adaptive_1024_small is not None:
            lines.append(
                f"- For `k={k_small}` at `1024 GPU`, {small_1024}. `{diagnostic_algorithm}` reports `p50={_format_seconds(adaptive_1024_small['latency_p50_s'])}`."
            )
    if large_128 and large_512 and large_1024:
        lines.append(
            f"- For `k={k_large}`, `128 GPU` -> {large_128}; `512 GPU` -> {large_512}; `1024 GPU` -> {large_1024}."
        )

    adaptive_large = _lookup_summary_row(
        h100_frame,
        cluster_type=h100_cluster,
        algorithm=diagnostic_algorithm,
        total_gpu=1024,
        k=k_large,
    )
    hu_pts_large = _lookup_summary_row(
        h100_frame,
        cluster_type=h100_cluster,
        algorithm="PTS",
        total_gpu=1024,
        k=k_large,
    )
    if adaptive_large is not None and hu_pts_large is not None:
        lines.append(
            f"- At `1024 GPU, k={k_large}`, `{diagnostic_algorithm}` and `PTS` differ by `{_format_seconds(abs(float(adaptive_large['latency_p50_s']) - float(hu_pts_large['latency_p50_s'])) )}` at p50."
        )

    lines.append(
        "- `EHA` is a latency reference only; quality claims must use the dispatch-quality metrics from the regenerated evaluation CSVs."
    )
    return lines


def _build_h100_synth_interpretation_lines(
    tier4_snapshot: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> List[str]:
    """Build interpretation bullets for H100 representative Tier 4 bounds."""

    if tier4_snapshot.empty:
        return ["- Tier 4 raw CSV is missing, so H100 synthesized-bound interpretation is unavailable."]

    lines: List[str] = []
    h100_cluster = _find_preferred_cluster(spec, "h100")
    k_small = min(int(value) for value in spec.representative_k_values)
    k_large = max(int(value) for value in spec.representative_k_values)
    h100_frame = tier4_snapshot[tier4_snapshot["cluster_type"] == h100_cluster].copy()
    if h100_frame.empty:
        return ["- H100 representative Tier 4 snapshot is empty."]

    synth_small_4096 = _lookup_summary_row(
        h100_frame,
        cluster_type=h100_cluster,
        algorithm=diagnostic_algorithm,
        total_gpu=4096,
        k=k_small,
    )
    synth_large_512 = _describe_algorithm_latencies(h100_frame, cluster_type=h100_cluster, total_gpu=512, k=k_large)
    synth_large_1024 = _describe_algorithm_latencies(h100_frame, cluster_type=h100_cluster, total_gpu=1024, k=k_large)
    synth_large_4096 = _describe_algorithm_latencies(h100_frame, cluster_type=h100_cluster, total_gpu=4096, k=k_large)

    lines.append(
        "- Tier 4 is a synthesized control-plane bound and should not be described as measured deployment latency."
    )
    if synth_small_4096 is not None:
        lines.append(
            f"- For `k={k_small}` at `4096 GPU`, `{diagnostic_algorithm}` has `p50={_format_seconds(synth_small_4096['latency_p50_s'])}`."
        )
    if synth_large_512 and synth_large_1024 and synth_large_4096:
        lines.append(
            f"- For `k={k_large}`, synthesized bounds are: `512 GPU` -> {synth_large_512}; `1024 GPU` -> {synth_large_1024}; `4096 GPU` -> {synth_large_4096}."
        )

    synth_large_primary = _lookup_summary_row(
        h100_frame,
        cluster_type=h100_cluster,
        algorithm=diagnostic_algorithm,
        total_gpu=4096,
        k=k_large,
    )
    if synth_large_primary is not None:
        lines.append(
            f"- `{diagnostic_algorithm}` at `4096 GPU, k={k_large}` has p50 `{_format_seconds(synth_large_primary['latency_p50_s'])}`, split into `predictor={_format_seconds(synth_large_primary['predictor_time_p50_s'])}` and `non-predictor={_format_seconds(synth_large_primary['non_predictor_p50_s'])}`."
        )

    return lines


def _build_het_scaled_interpretation_lines(
    tier2_snapshot: pd.DataFrame,
    audit_df: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> List[str]:
    """Build interpretation bullets for the Het-4Mix representative Tier 2 slice."""

    if tier2_snapshot.empty:
        return ["- Het-4Mix representative Tier 2 raw CSV is missing."]

    lines: List[str] = []
    h100_cluster = _find_preferred_cluster(spec, "h100")
    het_cluster = next((cluster for cluster in spec.cluster_types if cluster != h100_cluster), h100_cluster)
    het_frame = tier2_snapshot[tier2_snapshot["cluster_type"] == het_cluster].copy()
    if het_frame.empty:
        return ["- Het-4Mix representative Tier 2 snapshot is empty."]

    k_small = min(int(value) for value in spec.representative_k_values)
    k_large = max(int(value) for value in spec.representative_k_values)
    audit_row = _lookup_summary_row(audit_df, cluster_type=het_cluster, tier="tier2")
    sample_row = _lookup_summary_row(
        het_frame,
        cluster_type=het_cluster,
        algorithm=diagnostic_algorithm,
        total_gpu=1024,
        k=k_large,
    )

    if audit_row is not None:
        repeat_hist = _extract_repeat_hist_from_note(str(audit_row.get("note", "")))
        if str(audit_row["status"]) == "complete":
            lines.append(
                f"- Completeness: `{het_cluster}` Tier 2 has `{int(audit_row['actual_rows'])}/{int(audit_row['expected_rows'])}` rows (`{float(audit_row['coverage_pct']):.1f}%`), so it can be used as a full readout."
            )
        else:
            repeat_text = f", representative point has `{int(sample_row['sample_count'])}` repeats" if sample_row is not None else ""
            hist_text = f", Tier 2 repeat histogram `{repeat_hist}`" if repeat_hist else ""
            lines.append(
                f"- Completeness: `{het_cluster}` Tier 2 has `{int(audit_row['actual_rows'])}/{int(audit_row['expected_rows'])}` rows (`{float(audit_row['coverage_pct']):.1f}%`){repeat_text}{hist_text}."
            )

    lines.append(
        f"- As with H100, the representative display excludes the exact-fit `64 GPU, k={k_large}` case from the scaled-search view."
    )

    small_1024 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=1024, k=k_small)
    large_128 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=128, k=k_large)
    large_512 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=512, k=k_large)
    large_1024 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=1024, k=k_large)

    if small_1024:
        lines.append(
            f"- For `k={k_small}` at `1024 GPU`, {small_1024}."
        )
    if large_128 and large_512 and large_1024:
        lines.append(
            f"- For `k={k_large}`, `128 GPU` -> {large_128}; `512 GPU` -> {large_512}; `1024 GPU` -> {large_1024}."
        )
    return lines


def _build_het_synth_interpretation_lines(
    tier4_snapshot: pd.DataFrame,
    audit_df: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> List[str]:
    """Build interpretation bullets for Het-4Mix representative Tier 4 bounds."""

    if tier4_snapshot.empty:
        return ["- Het-4Mix representative Tier 4 raw CSV is missing."]

    lines: List[str] = []
    h100_cluster = _find_preferred_cluster(spec, "h100")
    het_cluster = next((cluster for cluster in spec.cluster_types if cluster != h100_cluster), h100_cluster)
    het_frame = tier4_snapshot[tier4_snapshot["cluster_type"] == het_cluster].copy()
    if het_frame.empty:
        return ["- Het-4Mix representative Tier 4 snapshot is empty."]

    k_small = min(int(value) for value in spec.representative_k_values)
    k_large = max(int(value) for value in spec.representative_k_values)
    audit_row = _lookup_summary_row(audit_df, cluster_type=het_cluster, tier="tier4")
    synth_small_4096 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=4096, k=k_small)
    synth_large_512 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=512, k=k_large)
    synth_large_1024 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=1024, k=k_large)
    synth_large_4096 = _describe_algorithm_latencies(het_frame, cluster_type=het_cluster, total_gpu=4096, k=k_large)
    synth_large_primary = _lookup_summary_row(
        het_frame,
        cluster_type=het_cluster,
        algorithm=diagnostic_algorithm,
        total_gpu=4096,
        k=k_large,
    )

    if audit_row is not None:
        lines.append(
            f"- `Het-4Mix` Tier 4 completeness is `{int(audit_row['actual_rows'])}/{int(audit_row['expected_rows'])}` rows (`{float(audit_row['coverage_pct']):.1f}%`)."
        )
    if synth_small_4096:
        lines.append(
            f"- For `k={k_small}` at `4096 GPU`, synthesized split is {synth_small_4096}."
        )
    if synth_large_512 and synth_large_1024 and synth_large_4096:
        lines.append(
            f"- For `k={k_large}`, `512 GPU` -> {synth_large_512}; `1024 GPU` -> {synth_large_1024}; `4096 GPU` -> {synth_large_4096}."
        )
    if synth_large_primary is not None:
        lines.append(
            f"- `{diagnostic_algorithm}` at `4096 GPU, k={k_large}` has p50 `{_format_seconds(synth_large_primary['latency_p50_s'])}`, `predictor={_format_seconds(synth_large_primary['predictor_time_p50_s'])}`, `non-predictor={_format_seconds(synth_large_primary['non_predictor_p50_s'])}`, and p95 `{_format_seconds(synth_large_primary['latency_p95_s'])}`."
        )

    return lines


def _build_usage_breakdown_interpretation_lines(
    tier2_summary: pd.DataFrame,
    tier4_summary: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> List[str]:
    """Build interpretation bullets for PTS usage and latency breakdown."""

    if tier2_summary.empty:
        return ["- Representative summary is missing, so usage and breakdown interpretation is unavailable."]

    lines: List[str] = []
    h100_cluster = _find_preferred_cluster(spec, "h100")
    h100_tier2 = tier2_summary[tier2_summary["cluster_type"] == h100_cluster].copy()
    h100_tier4 = tier4_summary[tier4_summary["cluster_type"] == h100_cluster].copy()
    if h100_tier2.empty:
        return ["- H100 representative summary is empty."]

    usage_min = float(h100_tier2["hu_pts_usage_rate_pct"].min())
    usage_max = float(h100_tier2["hu_pts_usage_rate_pct"].max())
    k_small = min(int(value) for value in spec.representative_k_values)
    k_large = max(int(value) for value in spec.representative_k_values)

    tier2_large_128 = _lookup_summary_row(h100_tier2, cluster_type=h100_cluster, total_gpu=128, k=k_large)
    tier2_large_1024 = _lookup_summary_row(h100_tier2, cluster_type=h100_cluster, total_gpu=1024, k=k_large)
    tier4_large_4096 = _lookup_summary_row(h100_tier4, cluster_type=h100_cluster, total_gpu=4096, k=k_large)
    tier4_small_4096 = _lookup_summary_row(h100_tier4, cluster_type=h100_cluster, total_gpu=4096, k=k_small)

    lines.append(
        f"- Usage diagnostics exclude exact-fit `64 GPU, k={k_large}` from the reviewer-facing representative view."
    )
    lines.append(
        f"- In the H100 representative slice, `{diagnostic_algorithm}` has PTS usage from `{usage_min:.1f}%` to `{usage_max:.1f}%`; {_describe_usage_regime(usage_min, usage_max)}"
    )
    lines.append(
        "- `predictor` is predictor-call time; `non-predictor` is `measured_wall_time_s - predictor_time_s` and includes EHA, PTS, adaptive, and bookkeeping work."
    )
    if tier2_large_128 is not None and tier2_large_1024 is not None:
        lines.append(
            f"- Tier 2 `k={k_large}` p50 predictor/non-predictor split: `128 GPU` `{_format_seconds(tier2_large_128['predictor_time_p50_s'])}/{_format_seconds(tier2_large_128['non_predictor_p50_s'])}`, `1024 GPU` `{_format_seconds(tier2_large_1024['predictor_time_p50_s'])}/{_format_seconds(tier2_large_1024['non_predictor_p50_s'])}`."
        )
    if tier4_small_4096 is not None and tier4_large_4096 is not None:
        lines.append(
            f"- Tier 4 `4096 GPU` p50 split: `k={k_small}` predictor/non-predictor `{_format_seconds(tier4_small_4096['predictor_time_p50_s'])}/{_format_seconds(tier4_small_4096['non_predictor_p50_s'])}`; `k={k_large}` `{_format_seconds(tier4_large_4096['predictor_time_p50_s'])}/{_format_seconds(tier4_large_4096['non_predictor_p50_s'])}`."
        )
    lines.append(
        "- Large-scale latency remains dominated by orchestration and search work unless the regenerated breakdown shows otherwise."
    )
    return lines


def _build_key_findings(
    audit_df: pd.DataFrame,
    tier1_summary: pd.DataFrame,
    tier2_summary: pd.DataFrame,
    tier4_summary: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
) -> List[str]:
    """Build the report's key-finding bullets."""

    findings: List[str] = []
    scope = _determine_report_scope(audit_df)
    if scope == "full":
        findings.append(
            "Both `H100_26H100_27H100_28H100_29` and `Het-4Mix` have complete `Tier1/Tier2/Predictor/Tier4` artifacts, so the report is a full reviewer-facing readout."
        )
    else:
        tier2_audit = audit_df[audit_df["tier"] == "tier2"].copy()
        incomplete_rows = tier2_audit[tier2_audit["status"] != "complete"]
        if not incomplete_rows.empty:
            row = incomplete_rows.iloc[0]
            findings.append(
                f"`{row['cluster_type']}` has incomplete Tier 2 coverage "
                f"(`{int(row['actual_rows'])}/{int(row['expected_rows'])}` rows, "
                f"`{float(row['coverage_pct']):.1f}%`); treat that cluster as partial/provisional."
            )

    h100_cluster = _find_preferred_cluster(spec, "h100")
    het_cluster = next((cluster for cluster in spec.cluster_types if cluster != h100_cluster), h100_cluster)
    measured_k = next(
        (
            int(value)
            for value in spec.representative_k_values
            if int(value) in set(pd.to_numeric(tier1_summary.get("k", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).tolist())
        ),
        min(int(value) for value in spec.tier1_k_values),
    )

    h100_common_k8 = _lookup_summary_row(
        tier1_summary,
        cluster_type=h100_cluster,
        contention_mode="common",
        k=measured_k,
    )
    het_common_k8 = _lookup_summary_row(
        tier1_summary,
        cluster_type=het_cluster,
        contention_mode="common",
        k=measured_k,
    )
    if h100_common_k8 is not None and het_common_k8 is not None:
        findings.append(
            f"In the `32-GPU measured` `common, k={measured_k}` case, `{diagnostic_algorithm}` p50 latency is "
            f"`{float(h100_common_k8['latency_p50_s']):.3f}s` on `{h100_cluster}` and "
            f"`{float(het_common_k8['latency_p50_s']):.3f}s` on `{het_cluster}`."
        )

    h100_1024_k64 = _lookup_summary_row(tier2_summary, cluster_type=h100_cluster, total_gpu=1024, k=64)
    h100_4096_k64 = _lookup_summary_row(tier4_summary, cluster_type=h100_cluster, total_gpu=4096, k=64)
    if h100_1024_k64 is not None and h100_4096_k64 is not None:
        findings.append(
            f"On `{h100_cluster}`, BandPilot representative `k=64` latency remains tail-heavy: "
            f"`1024 GPU p50/p95={float(h100_1024_k64['latency_p50_s']):.3f}/{float(h100_1024_k64['latency_p95_s']):.3f}s`, "
            f"`4096 GPU synthesized p50/p95={float(h100_4096_k64['latency_p50_s']):.3f}/{float(h100_4096_k64['latency_p95_s']):.3f}s`."
        )

    het_1024_k64 = _lookup_summary_row(tier2_summary, cluster_type=het_cluster, total_gpu=1024, k=64)
    het_4096_k64 = _lookup_summary_row(tier4_summary, cluster_type=het_cluster, total_gpu=4096, k=64)
    if het_1024_k64 is not None and het_4096_k64 is not None:
        findings.append(
            f"On `{het_cluster}`, BandPilot representative `1024 GPU, k=64` latency is "
            f"`p50/p95={float(het_1024_k64['latency_p50_s']):.3f}/{float(het_1024_k64['latency_p95_s']):.3f}s`, "
            f"and the `4096 GPU` synthesized bound is "
            f"`{float(het_4096_k64['latency_p50_s']):.3f}/{float(het_4096_k64['latency_p95_s']):.3f}s`."
        )

    h100_usage_rows = tier2_summary[tier2_summary["cluster_type"] == h100_cluster]
    het_usage_rows = tier2_summary[tier2_summary["cluster_type"] == het_cluster]
    if not h100_usage_rows.empty and not het_usage_rows.empty:
        h100_usage_mean = float(h100_usage_rows["hu_pts_usage_rate_pct"].mean())
        het_usage_mean = float(het_usage_rows["hu_pts_usage_rate_pct"].mean())
        findings.append(
            f"In the representative Tier 2 slice, BandPilot PTS usage averages "
            f"`H100 {h100_usage_mean:.1f}% / Het-4Mix {het_usage_mean:.1f}%`; use the breakdown plots to interpret whether adaptive skipping reduces median latency."
        )

    return findings


def _save_completeness_matrix(audit_df: pd.DataFrame, figure_path: Path) -> Optional[Path]:
    """Save Figure 1: cluster-by-tier completeness heatmap."""

    if audit_df.empty:
        return None

    clusters = list(dict.fromkeys(audit_df["cluster_type"].tolist()))
    matrix = np.zeros((len(clusters), len(TIER_ORDER)), dtype=float)
    annotations: List[List[str]] = [["" for _ in TIER_ORDER] for _ in clusters]

    for row in audit_df.itertuples(index=False):
        cluster_idx = clusters.index(str(row.cluster_type))
        tier_idx = TIER_ORDER.index(str(row.tier))
        matrix[cluster_idx, tier_idx] = STATUS_TO_SCORE[str(row.status)]
        annotations[cluster_idx][tier_idx] = (
            f"{STATUS_TO_LABEL[str(row.status)]}\n{int(row.actual_rows)}/{int(row.expected_rows)}"
        )

    fig, ax = plt.subplots(figsize=(10.0, 3.8 + 0.65 * len(clusters)))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(TIER_ORDER)))
    ax.set_xticklabels([TIER_SHORT_NAMES[tier] for tier in TIER_ORDER], fontsize=10)
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_yticklabels(clusters, fontsize=10)
    ax.set_title("Artifact completeness audit")

    for row_idx, cluster in enumerate(clusters):
        for col_idx, tier in enumerate(TIER_ORDER):
            ax.text(col_idx, row_idx, annotations[row_idx][col_idx], ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Completeness score")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_tier1_measured_latency_plot(artifact_dir: Path, spec: BenchmarkSpec, figure_path: Path) -> Optional[Path]:
    """Save Figure 2: per-cluster 32-GPU measured latency."""

    tier1_df = _concat_cluster_frames(artifact_dir, spec, "tier1")
    if tier1_df.empty:
        return None

    contention_modes = list(spec.tier1_contention_modes)
    clusters = list(dict.fromkeys(tier1_df["cluster_type"].tolist()))
    fig, axes = plt.subplots(
        len(clusters),
        len(contention_modes),
        figsize=(5.1 * len(contention_modes), 3.8 * max(1, len(clusters))),
        squeeze=False,
        sharey="row",
    )

    for row_idx, cluster_type in enumerate(clusters):
        for col_idx, contention_mode in enumerate(contention_modes):
            ax = axes[row_idx][col_idx]
            subset = tier1_df[
                (tier1_df["cluster_type"] == cluster_type)
                & (tier1_df["contention_mode"] == contention_mode)
            ]
            for algorithm in ALGORITHM_ORDER:
                algo_subset = subset[subset["algorithm"] == algorithm]
                if algo_subset.empty:
                    continue
                stats = (
                    algo_subset.groupby("k", as_index=False)["measured_wall_time_s"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                ax.errorbar(
                    stats["k"],
                    stats["mean"],
                    yerr=stats["std"].fillna(0.0),
                    marker=ALGORITHM_MARKERS[algorithm],
                    linestyle=ALGORITHM_LINESTYLES[algorithm],
                    linewidth=1.8,
                    color=ALGORITHM_COLORS[algorithm],
                    capsize=3,
                    label=algorithm,
                )
            ax.set_title(f"{cluster_type} | {contention_mode}")
            ax.set_xlabel("Requested GPUs (k)")
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("Measured latency (s)")

    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_representative_latency_plot(
    frame: pd.DataFrame,
    cluster_type: str,
    figure_path: Path,
    title_prefix: str,
    latency_p50_column: str,
    latency_p95_column: str,
) -> Optional[Path]:
    """Save a representative latency plot shared by Tier 2 and Tier 4."""

    subset = frame[frame["cluster_type"] == cluster_type].copy()
    if subset.empty:
        return None

    k_values = sorted(int(value) for value in subset["k"].astype(int).unique().tolist())
    fig, axes = plt.subplots(1, len(k_values), figsize=(6.2 * max(1, len(k_values)), 4.7), squeeze=False, sharey=True)

    for col_idx, k in enumerate(k_values):
        ax = axes[0][col_idx]
        k_subset = subset[subset["k"].astype(int) == int(k)].copy()
        for algorithm in ALGORITHM_ORDER:
            algo_subset = k_subset[k_subset["algorithm"] == algorithm]
            if algo_subset.empty:
                continue
            grouped = (
                algo_subset.groupby("total_gpu", as_index=False)
                .agg(
                    latency_p50=(latency_p50_column, lambda series: _percentile_or_nan(series, 50)),
                    latency_p95=(latency_p95_column, lambda series: _percentile_or_nan(series, 95)),
                )
                .sort_values("total_gpu")
            )
            ax.plot(
                grouped["total_gpu"],
                grouped["latency_p50"],
                marker=ALGORITHM_MARKERS[algorithm],
                linestyle=ALGORITHM_LINESTYLES[algorithm],
                linewidth=1.9,
                color=ALGORITHM_COLORS[algorithm],
                label=f"{algorithm} p50",
            )
            ax.plot(
                grouped["total_gpu"],
                grouped["latency_p95"],
                linestyle=":",
                linewidth=1.4,
                color=ALGORITHM_COLORS[algorithm],
                label=f"{algorithm} p95",
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{title_prefix} | k={k}")
        ax.set_xlabel("Total GPUs")
        ax.grid(True, which="both", alpha=0.25)
        if col_idx == 0:
            ax.set_ylabel("Latency (s)")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_h100_usage_plot(
    tier2_df: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
    cluster_type: str,
    figure_path: Path,
) -> Optional[Path]:
    """Figure 6: representative Tier 2 slice PTS usage rate."""

    subset = _filter_representative_display_slice(tier2_df, spec, diagnostic_algorithm)
    subset = subset[subset["cluster_type"] == cluster_type].copy()
    if subset.empty or "hu_pts_usage_rate" not in subset.columns:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for k in sorted(int(value) for value in subset["k"].astype(int).unique().tolist()):
        k_subset = subset[subset["k"].astype(int) == int(k)]
        grouped = (
            k_subset.groupby("total_gpu", as_index=False)["hu_pts_usage_rate"]
            .mean()
            .sort_values("total_gpu")
        )
        ax.plot(
            grouped["total_gpu"],
            grouped["hu_pts_usage_rate"] * 100.0,
            marker="o",
            linewidth=1.9,
            label=f"k={k}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total GPUs")
    ax.set_ylabel("PTS usage rate (%)")
    ax.set_ylim(0.0, 105.0)
    ax.set_title(f"{cluster_type} representative PTS usage")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_h100_breakdown_plot(
    tier2_df: pd.DataFrame,
    spec: BenchmarkSpec,
    diagnostic_algorithm: str,
    cluster_type: str,
    figure_path: Path,
) -> Optional[Path]:
    """Save Figure 7: predictor versus non-predictor latency breakdown."""

    subset = _filter_representative_display_slice(tier2_df, spec, diagnostic_algorithm)
    subset = subset[subset["cluster_type"] == cluster_type].copy()
    if subset.empty:
        return None

    k_values = sorted(int(value) for value in subset["k"].astype(int).unique().tolist())
    fig, axes = plt.subplots(1, len(k_values), figsize=(6.2 * max(1, len(k_values)), 4.4), squeeze=False, sharey=True)

    for col_idx, k in enumerate(k_values):
        ax = axes[0][col_idx]
        k_subset = subset[subset["k"].astype(int) == int(k)]
        grouped = (
            k_subset.groupby("total_gpu", as_index=False)
            .agg(
                predictor_p50=("predictor_time_s", lambda series: _percentile_or_nan(series, 50)),
                non_predictor_p50=("non_predictor_search_time_s", lambda series: _percentile_or_nan(series, 50)),
            )
            .sort_values("total_gpu")
        )
        x_labels = [str(int(value)) for value in grouped["total_gpu"].tolist()]
        x_positions = np.arange(len(x_labels))
        predictor = grouped["predictor_p50"].to_numpy(dtype=float)
        non_predictor = grouped["non_predictor_p50"].to_numpy(dtype=float)

        ax.bar(x_positions, non_predictor, label="Non-predictor", color="#4c78a8")
        ax.bar(x_positions, predictor, bottom=non_predictor, label="Predictor", color="#f58518")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{cluster_type} | k={k}")
        ax.set_xlabel("Total GPUs")
        ax.grid(True, axis="y", alpha=0.25)
        if col_idx == 0:
            ax.set_ylabel("Latency p50 (s)")

    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _save_figures(
    artifact_dir: Path,
    spec: BenchmarkSpec,
    audit_df: pd.DataFrame,
    report_paths: Mapping[str, Path],
    diagnostic_algorithm: str,
) -> Dict[str, str]:
    """Generate all report figures and return Markdown-relative paths."""

    figure_dir = report_paths["figure_dir"]
    relative_paths: Dict[str, str] = {}

    completeness_path = _save_completeness_matrix(audit_df, figure_dir / "artifact_completeness_matrix.png")
    if completeness_path is not None:
        relative_paths["artifact_completeness_matrix"] = str(Path("figures") / figure_dir.name / completeness_path.name)

    tier1_path = _save_tier1_measured_latency_plot(artifact_dir, spec, figure_dir / "tier1_measured_latency.png")
    if tier1_path is not None:
        relative_paths["tier1_measured_latency"] = str(Path("figures") / figure_dir.name / tier1_path.name)

    tier2_df = _concat_cluster_frames(artifact_dir, spec, "tier2")
    tier4_df = _concat_cluster_frames(artifact_dir, spec, "tier4")
    # Representative display excludes exact-fit `total_gpu == k` rows.
    tier2_representative_df = _filter_representative_display_context(tier2_df, spec)
    tier4_representative_df = _filter_representative_display_context(tier4_df, spec)

    h100_cluster = _find_preferred_cluster(spec, "h100")
    het_cluster = next((cluster for cluster in spec.cluster_types if cluster != h100_cluster), h100_cluster)
    het_tier2_audit = _lookup_summary_row(audit_df, cluster_type=het_cluster, tier="tier2")
    het_tier4_audit = _lookup_summary_row(audit_df, cluster_type=het_cluster, tier="tier4")

    h100_tier2_path = _save_representative_latency_plot(
        frame=tier2_representative_df,
        cluster_type=h100_cluster,
        figure_path=figure_dir / "h100_scaled_latency_representative.png",
        title_prefix=f"{h100_cluster} representative scaled trace",
        latency_p50_column="measured_wall_time_s",
        latency_p95_column="measured_wall_time_s",
    )
    if h100_tier2_path is not None:
        relative_paths["h100_scaled_latency_representative"] = str(Path("figures") / figure_dir.name / h100_tier2_path.name)

    h100_tier4_path = _save_representative_latency_plot(
        frame=tier4_representative_df,
        cluster_type=h100_cluster,
        figure_path=figure_dir / "h100_synthesized_latency_representative.png",
        title_prefix=f"{h100_cluster} representative synthesized bound",
        latency_p50_column="synthesized_wall_time_p50_s",
        latency_p95_column="synthesized_wall_time_p95_s",
    )
    if h100_tier4_path is not None:
        relative_paths["h100_synthesized_latency_representative"] = str(Path("figures") / figure_dir.name / h100_tier4_path.name)

    het_tier2_title = (
        f"{het_cluster} representative scaled trace"
        if het_tier2_audit is None or str(het_tier2_audit["status"]) == "complete"
        else f"PARTIAL {het_cluster} representative scaled trace"
    )
    het_tier2_path = _save_representative_latency_plot(
        frame=tier2_representative_df,
        cluster_type=het_cluster,
        figure_path=figure_dir / "het4mix_scaled_latency_representative.png",
        title_prefix=het_tier2_title,
        latency_p50_column="measured_wall_time_s",
        latency_p95_column="measured_wall_time_s",
    )
    if het_tier2_path is not None:
        relative_paths["het4mix_scaled_latency_representative"] = str(Path("figures") / figure_dir.name / het_tier2_path.name)

    het_tier4_title = (
        f"{het_cluster} representative synthesized bound"
        if het_tier4_audit is None or str(het_tier4_audit["status"]) == "complete"
        else f"PARTIAL {het_cluster} representative synthesized bound"
    )
    het_tier4_path = _save_representative_latency_plot(
        frame=tier4_representative_df,
        cluster_type=het_cluster,
        figure_path=figure_dir / "het4mix_synthesized_latency_representative.png",
        title_prefix=het_tier4_title,
        latency_p50_column="synthesized_wall_time_p50_s",
        latency_p95_column="synthesized_wall_time_p95_s",
    )
    if het_tier4_path is not None:
        relative_paths["het4mix_synthesized_latency_representative"] = str(Path("figures") / figure_dir.name / het_tier4_path.name)

    usage_path = _save_h100_usage_plot(
        tier2_df=tier2_df,
        spec=spec,
        diagnostic_algorithm=diagnostic_algorithm,
        cluster_type=h100_cluster,
        figure_path=figure_dir / "h100_hu_pts_usage_rate.png",
    )
    if usage_path is not None:
        relative_paths["h100_hu_pts_usage_rate"] = str(Path("figures") / figure_dir.name / usage_path.name)

    breakdown_path = _save_h100_breakdown_plot(
        tier2_df=tier2_df,
        spec=spec,
        diagnostic_algorithm=diagnostic_algorithm,
        cluster_type=h100_cluster,
        figure_path=figure_dir / "h100_latency_breakdown.png",
    )
    if breakdown_path is not None:
        relative_paths["h100_latency_breakdown"] = str(Path("figures") / figure_dir.name / breakdown_path.name)

    return relative_paths


def _build_report_text(
    *,
    artifact_dir: Path,
    spec: BenchmarkSpec,
    benchmark_config: Path,
    mode: str,
    partial_clusters: Sequence[str],
    audit_df: pd.DataFrame,
    tier1_summary: pd.DataFrame,
    tier2_summary: pd.DataFrame,
    tier4_summary: pd.DataFrame,
    tier1_snapshot: pd.DataFrame,
    tier2_snapshot: pd.DataFrame,
    tier4_snapshot: pd.DataFrame,
    figure_paths: Mapping[str, str],
    report_paths: Mapping[str, Path],
    log_window: Mapping[str, str],
    key_findings: Sequence[str],
) -> str:
    """Build the full Markdown report text."""

    diagnostic_algorithm = _select_diagnostic_algorithm(spec)
    comparison_algorithms = [str(value) for value in (spec.public_algorithms or spec.algorithms)]
    report_scope = _determine_report_scope(audit_df)
    audit_columns = [
        "cluster_type",
        "tier_display_name",
        "status",
        "actual_rows",
        "expected_rows",
        "coverage_pct",
        "note",
    ]
    audit_view = audit_df[audit_columns].copy()
    audit_view = audit_view.rename(
        columns={
            "cluster_type": "Cluster",
            "tier_display_name": "Tier",
            "status": "Status",
            "actual_rows": "Actual Rows",
            "expected_rows": "Expected Rows",
            "coverage_pct": "Coverage (%)",
            "note": "Note",
        }
    )

    title = (
        "# Scalability Full Well-Show Report"
        if report_scope == "full"
        else "# Scalability Partial Well-Show Report"
    )
    h100_cluster = _find_preferred_cluster(spec, "h100")
    het_cluster = next((cluster for cluster in spec.cluster_types if cluster != h100_cluster), h100_cluster)
    het_tier2_status = _lookup_summary_row(audit_df, cluster_type=het_cluster, tier="tier2")
    het_tier4_status = _lookup_summary_row(audit_df, cluster_type=het_cluster, tier="tier4")
    het_tier2_heading = (
        f"#### {het_cluster}"
        if het_tier2_status is None or str(het_tier2_status["status"]) == "complete"
        else f"#### {het_cluster} (Partial)"
    )
    het_tier4_heading = (
        f"#### {het_cluster}"
        if het_tier4_status is None or str(het_tier4_status["status"]) == "complete"
        else f"#### {het_cluster} (Partial)"
    )

    lines: List[str] = [
        title,
        "",
        "## 1. Run Status",
        "",
        f"- Artifact directory: `{artifact_dir}`",
        f"- Benchmark config: `{benchmark_config}`",
        f"- Report scope: `{report_scope}`",
        f"- Builder mode: `{mode}`",
        f"- Allowed partial clusters: `{', '.join(partial_clusters) if partial_clusters else 'None'}`",
        f"- Log window: `{log_window.get('start_time', '')}` -> `{log_window.get('end_time', '')}`",
        f"- Comparison algorithms: `{_format_sequence_literal(comparison_algorithms)}`",
        f"- BandPilot diagnostics algorithm: `{diagnostic_algorithm}`",
        "",
        _frame_to_markdown(audit_view),
        "",
        "## 2. Experiment Design",
        "",
    ]

    for item in _build_experiment_design_lines(spec):
        lines.append(item)

    lines.extend(["", "## 3. Metric Definition", ""])
    for item in _build_metric_definition_lines():
        lines.append(item)

    lines.extend(["", "## 4. Executive Summary", ""])
    if key_findings:
        for finding in key_findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- No complete artifact summary is available; inspect the audit table above.")

    lines.extend(
        [
            "",
            "## 5. Evidence Boundary",
            "",
            "- `Tier 1` is `measured`: it uses the 32-GPU benchmark traces.",
            "- `Tier 2` is `simulated`: it uses scaled predictors and scaled managers.",
            "- `Tier 4` is `synthesized`: it combines scaled traces with predictor microbenchmarks to bound control-plane latency.",
            (
                "- Both `H100_26H100_27H100_28H100_29` and `Het-4Mix` are complete in this full artifact package."
                if report_scope == "full"
                else "- Some clusters are incomplete; those entries should be treated as partial or provisional."
            ),
            "",
            "## 6. Figure-By-Figure Interpretation",
            "",
            "### 6.1 Artifact Completeness Audit",
            "",
        ]
    )

    if "artifact_completeness_matrix" in figure_paths:
        lines.extend([f"![Artifact completeness matrix]({figure_paths['artifact_completeness_matrix']})", ""])
    for item in _build_completeness_interpretation_lines(audit_df, spec):
        lines.append(item)
    lines.append("")

    lines.extend(["### 6.2 32-GPU Measured Latency", ""])
    if "tier1_measured_latency" in figure_paths:
        lines.extend([f"![Tier1 measured latency]({figure_paths['tier1_measured_latency']})", ""])
    for item in _build_tier1_interpretation_lines(tier1_snapshot, spec, diagnostic_algorithm):
        lines.append(item)
    lines.append("")

    lines.extend(["### 6.3 H100 Full Representative Scaling", ""])
    if "h100_scaled_latency_representative" in figure_paths:
        lines.extend([f"![H100 scaled latency representative]({figure_paths['h100_scaled_latency_representative']})", ""])
    for item in _build_h100_scaled_interpretation_lines(tier2_snapshot, spec, diagnostic_algorithm):
        lines.append(item)
    lines.append("")

    lines.extend(["### 6.4 H100 Synthesized Control-Plane Bound", ""])
    if "h100_synthesized_latency_representative" in figure_paths:
        lines.extend([f"![H100 synthesized latency representative]({figure_paths['h100_synthesized_latency_representative']})", ""])
    for item in _build_h100_synth_interpretation_lines(tier4_snapshot, spec, diagnostic_algorithm):
        lines.append(item)
    lines.append("")

    lines.extend(
        [
            "### 6.5 Het-4Mix Representative Scaling"
            if report_scope == "full"
            else "### 6.5 Het-4Mix Partial Representative Scaling",
            "",
        ]
    )
    if "het4mix_scaled_latency_representative" in figure_paths:
        lines.extend([f"![Het4Mix scaled latency representative]({figure_paths['het4mix_scaled_latency_representative']})", ""])
    for item in _build_het_scaled_interpretation_lines(tier2_snapshot, audit_df, spec, diagnostic_algorithm):
        lines.append(item)
    lines.append("")

    lines.extend(["### 6.6 Het-4Mix Synthesized Control-Plane Bound", ""])
    if "het4mix_synthesized_latency_representative" in figure_paths:
        lines.extend([f"![Het4Mix synthesized latency representative]({figure_paths['het4mix_synthesized_latency_representative']})", ""])
    for item in _build_het_synth_interpretation_lines(tier4_snapshot, audit_df, spec, diagnostic_algorithm):
        lines.append(item)
    lines.append("")

    lines.extend(["### 6.7 H100 Usage And Breakdown", ""])
    if "h100_hu_pts_usage_rate" in figure_paths:
        lines.extend([f"![H100 PTS usage rate]({figure_paths['h100_hu_pts_usage_rate']})", ""])
    if "h100_latency_breakdown" in figure_paths:
        lines.extend([f"![H100 latency breakdown]({figure_paths['h100_latency_breakdown']})", ""])
    for item in _build_usage_breakdown_interpretation_lines(tier2_summary, tier4_summary, spec, diagnostic_algorithm):
        lines.append(item)
    lines.append("")

    lines.extend(["## 7. Selected Tables", ""])
    lines.extend(["### 7.1 Tier 1 Cross-Algorithm Diagnostic Snapshots", "", _frame_to_markdown(tier1_snapshot), ""])
    lines.extend(["### 7.2 Tier 2 Representative Snapshots", ""])
    lines.extend(
        [
            f"#### {h100_cluster}",
            "",
            _frame_to_markdown(_subset_frame_by_cluster(tier2_snapshot, h100_cluster)),
            "",
            het_tier2_heading,
            "",
            _frame_to_markdown(_subset_frame_by_cluster(tier2_snapshot, het_cluster)),
            "",
        ]
    )
    lines.extend(
        [
            "### 7.3 Tier 4 Representative Snapshots",
            "",
            f"#### {h100_cluster}",
            "",
            _frame_to_markdown(_subset_frame_by_cluster(tier4_snapshot, h100_cluster)),
            "",
            het_tier4_heading,
            "",
            _frame_to_markdown(_subset_frame_by_cluster(tier4_snapshot, het_cluster)),
            "",
            "### 7.4 BandPilot Diagnostic Summaries",
            "",
            "- `BandPilot` diagnostics summarize representative usage and latency breakdowns for the public report.",
            "",
            "#### Tier 1 Highlights",
            "",
            _frame_to_markdown(tier1_summary),
            "",
            "#### Tier 2 Representative Summary",
            "",
            _frame_to_markdown(tier2_summary),
            "",
            "#### Tier 4 Representative Summary",
            "",
            _frame_to_markdown(tier4_summary),
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## 8. Current Limits And Next Action",
            "",
            (
                "- The artifact package is complete for the current control-plane evidence boundary: `Tier 1=measured`, `Tier 2=simulated`, and `Tier 4=synthesized`."
                if report_scope == "full"
                else "- Incomplete clusters should not be used for final scalability claims without rerunning the missing stages."
            ),
            (
                "- For the response letter or manuscript, use `evaluation/scalability/benchmark.py` outputs and `scalability_latency_summary.csv/.tex` as the primary tables."
                if report_scope == "full"
                else "- Until rerun completion, treat benchmark public outputs and this report as partial evidence."
            ),
            (
                "- If a benchmark rerun changes raw artifacts, regenerate the summaries and this report together."
                if report_scope == "full"
                else "- Recommended next action: finish missing artifacts before using this as a reviewer-facing final readout."
            ),
            "",
            "## 9. Artifact Paths",
            "",
            f"- Run-specific report: `{report_paths['report_path']}`",
            f"- Latest report: `{report_paths['latest_report_path']}`",
            f"- Figure directory: `{report_paths['figure_dir']}`",
            f"- Audit CSV: `{report_paths['audit_csv_path']}`",
            f"- Tier 1 summary CSV: `{report_paths['tier1_summary_path']}`",
            f"- Tier 2 summary CSV: `{report_paths['tier2_summary_path']}`",
            f"- Tier 4 summary CSV: `{report_paths['tier4_summary_path']}`",
            f"- Manifest JSON: `{report_paths['manifest_path']}`",
        ]
    )

    return "\n".join(lines)


def build_report(
    *,
    artifact_dir: Path,
    report_dir: Path,
    benchmark_config: Path,
    mode: str,
    partial_clusters: Sequence[str],
) -> Tuple[Path, Path]:
    """Build audit outputs, figures, Markdown report, and latest-report copy."""

    spec = _load_benchmark_spec(benchmark_config)
    audit_df = build_artifact_audit(artifact_dir, spec)
    validate_artifact_audit(audit_df, mode, partial_clusters)

    report_paths = _resolve_report_paths(artifact_dir, report_dir)
    diagnostic_algorithm = _select_diagnostic_algorithm(spec)
    report_scope = _determine_report_scope(audit_df)
    comparison_algorithms = [str(value) for value in (spec.public_algorithms or spec.algorithms)]
    tier1_summary = _build_tier1_highlights(artifact_dir, spec, diagnostic_algorithm)
    tier2_summary = _build_tier2_representative_summary(artifact_dir, spec, audit_df, diagnostic_algorithm)
    tier4_summary = _build_tier4_representative_summary(artifact_dir, spec, audit_df, diagnostic_algorithm)
    tier1_snapshot = _build_tier1_algorithm_snapshot(artifact_dir, spec)
    tier2_snapshot = _build_tier2_algorithm_snapshot(artifact_dir, spec, audit_df)
    tier4_snapshot = _build_tier4_algorithm_snapshot(artifact_dir, spec, audit_df)
    log_window = _extract_log_window(artifact_dir / "search_overhead.log")
    key_findings = _build_key_findings(
        audit_df=audit_df,
        tier1_summary=tier1_summary,
        tier2_summary=tier2_summary,
        tier4_summary=tier4_summary,
        spec=spec,
        diagnostic_algorithm=diagnostic_algorithm,
    )
    figure_paths = _save_figures(
        artifact_dir=artifact_dir,
        spec=spec,
        audit_df=audit_df,
        report_paths=report_paths,
        diagnostic_algorithm=diagnostic_algorithm,
    )

    # Persist traceability CSVs used by the report and response letter.
    audit_df.to_csv(report_paths["audit_csv_path"], index=False)
    tier1_summary.to_csv(report_paths["tier1_summary_path"], index=False)
    tier2_summary.to_csv(report_paths["tier2_summary_path"], index=False)
    tier4_summary.to_csv(report_paths["tier4_summary_path"], index=False)

    report_text = _build_report_text(
        artifact_dir=artifact_dir,
        spec=spec,
        benchmark_config=benchmark_config,
        mode=mode,
        partial_clusters=partial_clusters,
        audit_df=audit_df,
        tier1_summary=tier1_summary,
        tier2_summary=tier2_summary,
        tier4_summary=tier4_summary,
        tier1_snapshot=tier1_snapshot,
        tier2_snapshot=tier2_snapshot,
        tier4_snapshot=tier4_snapshot,
        figure_paths=figure_paths,
        report_paths=report_paths,
        log_window=log_window,
        key_findings=key_findings,
    )
    report_paths["report_path"].write_text(report_text, encoding="utf-8")
    shutil.copyfile(report_paths["report_path"], report_paths["latest_report_path"])

    manifest = {
        "artifact_dir": str(artifact_dir),
        "benchmark_config": str(benchmark_config),
        "report_scope": str(report_scope),
        "mode": str(mode),
        "partial_clusters": [str(cluster) for cluster in partial_clusters],
        "comparison_algorithms": list(comparison_algorithms),
        "diagnostic_algorithm": str(diagnostic_algorithm),
        "primary_algorithm": str(diagnostic_algorithm),
        "log_window": dict(log_window),
        "spec": asdict(spec),
        "display_policy": {
            "representative_display_excludes_exact_fit_boundary": True,
            "representative_display_rule": "exclude rows where total_gpu == k from reviewer-facing representative outputs; raw benchmark artifacts and completeness audit remain unchanged",
        },
        "generated_files": {key: str(path) for key, path in report_paths.items()},
        "figure_paths": dict(figure_paths),
        "audit": audit_df.to_dict(orient="records"),
    }
    report_paths["manifest_path"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return report_paths["report_path"], report_paths["latest_report_path"]


def main() -> None:
    """CLI entry point for strict or partial scalability artifact reports."""

    args = parse_args()
    report_path, latest_report_path = build_report(
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        benchmark_config=args.benchmark_config,
        mode=args.mode,
        partial_clusters=args.partial_clusters,
    )
    print(f"Scalability report saved to {report_path}")
    print(f"Latest report updated at {latest_report_path}")


if __name__ == "__main__":
    main()
