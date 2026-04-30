"""Build Markdown reports for the scalability PTS sidecar.

The report states the simulated evidence boundary and summarizes speedup,
same-combo rate, bandwidth delta, and predictor/non-predictor timing from
regenerated sidecar outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import math

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse report-builder command-line arguments."""

    parser = argparse.ArgumentParser(description="Build PTS-sidecar markdown report")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Summary CSV written by analyze.py")
    parser.add_argument(
        "--breakdown-csv",
        type=Path,
        required=True,
        help="Breakdown CSV written by analyze.py",
    )
    parser.add_argument("--metadata-json", type=Path, required=True, help="Run metadata JSON written by runner.py")
    parser.add_argument("--figure-dir", type=Path, required=True, help="Directory that stores figure PNG/PDF files")
    parser.add_argument("--report-dir", type=Path, required=True, help="Directory for markdown reports")
    return parser.parse_args()


def _is_nan(value: object) -> bool:
    """Return whether a value should be treated as missing."""

    return value is None or (isinstance(value, float) and math.isnan(value))


def _format_pct(value: float) -> str:
    """Format a percentage value."""

    if _is_nan(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _format_ratio_pct(value: float) -> str:
    """Format a ratio stored in the 0..1 range as a percentage."""

    if _is_nan(value):
        return "N/A"
    return f"{float(value) * 100.0:.1f}%"


def _format_x(value: float) -> str:
    """Format a speedup factor."""

    if _is_nan(value):
        return "N/A"
    return f"{float(value):.2f}x"


def _format_seconds(value: float) -> str:
    """Format a duration in seconds."""

    if _is_nan(value):
        return "N/A"
    return f"{float(value):.4f}s"


def _safe_float(value: object) -> Optional[float]:
    """Convert a CSV value to float, returning None for missing values."""

    if _is_nan(value):
        return None
    return float(value)


def _row_value(row: pd.Series, *candidate_columns: str) -> object:
    """Return the first available value from a list of compatible column names."""

    for column in candidate_columns:
        if column in row.index:
            return row[column]
    raise KeyError(f"None of the candidate columns exist: {candidate_columns}")


def _lookup_latency_from_breakdown(
    breakdown_df: pd.DataFrame,
    *,
    cluster_type: str,
    total_gpu: int,
    algorithm: str,
) -> Optional[float]:
    """Look up mean latency for one cluster/scale/algorithm row."""

    matched = breakdown_df[
        (breakdown_df["cluster_type"] == cluster_type)
        & (breakdown_df["total_gpu"].astype(int) == int(total_gpu))
        & (breakdown_df["algorithm"] == algorithm)
    ]
    if matched.empty:
        return None
    return _safe_float(matched.iloc[0]["latency_mean_s"])


def _relative_path(from_dir: Path, to_path: Path) -> str:
    """Build a report-relative Markdown path."""

    return os.path.relpath(to_path, start=from_dir).replace(os.sep, "/")


def _build_result_table(summary_df: pd.DataFrame) -> List[str]:
    """Build the speedup and quality summary table."""

    lines = [
        "| Cluster | Total GPU | Speedup mean | Speedup p50 | Speedup p95 | same_combo_rate | BW delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_df = summary_df.sort_values(["cluster_type", "total_gpu"]).reset_index(drop=True)
    for _, row in ordered_df.iterrows():
        lines.append(
            "| "
            f"{row['cluster_type']} | "
            f"{int(row['total_gpu'])} | "
            f"{_format_x(row['speedup_mean'])} | "
            f"{_format_x(row['speedup_p50'])} | "
            f"{_format_x(row['speedup_p95'])} | "
            f"{_format_ratio_pct(row['same_combo_rate'])} | "
            f"{_format_pct(row['bw_delta_pct_pts_vs_legacy_pts_mean'])} |"
        )
    return lines


def _build_latency_table(breakdown_df: pd.DataFrame) -> List[str]:
    """Build the predictor/non-predictor latency breakdown table."""

    pivot_df = (
        breakdown_df.pivot_table(
            index=["cluster_type", "total_gpu"],
            columns="algorithm",
            values=[
                "latency_mean_s",
                "predictor_time_mean_s",
                "non_predictor_time_mean_s",
            ],
            aggfunc="first",
        )
        .sort_index()
    )

    lines = [
        "| Cluster | Total GPU | legacy-PTS latency | PTS latency | legacy-PTS predictor | PTS predictor | legacy-PTS non-predictor | PTS non-predictor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (cluster_type, total_gpu), row in pivot_df.iterrows():
        lines.append(
            "| "
            f"{cluster_type} | "
            f"{int(total_gpu)} | "
            f"{_format_seconds(row[('latency_mean_s', 'legacy-PTS')])} | "
            f"{_format_seconds(row[('latency_mean_s', 'PTS')])} | "
            f"{_format_seconds(row[('predictor_time_mean_s', 'legacy-PTS')])} | "
            f"{_format_seconds(row[('predictor_time_mean_s', 'PTS')])} | "
            f"{_format_seconds(row[('non_predictor_time_mean_s', 'legacy-PTS')])} | "
            f"{_format_seconds(row[('non_predictor_time_mean_s', 'PTS')])} |"
        )
    return lines


def _cluster_lines(summary_df: pd.DataFrame) -> Iterable[str]:
    """Build per-cluster speedup bullets."""

    for cluster_type, cluster_df in summary_df.groupby("cluster_type", sort=True):
        best_row = cluster_df.loc[cluster_df["speedup_mean"].astype(float).idxmax()]
        worst_row = cluster_df.loc[cluster_df["speedup_mean"].astype(float).idxmin()]
        bw_series = pd.to_numeric(cluster_df["bw_delta_pct_pts_vs_legacy_pts_mean"], errors="coerce")
        max_bw_drift = bw_series.abs().max()
        same_combo_min = cluster_df["same_combo_rate"].astype(float).min()
        yield (
            f"- `{cluster_type}`: PTS speedup over legacy-PTS ranges from "
            f"`{_format_x(worst_row['speedup_mean'])} ~ {_format_x(best_row['speedup_mean'])}`,"
            f" with the best speedup at `total_gpu={int(best_row['total_gpu'])}`."
        )
        drift_text = _format_pct(max_bw_drift)
        yield (
            f"- For this cluster, the minimum `same_combo_rate` is `{_format_ratio_pct(same_combo_min)}` "
            f"and the maximum absolute PTS-vs-legacy-PTS bandwidth delta is `{drift_text}`."
        )


def _build_breakdown_interpretation(breakdown_df: pd.DataFrame) -> List[str]:
    """Build latency-breakdown interpretation bullets."""

    lines: List[str] = []
    pivot_df = (
        breakdown_df.pivot_table(
            index=["cluster_type", "total_gpu"],
            columns="algorithm",
            values=[
                "latency_mean_s",
                "predictor_time_mean_s",
                "non_predictor_time_mean_s",
            ],
            aggfunc="first",
        )
        .sort_index()
    )
    for (cluster_type, total_gpu), row in pivot_df.iterrows():
        pts_total = float(row[("latency_mean_s", "legacy-PTS")])
        hu_total = float(row[("latency_mean_s", "PTS")])
        pts_pred = float(row[("predictor_time_mean_s", "legacy-PTS")])
        hu_pred = float(row[("predictor_time_mean_s", "PTS")])
        pts_non = float(row[("non_predictor_time_mean_s", "legacy-PTS")])
        hu_non = float(row[("non_predictor_time_mean_s", "PTS")])
        lines.append(
            f"- `{cluster_type}, {int(total_gpu)} GPU`: legacy-PTS/PTS total latency "
            f"`{_format_seconds(pts_total)}` / `{_format_seconds(hu_total)}`; "
            f"predictor time `{_format_seconds(pts_pred)}` / `{_format_seconds(hu_pred)}`; "
            f"non-predictor time `{_format_seconds(pts_non)}` / `{_format_seconds(hu_non)}`."
        )
    return lines


def _build_cluster_specific_findings(summary_df: pd.DataFrame) -> List[str]:
    """Build speedup trend bullets per cluster."""

    lines: List[str] = []
    for cluster_type, cluster_df in summary_df.groupby("cluster_type", sort=True):
        cluster_df = cluster_df.sort_values("total_gpu").reset_index(drop=True)
        first_row = cluster_df.iloc[0]
        last_row = cluster_df.iloc[-1]
        lines.append(
            f"- `{cluster_type}` speedup trend: `total_gpu={int(first_row['total_gpu'])}` gives "
            f"`{_format_x(first_row['speedup_mean'])}`, while `total_gpu={int(last_row['total_gpu'])}` gives "
            f"`{_format_x(last_row['speedup_mean'])}`."
        )
    return lines


def build_report(
    summary_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    metadata: Dict[str, object],
    figure_dir: Path,
    report_dir: Path,
) -> Dict[str, Path]:
    """Build the PTS-sidecar Markdown report."""

    report_dir.mkdir(parents=True, exist_ok=True)
    run_tag = str(metadata["run_tag"])
    report_path = report_dir / f"{run_tag}.md"
    latest_path = report_dir / "latest_report.md"

    clusters = ", ".join(str(value) for value in metadata["cluster_types"])
    gpu_counts = ", ".join(str(value) for value in metadata["gpu_counts"])
    gpu_count_values = [int(value) for value in metadata["gpu_counts"]]
    k_value = int(metadata["k_value"])
    repeat_num = int(metadata["repeat_num"])
    contention_mode = str(metadata["contention_mode"])
    avail_ratio = float(metadata["avail_ratio"])
    inter_pod_factor = float(metadata["inter_pod_factor"])
    evidence_type = str(metadata.get("evidence_type", "simulated"))

    best_global = summary_df.loc[summary_df["speedup_mean"].astype(float).idxmax()]
    best_speedup_line = (
        f"`{best_global['cluster_type']}` at `total_gpu={int(best_global['total_gpu'])}` "
        f"has the largest speedup: `{_format_x(best_global['speedup_mean'])}`."
    )
    total_case_count = int(len(summary_df))
    if len(gpu_count_values) == 1:
        scale_scope_text = f"`total_gpu={gpu_count_values[0]}`"
    else:
        scale_scope_text = f"`total_gpu={min(gpu_count_values)} -> {max(gpu_count_values)}`"
    speedup_png_rel = _relative_path(report_path.parent, figure_dir / "pts_speedup_vs_legacy_pts.png")
    latency_bar_png_rel = _relative_path(report_path.parent, figure_dir / "latency_bar_legacy_pts_vs_pts.png")
    breakdown_png_rel = _relative_path(report_path.parent, figure_dir / "latency_breakdown_legacy_pts_vs_pts.png")
    h100_df = summary_df[summary_df["cluster_type"] == "H100_26H100_27H100_28H100_29"]
    h100_extreme_row = None
    if not h100_df.empty:
        h100_extreme_row = h100_df.loc[h100_df["speedup_mean"].astype(float).idxmax()]
    het_df = summary_df[summary_df["cluster_type"] == "Het-4Mix"].sort_values("total_gpu")
    het_same_combo_line = (
        "- `Het-4Mix` `same_combo_rate` is "
        + " / ".join(
            f"`{_format_ratio_pct(row['same_combo_rate'])} @ {int(row['total_gpu'])} GPU`"
            for _, row in het_df.iterrows()
        )
        + "; lower values indicate cases where PTS changes the selected GPU set."
        if not het_df.empty
        else ""
    )
    h100_pts_latency = None
    h100_pts_new_latency = None
    if h100_extreme_row is not None:
        h100_pts_latency = _safe_float(
            _row_value(h100_extreme_row, "legacy-PTS_latency_mean_s", "pts-only_latency_mean_s")
        ) if (
            "legacy-PTS_latency_mean_s" in h100_extreme_row.index
            or "pts-only_latency_mean_s" in h100_extreme_row.index
        ) else _lookup_latency_from_breakdown(
            breakdown_df,
            cluster_type=str(h100_extreme_row["cluster_type"]),
            total_gpu=int(h100_extreme_row["total_gpu"]),
            algorithm="legacy-PTS",
        )
        h100_pts_new_latency = _safe_float(
            _row_value(h100_extreme_row, "PTS_latency_mean_s", "hu-pts-only_latency_mean_s")
        ) if (
            "PTS_latency_mean_s" in h100_extreme_row.index
            or "hu-pts-only_latency_mean_s" in h100_extreme_row.index
        ) else _lookup_latency_from_breakdown(
            breakdown_df,
            cluster_type=str(h100_extreme_row["cluster_type"]),
            total_gpu=int(h100_extreme_row["total_gpu"]),
            algorithm="PTS",
        )

    lines = [
        "# PTS vs legacy-PTS Sidecar",
        "",
        "## Summary",
        "",
        f"- Best observed speedup: {best_speedup_line}",
        f"- Scope: `{total_case_count}` cluster/scale cases; values below `1.0x` mean PTS is slower than legacy-PTS for that slice.",
        "- `same_combo_rate` reports how often both methods select the same GPU set.",
        "",
        "## Protocol",
        "",
        f"- Clusters: `{clusters}`",
        f"- Total GPU counts: `{gpu_counts}`",
        f"- Scenario: `k={k_value}, contention_mode={contention_mode}, avail_ratio={avail_ratio}, inter_pod_factor={inter_pod_factor}`",
        f"- Repeats: `{repeat_num}`",
        "- Algorithms: `legacy-PTS` and `PTS`",
        "",
        "## Evidence Boundary",
        "",
        f"- Evidence type: `{evidence_type}` from a scaled-search sidecar.",
        "- These rows are not deployment measurements; use them only for control-plane scalability analysis.",
        "",
        "## Metric Definitions",
        "",
        "- `Speedup over legacy-PTS = latency(legacy-PTS) / latency(PTS)`.",
        "- `same_combo_rate` is the fraction of cases where both algorithms select the same GPU set.",
        "- `bw_delta_pct_pts_vs_legacy_pts_mean` is PTS bandwidth minus legacy-PTS bandwidth, reported as a percentage of legacy-PTS bandwidth.",
        "- `predictor` and `non-predictor` follow the same decomposition used by the scalability benchmark.",
        "",
        "## Result Table",
        "",
        *_build_result_table(summary_df),
        "",
        "## Cluster Findings",
        "",
        *list(_cluster_lines(summary_df)),
        "",
        "## Figure 1: PTS Speedup Over legacy-PTS",
        "",
        f"![PTS speedup over legacy-PTS]({speedup_png_rel})",
        "",
        f"- The sidecar covers {scale_scope_text} and compares PTS against legacy-PTS.",
        "- Exact-fit cases are not included in this scaled-search slice.",
        *_build_cluster_specific_findings(summary_df),
        f"- Each plotted point aggregates `{repeat_num}` repeats.",
        "",
        "## Figure 2: Total Latency",
        "",
        f"![Total latency comparison: legacy-PTS vs PTS]({latency_bar_png_rel})",
        "",
        f"- Compared scales: `{', '.join(str(v) for v in gpu_count_values)} GPU`.",
        "- Lower latency indicates less control-plane overhead for the same requested allocation size.",
        (
            f"- H100 best-speedup case at `total_gpu={int(h100_extreme_row['total_gpu'])}`: "
            f"legacy-PTS `{_format_seconds(h100_pts_latency)}`, PTS `{_format_seconds(h100_pts_new_latency)}`."
            if h100_extreme_row is not None and h100_pts_latency is not None and h100_pts_new_latency is not None
            else "- H100 best-speedup latency details are unavailable."
        ),
        "",
        "## Figure 3: Latency Breakdown",
        "",
        f"![Latency breakdown of legacy-PTS and PTS]({breakdown_png_rel})",
        "",
        "- `predictor` is predictor-call time; `non-predictor` includes search, bookkeeping, and contention accounting.",
        "",
        "### Breakdown Table",
        "",
        *_build_latency_table(breakdown_df),
        "",
        "### Breakdown Notes",
        "",
        *_build_breakdown_interpretation(breakdown_df),
        "",
        "## Notes",
        "",
        "- This sidecar isolates `k=64`; it does not by itself support claims for all request sizes or contention modes.",
        "- Pair the sidecar with Tier 1 measured latency and full dispatch-quality metrics before using it as paper evidence.",
        "- The public naming is `PTS` for the current PTS path and `legacy-PTS` for the exact legacy PTS path.",
        het_same_combo_line,
        "",
    ]

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    latest_path.write_text(report_text, encoding="utf-8")
    return {"report_md": report_path, "latest_report_md": latest_path}


def main() -> None:
    """Run the PTS-sidecar report-builder CLI."""

    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    breakdown_df = pd.read_csv(args.breakdown_csv)
    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    build_report(
        summary_df=summary_df,
        breakdown_df=breakdown_df,
        metadata=metadata,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
