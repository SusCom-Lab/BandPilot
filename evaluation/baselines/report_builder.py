"""Build summaries and reviewer-facing reports for the isolated baseline suite.

Function:
- Convert raw per-case baseline rows into compact summary CSVs and a Markdown
  report kept entirely under ``evaluation/baselines``.
- Generate a small set of figures when plotting dependencies are available.

Design:
- All aggregation logic is local to this directory, so no existing plotting or
  report code needs to be edited.
- The report emphasizes direct answers to the reviewer's baseline question:
  topology-aware vs bandwidth-aware vs simple learned model vs the configured
  reference algorithm.

Usage:
- ``python -m evaluation.baselines.report_builder --artifact-dir ...``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from algorithms.network_baselines import normalize_network_baseline_series


def parse_args() -> argparse.Namespace:
    """Parse the report-builder CLI."""
    parser = argparse.ArgumentParser(description="Build isolated baseline-suite reports")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Artifact directory produced by evaluation.baselines.run_suite",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional explicit report directory; defaults to <artifact-dir>/../reports-like location",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Optional explicit figure directory",
    )
    return parser.parse_args()


def _resolve_report_paths(artifact_dir: Path, report_dir: Path | None, figure_dir: Path | None) -> Tuple[Path, Path]:
    """Resolve report and figure output directories relative to the artifact directory."""
    run_tag = artifact_dir.name
    if report_dir is None:
        report_dir = artifact_dir.parents[1] / "reports" / run_tag
    if figure_dir is None:
        figure_dir = artifact_dir.parents[1] / "figures" / run_tag
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, figure_dir


def _load_metadata(artifact_dir: Path) -> Dict[str, object]:
    """Load run metadata so reports can follow the runner's reference label."""
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        return {"reference_algorithm": "BandPilot"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "reference_algorithm" in metadata:
        metadata["reference_algorithm"] = normalize_network_baseline_series(
            [str(metadata["reference_algorithm"])]
        )[0]
    return metadata


def _normalize_algorithm_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy `NetworkLocality` labels to `CasCore`."""

    normalized = rows.copy()
    if "algorithm" in normalized.columns:
        normalized["algorithm"] = normalize_network_baseline_series(normalized["algorithm"].tolist())
    return normalized


def _build_pairwise_summary(rows: pd.DataFrame, reference_algorithm: str) -> pd.DataFrame:
    """Compute per-algorithm deltas relative to the configured reference."""
    baseline = rows[rows["algorithm"] == reference_algorithm][
        ["cluster_type", "contention_mode", "test_num", "repeat_idx", "final_utilization", "elapsed_time"]
    ].rename(
        columns={
            "final_utilization": "reference_final_utilization",
            "elapsed_time": "reference_elapsed_time",
        }
    )
    merged = rows.merge(
        baseline,
        on=["cluster_type", "contention_mode", "test_num", "repeat_idx"],
        how="left",
    )
    merged["util_delta_vs_reference"] = merged["final_utilization"] - merged["reference_final_utilization"]
    merged["latency_delta_ms_vs_reference"] = (
        merged["elapsed_time"] - merged["reference_elapsed_time"]
    ) * 1000.0
    summary = (
        merged.groupby(["cluster_type", "algorithm"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            mean_util_delta_vs_reference=("util_delta_vs_reference", "mean"),
            mean_latency_delta_ms_vs_reference=("latency_delta_ms_vs_reference", "mean"),
        )
        .sort_values(["cluster_type", "mean_util_delta_vs_reference"], ascending=[True, False])
    )
    # Keep the legacy column aliases so old downstream readers do not break.
    summary["mean_util_delta_vs_bandpilot"] = summary["mean_util_delta_vs_reference"]
    summary["mean_latency_delta_ms_vs_bandpilot"] = summary["mean_latency_delta_ms_vs_reference"]
    return summary


def _write_markdown_report(
    report_path: Path,
    overall: pd.DataFrame,
    pairwise: pd.DataFrame,
    rows_path: Path,
    summary_path: Path,
    reference_algorithm: str,
) -> None:
    """Write a compact Markdown report summarizing the isolated baseline run."""
    def _frame_to_text(frame: pd.DataFrame) -> str:
        # ``to_markdown`` is nicer for human inspection, but it requires the
        # optional ``tabulate`` dependency. Fall back to CSV-style text if the
        # environment does not provide it.
        try:
            return frame.to_markdown(index=False)
        except Exception:
            return frame.to_csv(index=False).strip()

    lines = [
        "# Isolated Baseline Suite Report",
        "",
        "## Purpose",
        "",
        f"- Compare topology-aware, bandwidth-aware, simple learned, and `{reference_algorithm}` baselines under the same single-contention protocol.",
        "- Keep all new artifacts local to `evaluation/baselines`.",
        "",
        "## Overall Summary",
        "",
        _frame_to_text(overall),
        "",
        f"## Delta Vs {reference_algorithm}",
        "",
        _frame_to_text(pairwise),
        "",
        "## Artifact Paths",
        "",
        f"- Raw rows: `{rows_path}`",
        f"- Overall summary: `{summary_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_build_figures(rows: pd.DataFrame, figure_dir: Path) -> None:
    """Generate a small set of figures when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Figure 1: mean final utilization by algorithm and cluster.
    overall = (
        rows.groupby(["cluster_type", "algorithm"], as_index=False)
        .agg(mean_final_utilization=("final_utilization", "mean"))
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = overall.pivot(index="algorithm", columns="cluster_type", values="mean_final_utilization")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean final utilization (%)")
    ax.set_title("Baseline spectrum: mean final utilization")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_spectrum_mean_utilization.png", dpi=200)
    plt.close(fig)

    # Figure 2: common-mode mean latency vs k.
    common_rows = rows[rows["contention_mode"] == "common"]
    latency = (
        common_rows.groupby(["cluster_type", "algorithm", "test_num"], as_index=False)
        .agg(mean_elapsed_ms=("elapsed_time", lambda series: series.mean() * 1000.0))
    )
    for cluster_type in sorted(latency["cluster_type"].unique()):
        cluster_df = latency[latency["cluster_type"] == cluster_type]
        fig, ax = plt.subplots(figsize=(8, 4))
        for algorithm in sorted(cluster_df["algorithm"].unique()):
            algo_df = cluster_df[cluster_df["algorithm"] == algorithm]
            ax.plot(algo_df["test_num"], algo_df["mean_elapsed_ms"], marker="o", label=algorithm)
        ax.set_xlabel("Requested GPUs (k)")
        ax.set_ylabel("Mean search latency (ms)")
        ax.set_title(f"{cluster_type}: common-mode latency")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{cluster_type}_common_latency_curve.png", dpi=200)
        plt.close(fig)


def main() -> None:
    """Build summaries, optional figures, and a Markdown report from raw suite rows."""
    args = parse_args()
    report_dir, figure_dir = _resolve_report_paths(args.artifact_dir, args.report_dir, args.figure_dir)
    metadata = _load_metadata(args.artifact_dir)
    reference_algorithm = str(metadata.get("reference_algorithm", "BandPilot"))
    rows_path = args.artifact_dir / "rows.csv"
    rows = _normalize_algorithm_labels(pd.read_csv(rows_path))

    overall = (
        rows.groupby(["cluster_type", "algorithm"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            mean_final_bw=("final_bw", "mean"),
            mean_final_utilization=("final_utilization", "mean"),
            mean_elapsed_time=("elapsed_time", "mean"),
            p95_elapsed_time=("elapsed_time", lambda series: series.quantile(0.95)),
        )
        .sort_values(["cluster_type", "mean_final_utilization"], ascending=[True, False])
    )
    cluster_mode_k = (
        rows.groupby(["cluster_type", "contention_mode", "test_num", "algorithm"], as_index=False)
        .agg(
            sample_count=("algorithm", "size"),
            mean_final_utilization=("final_utilization", "mean"),
            mean_elapsed_time=("elapsed_time", "mean"),
        )
        .sort_values(["cluster_type", "contention_mode", "test_num", "algorithm"])
    )
    pairwise = _build_pairwise_summary(rows, reference_algorithm=reference_algorithm)

    overall_path = args.artifact_dir / "summary_overall.csv"
    cluster_mode_k_path = args.artifact_dir / "summary_cluster_mode_k.csv"
    pairwise_path = args.artifact_dir / "summary_vs_reference.csv"
    legacy_pairwise_path = args.artifact_dir / "summary_vs_bandpilot.csv"
    overall.to_csv(overall_path, index=False)
    cluster_mode_k.to_csv(cluster_mode_k_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    pairwise.to_csv(legacy_pairwise_path, index=False)

    _maybe_build_figures(rows, figure_dir)
    _write_markdown_report(
        report_path=report_dir / "baseline_suite_report.md",
        overall=overall,
        pairwise=pairwise,
        rows_path=rows_path,
        summary_path=overall_path,
        reference_algorithm=reference_algorithm,
    )
    print(f"Baseline report saved to {report_dir / 'baseline_suite_report.md'}")


if __name__ == "__main__":
    main()
