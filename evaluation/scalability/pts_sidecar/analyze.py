"""Analyze the scalability PTS sidecar.

The analyzer consumes raw runner rows, computes paired legacy-PTS versus PTS
latency/speedup summaries, and writes regenerated CSV outputs for
plotting and report generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


LEGACY_PTS_ALGO = "legacy-PTS"
PTS_ALGO = "PTS"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for aggregating a raw PTS-sidecar CSV."""

    parser = argparse.ArgumentParser(description="Aggregate PTS-sidecar raw rows")
    parser.add_argument("--raw-csv", type=Path, required=True, help="Raw CSV written by runner.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for aggregated CSV outputs")
    return parser.parse_args()


def _build_pair_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw rows into a case-aligned paired wide table.

    Each case must contain both `legacy-PTS` and `PTS`. The resulting frame is
    used to compute latency speedup, bandwidth drift, and combo equality.
    """

    if raw_df.empty:
        return pd.DataFrame()

    case_keys = [
        "cluster_type",
        "total_gpu",
        "k",
        "avail_ratio",
        "contention_mode",
        "inter_pod_factor",
        "repeat_idx",
        "seed",
    ]
    pivot_columns = [
        "algorithm",
        "measured_wall_time_s",
        "predictor_time_s",
        "predictor_calls",
        "contention_time_s",
        "pts_phase_time_s",
        "non_predictor_search_time_s",
        "final_bw",
        "combo_signature",
        "evidence_type",
        "latency_evidence_kind",
        "bandwidth_evidence_kind",
    ]
    missing = [column for column in case_keys + pivot_columns if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {missing}")

    # Keep only the paired algorithms before pivoting.
    pair_df = raw_df[raw_df["algorithm"].isin([LEGACY_PTS_ALGO, PTS_ALGO])].copy()
    if pair_df.empty:
        raise ValueError("Raw CSV does not contain legacy-PTS / PTS rows.")

    pair_df = pair_df.sort_values(case_keys + ["algorithm"]).reset_index(drop=True)
    wide = pair_df.pivot_table(
        index=case_keys,
        columns="algorithm",
        values=[
            "measured_wall_time_s",
            "predictor_time_s",
            "predictor_calls",
            "contention_time_s",
            "pts_phase_time_s",
            "non_predictor_search_time_s",
            "final_bw",
            "combo_signature",
            "evidence_type",
            "latency_evidence_kind",
            "bandwidth_evidence_kind",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{algo}__{metric}" for metric, algo in wide.columns]
    wide = wide.reset_index()

    required_pairs = [
        f"{LEGACY_PTS_ALGO}__measured_wall_time_s",
        f"{PTS_ALGO}__measured_wall_time_s",
    ]
    if wide[required_pairs].isna().any().any():
        raise ValueError("Found incomplete paired rows: some cases miss either legacy-PTS or PTS.")

    # Derived metrics: speedup, quality drift, and whether the selected combo matches.
    wide["speedup_vs_legacy_pts"] = (
        wide[f"{LEGACY_PTS_ALGO}__measured_wall_time_s"].astype(float)
        / wide[f"{PTS_ALGO}__measured_wall_time_s"].astype(float)
    )
    wide["bw_delta_pct_pts_vs_legacy_pts"] = np.where(
        wide[f"{LEGACY_PTS_ALGO}__final_bw"].astype(float) > 0,
        (
            wide[f"{PTS_ALGO}__final_bw"].astype(float)
            - wide[f"{LEGACY_PTS_ALGO}__final_bw"].astype(float)
        )
        / wide[f"{LEGACY_PTS_ALGO}__final_bw"].astype(float)
        * 100.0,
        np.nan,
    )
    wide["same_combo"] = (
        wide[f"{PTS_ALGO}__combo_signature"].fillna("").astype(str)
        == wide[f"{LEGACY_PTS_ALGO}__combo_signature"].fillna("").astype(str)
    )
    return wide


def _summarize_algorithm(
    pair_df: pd.DataFrame,
    algorithm: str,
) -> Dict[str, float]:
    """Aggregate latency, predictor, contention, and bandwidth metrics."""

    latency = pair_df[f"{algorithm}__measured_wall_time_s"].astype(float)
    predictor = pair_df[f"{algorithm}__predictor_time_s"].astype(float)
    non_predictor = pair_df[f"{algorithm}__non_predictor_search_time_s"].astype(float)
    contention = pair_df[f"{algorithm}__contention_time_s"].astype(float)
    pts_phase = pair_df[f"{algorithm}__pts_phase_time_s"].astype(float)
    final_bw = pair_df[f"{algorithm}__final_bw"].astype(float)

    predictor_share_pct = np.where(latency > 0, predictor / latency * 100.0, np.nan)
    non_predictor_share_pct = np.where(latency > 0, non_predictor / latency * 100.0, np.nan)
    contention_share_pct = np.where(latency > 0, contention / latency * 100.0, np.nan)

    return {
        f"{algorithm.lower()}_latency_mean_s": float(latency.mean()),
        f"{algorithm.lower()}_latency_std_s": float(latency.std(ddof=0)),
        f"{algorithm.lower()}_latency_p50_s": float(np.percentile(latency, 50)),
        f"{algorithm.lower()}_latency_p95_s": float(np.percentile(latency, 95)),
        f"{algorithm.lower()}_predictor_time_mean_s": float(predictor.mean()),
        f"{algorithm.lower()}_non_predictor_time_mean_s": float(non_predictor.mean()),
        f"{algorithm.lower()}_contention_time_mean_s": float(contention.mean()),
        f"{algorithm.lower()}_pts_phase_time_mean_s": float(pts_phase.mean()),
        f"{algorithm.lower()}_predictor_share_pct_mean": float(np.nanmean(predictor_share_pct)),
        f"{algorithm.lower()}_non_predictor_share_pct_mean": float(np.nanmean(non_predictor_share_pct)),
        f"{algorithm.lower()}_contention_share_pct_mean": float(np.nanmean(contention_share_pct)),
        f"{algorithm.lower()}_final_bw_mean": float(final_bw.mean()),
        f"{algorithm.lower()}_final_bw_std": float(final_bw.std(ddof=0)),
    }


def build_summary_tables(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build paired rows, summary rows, and latency-breakdown rows from raw data."""

    paired_df = _build_pair_frame(raw_df)
    if paired_df.empty:
        empty = pd.DataFrame()
        return paired_df, empty, empty

    summary_rows = []
    breakdown_rows = []

    group_keys = ["cluster_type", "total_gpu", "k", "avail_ratio", "contention_mode", "inter_pod_factor"]
    for key_values, group_df in paired_df.groupby(group_keys, sort=True):
        cluster_type, total_gpu, k_value, avail_ratio, contention_mode, inter_pod_factor = key_values

        # Summarize each paired scenario group with speedup and quality metrics.
        summary_row = {
            "cluster_type": cluster_type,
            "total_gpu": int(total_gpu),
            "k": int(k_value),
            "avail_ratio": float(avail_ratio),
            "contention_mode": str(contention_mode),
            "inter_pod_factor": float(inter_pod_factor),
            "repeat_count": int(len(group_df)),
            "evidence_type": str(group_df[f"{LEGACY_PTS_ALGO}__evidence_type"].iloc[0]),
            "latency_evidence_kind": str(group_df[f"{LEGACY_PTS_ALGO}__latency_evidence_kind"].iloc[0]),
            "bandwidth_evidence_kind": str(group_df[f"{LEGACY_PTS_ALGO}__bandwidth_evidence_kind"].iloc[0]),
            "speedup_mean": float(group_df["speedup_vs_legacy_pts"].mean()),
            "speedup_std": float(group_df["speedup_vs_legacy_pts"].std(ddof=0)),
            "speedup_p50": float(np.percentile(group_df["speedup_vs_legacy_pts"], 50)),
            "speedup_p95": float(np.percentile(group_df["speedup_vs_legacy_pts"], 95)),
            "same_combo_rate": float(group_df["same_combo"].mean()),
            "bw_delta_pct_pts_vs_legacy_pts_mean": float(group_df["bw_delta_pct_pts_vs_legacy_pts"].mean()),
            "bw_delta_pct_pts_vs_legacy_pts_std": float(group_df["bw_delta_pct_pts_vs_legacy_pts"].std(ddof=0)),
        }
        summary_row.update(_summarize_algorithm(group_df, LEGACY_PTS_ALGO))
        summary_row.update(_summarize_algorithm(group_df, PTS_ALGO))
        summary_rows.append(summary_row)

        # Breakdown rows retain one row per cluster, scale, and algorithm.
        for algorithm in [LEGACY_PTS_ALGO, PTS_ALGO]:
            breakdown_rows.append(
                {
                    "cluster_type": cluster_type,
                    "total_gpu": int(total_gpu),
                    "algorithm": algorithm,
                    "repeat_count": int(len(group_df)),
                    "latency_mean_s": float(group_df[f"{algorithm}__measured_wall_time_s"].astype(float).mean()),
                    "predictor_time_mean_s": float(group_df[f"{algorithm}__predictor_time_s"].astype(float).mean()),
                    "non_predictor_time_mean_s": float(group_df[f"{algorithm}__non_predictor_search_time_s"].astype(float).mean()),
                    "contention_time_mean_s": float(group_df[f"{algorithm}__contention_time_s"].astype(float).mean()),
                    "pts_phase_time_mean_s": float(group_df[f"{algorithm}__pts_phase_time_s"].astype(float).mean()),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["cluster_type", "total_gpu"]).reset_index(drop=True)
    breakdown_df = pd.DataFrame(breakdown_rows).sort_values(["cluster_type", "total_gpu", "algorithm"]).reset_index(drop=True)
    return paired_df, summary_df, breakdown_df


def write_summary_artifacts(raw_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Write summary CSVs and an analysis manifest under the output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_df, summary_df, breakdown_df = build_summary_tables(raw_df)

    paired_path = output_dir / "paired_rows.csv"
    summary_path = output_dir / "summary.csv"
    breakdown_path = output_dir / "breakdown_summary.csv"

    paired_df.to_csv(paired_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    breakdown_df.to_csv(breakdown_path, index=False)

    manifest = {
        "paired_rows_csv": str(paired_path),
        "summary_csv": str(summary_path),
        "breakdown_csv": str(breakdown_path),
        "pair_count": int(len(paired_df)),
        "summary_row_count": int(len(summary_df)),
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "paired_rows_csv": paired_path,
        "summary_csv": summary_path,
        "breakdown_csv": breakdown_path,
        "analysis_manifest_json": manifest_path,
    }


def main() -> None:
    """CLI entry point for summary and breakdown generation."""

    args = parse_args()
    raw_df = pd.read_csv(args.raw_csv)
    write_summary_artifacts(raw_df=raw_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
