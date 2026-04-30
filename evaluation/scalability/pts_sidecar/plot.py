"""Plot the scalability PTS-sidecar summaries.

The plotter renders English-labeled figures for PTS speedup, grouped
latency, and latency-breakdown comparisons from regenerated summary CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


LEGACY_PTS_ALGO = "legacy-PTS"
PTS_ALGO = "PTS"
CLUSTER_COLORS = {
    "H100_26H100_27H100_28H100_29": "#1f77b4",
    "Het-4Mix": "#d62728",
}
ALGO_COLORS = {
    LEGACY_PTS_ALGO: "#9c755f",
    PTS_ALGO: "#59a14f",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PTS-sidecar figure generation."""

    parser = argparse.ArgumentParser(description="Build PTS-sidecar figures")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Summary CSV written by analyze.py")
    parser.add_argument("--breakdown-csv", type=Path, required=True, help="Breakdown CSV written by analyze.py")
    parser.add_argument("--figure-dir", type=Path, required=True, help="Directory for PNG/PDF figures")
    return parser.parse_args()


def _cluster_order(summary_df: pd.DataFrame) -> list[str]:
    """Return a stable cluster display order with H100 first when present."""

    preferred = ["H100_26H100_27H100_28H100_29", "Het-4Mix"]
    existing = summary_df["cluster_type"].drop_duplicates().astype(str).tolist()
    ordered = [cluster for cluster in preferred if cluster in existing]
    ordered.extend(cluster for cluster in existing if cluster not in ordered)
    return ordered


def _save_figure(fig: plt.Figure, base_path: Path) -> Dict[str, Path]:
    """Save a matplotlib figure as both PNG and PDF."""

    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}


def build_speedup_figure(summary_df: pd.DataFrame, figure_dir: Path) -> Dict[str, Path]:
    """Build the PTS speedup-over-legacy-PTS figure."""

    clusters = _cluster_order(summary_df)
    fig, axes = plt.subplots(1, len(clusters), figsize=(6.2 * len(clusters), 4.6), sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    for ax, cluster_type in zip(axes, clusters):
        cluster_df = summary_df[summary_df["cluster_type"] == cluster_type].sort_values("total_gpu")
        x = cluster_df["total_gpu"].astype(int).to_numpy()
        y = cluster_df["speedup_mean"].astype(float).to_numpy()
        yerr = cluster_df["speedup_std"].astype(float).to_numpy()

        # Error bars show cross-repeat standard deviation.
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=4,
            color=CLUSTER_COLORS.get(cluster_type, "#1f77b4"),
            label="PTS / legacy-PTS",
        )
        ax.axhline(1.0, color="#444444", linewidth=1.2, linestyle="--")
        ax.set_xticks(x)
        ax.set_xlabel("Total GPUs")
        ax.set_title(cluster_type)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
        for x_value, y_value in zip(x, y):
            ax.annotate(f"{y_value:.2f}x", (x_value, y_value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    axes[0].set_ylabel("Speedup over legacy-PTS (x)")
    fig.suptitle("PTS speedup over legacy-PTS", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_figure(fig, figure_dir / "pts_speedup_vs_legacy_pts")


def build_breakdown_figure(breakdown_df: pd.DataFrame, figure_dir: Path) -> Dict[str, Path]:
    """Build the predictor versus non-predictor latency-breakdown figure."""

    clusters = _cluster_order(breakdown_df)
    fig, axes = plt.subplots(1, len(clusters), figsize=(7.3 * len(clusters), 5.2), sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    for ax, cluster_type in zip(axes, clusters):
        cluster_df = breakdown_df[breakdown_df["cluster_type"] == cluster_type].copy()
        cluster_df["bar_label"] = cluster_df.apply(
            lambda row: f"{int(row['total_gpu'])}\n{row['algorithm']}",
            axis=1,
        )
        cluster_df = cluster_df.sort_values(["total_gpu", "algorithm"]).reset_index(drop=True)

        x = np.arange(len(cluster_df))
        predictor = cluster_df["predictor_time_mean_s"].astype(float).to_numpy()
        non_predictor = cluster_df["non_predictor_time_mean_s"].astype(float).to_numpy()

        # The breakdown separates predictor time from all non-predictor search
        # work; contention and PTS-phase time are already included there.
        bar_colors = [ALGO_COLORS.get(algo, "#4c78a8") for algo in cluster_df["algorithm"].astype(str)]
        ax.bar(x, predictor, color="#bab0ab", edgecolor="white", linewidth=0.6, label="Predictor")
        ax.bar(
            x,
            non_predictor,
            bottom=predictor,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.6,
            label="Non-predictor",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_df["bar_label"].tolist())
        ax.set_xlabel("Scale and algorithm")
        ax.set_title(cluster_type)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

    # Use one legend to distinguish predictor time from algorithm-specific
    # non-predictor time.
    legend_handles = [
        Patch(facecolor="#bab0ab", edgecolor="white", label="Predictor"),
        Patch(facecolor=ALGO_COLORS[LEGACY_PTS_ALGO], edgecolor="white", label="Non-predictor (legacy-PTS)"),
        Patch(facecolor=ALGO_COLORS[PTS_ALGO], edgecolor="white", label="Non-predictor (PTS)"),
    ]
    axes[0].set_ylabel("Latency (s)")
    fig.suptitle("Latency breakdown of legacy-PTS and PTS", fontsize=14, y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=1.6,
    )
    # Reserve top space for the shared legend before applying tight layout.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    return _save_figure(fig, figure_dir / "latency_breakdown_legacy_pts_vs_pts")


def build_latency_comparison_figure(breakdown_df: pd.DataFrame, figure_dir: Path) -> Dict[str, Path]:
    """Build the latency comparison for `legacy-PTS` and `PTS`.

    The figure complements the speedup chart with grouped bars over
    `latency_mean_s`, making the absolute 128/256-GPU gap visible.
    """

    clusters = _cluster_order(breakdown_df)
    fig, axes = plt.subplots(1, len(clusters), figsize=(6.8 * len(clusters), 4.8), sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    bar_width = 0.34
    algo_order = [LEGACY_PTS_ALGO, PTS_ALGO]
    legend_handles = [
        Patch(facecolor=ALGO_COLORS[LEGACY_PTS_ALGO], edgecolor="white", label=LEGACY_PTS_ALGO),
        Patch(facecolor=ALGO_COLORS[PTS_ALGO], edgecolor="white", label=PTS_ALGO),
    ]

    for ax, cluster_type in zip(axes, clusters):
        cluster_df = (
            breakdown_df[breakdown_df["cluster_type"] == cluster_type]
            .pivot_table(
                index="total_gpu",
                columns="algorithm",
                values="latency_mean_s",
                aggfunc="first",
            )
            .sort_index()
        )
        x = np.arange(len(cluster_df.index))

        for algo_idx, algorithm in enumerate(algo_order):
            latency = cluster_df[algorithm].astype(float).to_numpy()
            offset = (algo_idx - 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                latency,
                width=bar_width,
                color=ALGO_COLORS[algorithm],
                edgecolor="white",
                linewidth=0.7,
                label=algorithm,
            )
            for bar, value in zip(bars, latency):
                ax.annotate(
                    f"{value:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    rotation=0,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(value)) for value in cluster_df.index.tolist()])
        ax.set_xlabel("Total GPUs")
        ax.set_title(cluster_type)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

    axes[0].set_ylabel("Latency (s)")
    fig.suptitle("Total latency comparison: legacy-PTS vs PTS", fontsize=14, y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    return _save_figure(fig, figure_dir / "latency_bar_legacy_pts_vs_pts")


def build_all_figures(summary_df: pd.DataFrame, breakdown_df: pd.DataFrame, figure_dir: Path) -> Dict[str, Path]:
    """Build all PTS-sidecar figures and return their paths."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    speedup_paths = build_speedup_figure(summary_df=summary_df, figure_dir=figure_dir)
    latency_bar_paths = build_latency_comparison_figure(breakdown_df=breakdown_df, figure_dir=figure_dir)
    breakdown_paths = build_breakdown_figure(breakdown_df=breakdown_df, figure_dir=figure_dir)
    return {
        "speedup_png": speedup_paths["png"],
        "speedup_pdf": speedup_paths["pdf"],
        "latency_bar_png": latency_bar_paths["png"],
        "latency_bar_pdf": latency_bar_paths["pdf"],
        "breakdown_png": breakdown_paths["png"],
        "breakdown_pdf": breakdown_paths["pdf"],
    }


def main() -> None:
    """CLI entry point for PTS-sidecar plotting."""

    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    breakdown_df = pd.read_csv(args.breakdown_csv)
    build_all_figures(summary_df=summary_df, breakdown_df=breakdown_df, figure_dir=args.figure_dir)


if __name__ == "__main__":
    main()
