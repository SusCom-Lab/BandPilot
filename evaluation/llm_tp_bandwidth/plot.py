"""Plot tensor-parallel bandwidth sidecar summaries.

The plotting code reads analyzer outputs and writes English-labeled PNG files
for step time, throughput, memory, and NCCL-sidecar summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from pandas.errors import EmptyDataError


def _load_summary(run_dir: Path) -> Dict[str, object]:
    """Load the analyzer summary for one run directory."""

    summary_dir = run_dir / "summary"
    run_summary = json.loads((summary_dir / "run_summary.json").read_text(encoding="utf-8"))
    iteration_df = pd.read_csv(summary_dir / "iteration_metrics.csv")
    phase_df = pd.read_csv(summary_dir / "phase_summary.csv")
    profiler_path = summary_dir / "profiler_steps.csv"
    if profiler_path.exists():
        try:
            profiler_df = pd.read_csv(profiler_path)
        except EmptyDataError:
            # `profile_steps=0` creates an empty file; treat it as no profiler rows.
            profiler_df = pd.DataFrame()
    else:
        profiler_df = pd.DataFrame()
    return {
        "run_summary": run_summary,
        "iteration_df": iteration_df,
        "phase_df": phase_df,
        "profiler_df": profiler_df,
    }


def _save_figure(path: Path) -> Path:
    """Save the current matplotlib figure and close it."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _plot_step_time_series(iteration_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    """Plot per-step wall time by phase."""

    if iteration_df.empty:
        return None

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4.5))
    for phase, phase_df in iteration_df.groupby("phase", sort=False):
        plt.plot(
            phase_df["step_idx"],
            phase_df["host_step_wall_time_ms"],
            marker="o",
            linewidth=1.5,
            label=f"{phase} total",
        )
    plt.xlabel("Step")
    plt.ylabel("Step Time (ms)")
    plt.title("Per-Step Training Time")
    plt.grid(alpha=0.25)
    plt.legend()
    return _save_figure(figure_dir / "step_time_series.png")


def _plot_phase_breakdown(phase_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    """Plot mean forward, backward, and optimizer time by phase."""

    if phase_df.empty:
        return None

    target_df = phase_df[phase_df["phase"].isin(["measured", "profile"])].copy()
    if target_df.empty:
        target_df = phase_df.copy()

    import matplotlib.pyplot as plt

    x = range(len(target_df))
    plt.figure(figsize=(8.5, 4.5))
    plt.bar(x, target_df["forward_mean_ms"], label="Forward")
    plt.bar(x, target_df["backward_mean_ms"], bottom=target_df["forward_mean_ms"], label="Backward")
    bottom = target_df["forward_mean_ms"] + target_df["backward_mean_ms"]
    plt.bar(x, target_df["optimizer_mean_ms"], bottom=bottom, label="Optimizer")
    plt.xticks(list(x), target_df["phase"])
    plt.ylabel("Average Time (ms)")
    plt.title("Average Phase Breakdown")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    return _save_figure(figure_dir / "phase_breakdown.png")


def _plot_nccl_profile(profiler_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    """Plot profiler-side NCCL CUDA time against total step time."""

    if profiler_df.empty or "nccl_cuda_time_ms" not in profiler_df.columns:
        return None

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.5, 4.5))
    plt.bar(profiler_df["step_idx"], profiler_df["host_step_wall_time_ms"], alpha=0.35, label="Step Total")
    plt.bar(profiler_df["step_idx"], profiler_df["nccl_cuda_time_ms"], alpha=0.85, label="NCCL CUDA")
    plt.xlabel("Profile Step")
    plt.ylabel("Time (ms)")
    plt.title("Profiler Sidecar NCCL Time")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    return _save_figure(figure_dir / "nccl_profile.png")


def _plot_memory_trace(iteration_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    """Plot per-step peak allocated and reserved GPU memory."""

    if iteration_df.empty:
        return None

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4.5))
    plt.plot(iteration_df["step_idx"], iteration_df["max_memory_allocated_mb"], marker="o", label="Allocated")
    plt.plot(iteration_df["step_idx"], iteration_df["max_memory_reserved_mb"], marker="s", label="Reserved")
    plt.xlabel("Step")
    plt.ylabel("Memory (MB)")
    plt.title("Peak GPU Memory Per Step")
    plt.grid(alpha=0.25)
    plt.legend()
    return _save_figure(figure_dir / "memory_trace.png")


def build_all_figures(run_dir: Path) -> Dict[str, Path]:
    """Build all figures for one run and return generated paths."""

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_summary(run_dir)
    iteration_df = payload["iteration_df"]
    phase_df = payload["phase_df"]
    profiler_df = payload["profiler_df"]

    figure_paths = {
        "step_time_series_png": _plot_step_time_series(iteration_df, figure_dir),
        "phase_breakdown_png": _plot_phase_breakdown(phase_df, figure_dir),
        "nccl_profile_png": _plot_nccl_profile(profiler_df, figure_dir),
        "memory_trace_png": _plot_memory_trace(iteration_df, figure_dir),
    }
    return {key: value for key, value in figure_paths.items() if value is not None}
