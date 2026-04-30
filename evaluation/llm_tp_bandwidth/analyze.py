"""Analyze raw tensor-parallel sidecar JSONL outputs.

The analyzer converts worker step records into iteration metrics, phase
summaries, profiler summaries, and `run_summary.json` for downstream plotting
and report generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.llm_tp_bandwidth.io_utils import load_json, write_json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sidecar analysis."""

    parser = argparse.ArgumentParser(description="Aggregate llm_tp_bandwidth raw JSONL")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory produced by runner.py")
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    """Read JSONL rows, returning an empty list when the file is absent."""

    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _to_frame(rows: List[Dict[str, object]]) -> pd.DataFrame:
    """Convert JSONL rows into a DataFrame."""

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric values."""

    return pd.to_numeric(series, errors="coerce")


def _phase_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize step metrics by phase."""

    if step_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for phase, phase_df in step_df.groupby("phase", sort=False):
        total_ms = _numeric(phase_df["host_step_wall_time_ms"])
        row = {
            "phase": str(phase),
            "step_count": int(len(phase_df)),
            "step_time_mean_ms": float(total_ms.mean()),
            "step_time_median_ms": float(total_ms.median()),
            "step_time_p95_ms": float(np.percentile(total_ms, 95)),
            "forward_mean_ms": float(_numeric(phase_df["forward_ms"]).mean()),
            "backward_mean_ms": float(_numeric(phase_df["backward_ms"]).mean()),
            "optimizer_mean_ms": float(_numeric(phase_df["optimizer_ms"]).mean()),
            "tokens_per_sec_mean": float(_numeric(phase_df["tokens_per_sec"]).mean()),
            "max_memory_allocated_mb_mean": float(_numeric(phase_df["max_memory_allocated_mb"]).mean()),
            "max_memory_allocated_mb_max": float(_numeric(phase_df["max_memory_allocated_mb"]).max()),
            "max_memory_reserved_mb_max": float(_numeric(phase_df["max_memory_reserved_mb"]).max()),
            "loss_mean": float(_numeric(phase_df["loss"]).mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _profiler_summary(profiler_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize NCCL profiler sidecar rows."""

    if profiler_df.empty:
        return pd.DataFrame()

    nccl_ms = _numeric(profiler_df["nccl_cuda_time_ms"])
    step_ms = _numeric(profiler_df["host_step_wall_time_ms"])
    nccl_share = np.where(step_ms > 0, nccl_ms / step_ms * 100.0, np.nan)

    summary_row = {
        "profile_step_count": int(len(profiler_df)),
        "nccl_cuda_time_mean_ms": float(nccl_ms.mean()),
        "nccl_cuda_time_median_ms": float(nccl_ms.median()),
        "nccl_cuda_time_p95_ms": float(np.percentile(nccl_ms, 95)),
        "nccl_share_mean_pct": float(np.nanmean(nccl_share)),
        "nccl_share_median_pct": float(np.nanmedian(nccl_share)),
        "nccl_kernel_count_mean": float(_numeric(profiler_df["nccl_kernel_count"]).mean()),
        "profile_step_time_mean_ms": float(step_ms.mean()),
    }
    return pd.DataFrame([summary_row])


def _run_summary(
    *,
    metadata: Dict[str, object],
    step_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    profiler_summary_df: pd.DataFrame,
) -> Dict[str, object]:
    """Build the `run_summary.json` payload.

    The summary is intentionally flat so report_builder and plots can consume it
    without re-reading raw JSONL rows.
    """

    measured_df = step_df[step_df["phase"] == "measured"].copy() if not step_df.empty else pd.DataFrame()
    measured_time = _numeric(measured_df["host_step_wall_time_ms"]) if not measured_df.empty else pd.Series(dtype=float)

    summary: Dict[str, object] = {
        "status": metadata.get("status", "unknown"),
        "evidence_type": metadata.get("evidence_type", "measured"),
        "run_tag": metadata.get("run_tag"),
        "selected_model": metadata.get("selected_model"),
        "model_path": metadata.get("selected_model_path"),
        "gpu_pair": metadata.get("gpu_pair"),
        "requested_bridge_type": metadata.get("requested_bridge_type"),
        "detected_bridge_type": metadata.get("detected_bridge_type"),
        "detected_topology_label": metadata.get("detected_topology_label"),
        "warmup_steps_requested": metadata.get("warmup_steps"),
        "measured_steps_requested": metadata.get("measured_steps"),
        "profile_steps_requested": metadata.get("profile_steps"),
        "completed_total_steps": int(len(step_df)),
        "completed_measured_steps": int(len(measured_df)),
    }

    if not measured_df.empty:
        summary.update(
            {
                "measured_step_time_mean_ms": float(measured_time.mean()),
                "measured_step_time_median_ms": float(measured_time.median()),
                "measured_step_time_p95_ms": float(np.percentile(measured_time, 95)),
                "measured_tokens_per_sec_mean": float(_numeric(measured_df["tokens_per_sec"]).mean()),
                "measured_forward_mean_ms": float(_numeric(measured_df["forward_ms"]).mean()),
                "measured_backward_mean_ms": float(_numeric(measured_df["backward_ms"]).mean()),
                "measured_optimizer_mean_ms": float(_numeric(measured_df["optimizer_ms"]).mean()),
                "measured_loss_mean": float(_numeric(measured_df["loss"]).mean()),
                "measured_max_memory_allocated_mb_max": float(_numeric(measured_df["max_memory_allocated_mb"]).max()),
            }
        )
    else:
        summary.update(
            {
                "measured_step_time_mean_ms": None,
                "measured_step_time_median_ms": None,
                "measured_step_time_p95_ms": None,
                "measured_tokens_per_sec_mean": None,
                "measured_forward_mean_ms": None,
                "measured_backward_mean_ms": None,
                "measured_optimizer_mean_ms": None,
                "measured_loss_mean": None,
                "measured_max_memory_allocated_mb_max": None,
            }
        )

    if not profiler_summary_df.empty:
        row = profiler_summary_df.iloc[0].to_dict()
        summary.update(row)
        # Pandas may round-trip counts through floats; restore integer semantics.
        summary["profile_step_count"] = int(round(float(summary["profile_step_count"])))
    else:
        summary.update(
            {
                "profile_step_count": 0,
                "nccl_cuda_time_mean_ms": None,
                "nccl_cuda_time_median_ms": None,
                "nccl_cuda_time_p95_ms": None,
                "nccl_share_mean_pct": None,
                "nccl_share_median_pct": None,
                "nccl_kernel_count_mean": None,
                "profile_step_time_mean_ms": None,
            }
        )

    summary["phase_rows"] = phase_df.to_dict(orient="records")
    return summary


def build_summary_artifacts(run_dir: Path) -> Dict[str, Path]:
    """Build summary CSV and JSON artifacts under a run directory."""

    raw_dir = run_dir / "raw"
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(run_dir / "run_metadata.json")
    step_rows = _read_jsonl(raw_dir / "step_metrics.jsonl")
    profiler_rows = _read_jsonl(raw_dir / "profiler_steps.jsonl")

    step_df = _to_frame(step_rows)
    profiler_df = _to_frame(profiler_rows)
    phase_df = _phase_summary(step_df)
    profiler_summary_df = _profiler_summary(profiler_df)
    run_summary = _run_summary(
        metadata=metadata,
        step_df=step_df,
        phase_df=phase_df,
        profiler_summary_df=profiler_summary_df,
    )

    iteration_csv = summary_dir / "iteration_metrics.csv"
    phase_csv = summary_dir / "phase_summary.csv"
    profiler_csv = summary_dir / "profiler_summary.csv"
    run_summary_json = summary_dir / "run_summary.json"

    step_df.to_csv(iteration_csv, index=False)
    phase_df.to_csv(phase_csv, index=False)
    profiler_df.to_csv(summary_dir / "profiler_steps.csv", index=False)
    profiler_summary_df.to_csv(profiler_csv, index=False)
    write_json(run_summary_json, run_summary)

    return {
        "iteration_csv": iteration_csv,
        "phase_csv": phase_csv,
        "profiler_csv": profiler_csv,
        "run_summary_json": run_summary_json,
    }


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    build_summary_artifacts(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
