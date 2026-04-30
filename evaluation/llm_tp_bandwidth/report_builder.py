"""Build Markdown reports for tensor-parallel sidecar runs.

The report builder combines run metadata, analyzer summaries, and generated
figures into a compact measured-evidence report for the local GPU pair.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from evaluation.llm_tp_bandwidth.io_utils import load_json, write_json


def _format_ms(value: object) -> str:
    """Format a millisecond value for Markdown tables."""

    if value is None:
        return "N/A"
    return f"{float(value):.2f} ms"


def _format_pct(value: object) -> str:
    """Format a percentage value for Markdown tables."""

    if value is None:
        return "N/A"
    return f"{float(value):.2f}%"


def _format_float(value: object, digits: int = 2) -> str:
    """Format a floating-point value with a configurable precision."""

    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_count(value: object) -> str:
    """Format a count value as an integer string.

    JSON/CSV round-trips may turn integer counts into values such as `1.0`.
    """

    if value is None:
        return "N/A"
    return str(int(round(float(value))))


def _relative_path(from_dir: Path, to_path: Optional[Path]) -> Optional[str]:
    """Return a Markdown-friendly relative path."""

    if to_path is None:
        return None
    return os.path.relpath(to_path, start=from_dir).replace(os.sep, "/")


def build_report(run_dir: Path, figure_paths: Dict[str, Path]) -> Dict[str, Path]:
    """Build the run-specific report and update `latest_report.md`."""

    summary = load_json(run_dir / "summary" / "run_summary.json")
    metadata = load_json(run_dir / "run_metadata.json")
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Two-GPU TP Bandwidth Experiment Report",
        "",
        "## Run Summary",
        f"- Run tag: `{summary.get('run_tag')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Evidence type: `{summary.get('evidence_type')}`",
        f"- GPU pair: `{summary.get('gpu_pair')}`",
        f"- Bridge type: requested=`{summary.get('requested_bridge_type')}`, detected=`{summary.get('detected_bridge_type')}`",
        f"- Raw topology label: `{summary.get('detected_topology_label')}`",
        f"- Selected model: `{summary.get('selected_model')}`",
        f"- Model path: `{summary.get('model_path')}`",
        f"- Requested steps: warmup=`{summary.get('warmup_steps_requested')}`, measured=`{summary.get('measured_steps_requested')}`, profile=`{summary.get('profile_steps_requested')}`",
        f"- Completed steps: total=`{summary.get('completed_total_steps')}`, measured=`{summary.get('completed_measured_steps')}`",
        "",
        "## Core Training Metrics",
        f"- Mean measured step time: {_format_ms(summary.get('measured_step_time_mean_ms'))}",
        f"- Median measured step time: {_format_ms(summary.get('measured_step_time_median_ms'))}",
        f"- P95 measured step time: {_format_ms(summary.get('measured_step_time_p95_ms'))}",
        f"- Mean tokens/s: {_format_float(summary.get('measured_tokens_per_sec_mean'))}",
        f"- Mean forward time: {_format_ms(summary.get('measured_forward_mean_ms'))}",
        f"- Mean backward time: {_format_ms(summary.get('measured_backward_mean_ms'))}",
        f"- Mean optimizer time: {_format_ms(summary.get('measured_optimizer_mean_ms'))}",
        f"- Peak allocated memory (max across measured steps): {_format_float(summary.get('measured_max_memory_allocated_mb_max'))} MB",
        "",
        "## NCCL Profiler Sidecar",
        f"- Profile step count: `{_format_count(summary.get('profile_step_count'))}`",
        f"- Mean NCCL CUDA time: {_format_ms(summary.get('nccl_cuda_time_mean_ms'))}",
        f"- Median NCCL CUDA time: {_format_ms(summary.get('nccl_cuda_time_median_ms'))}",
        f"- P95 NCCL CUDA time: {_format_ms(summary.get('nccl_cuda_time_p95_ms'))}",
        f"- Mean NCCL share of profiled step: {_format_pct(summary.get('nccl_share_mean_pct'))}",
        f"- Mean NCCL kernel count: {_format_float(summary.get('nccl_kernel_count_mean'))}",
        "",
        "## Notes",
        "- This artifact is a `measured` two-GPU TP training sidecar.",
        "- NCCL timing comes from a short profiler sidecar and should be read as explanatory evidence for the main measured run.",
    ]

    if "aborted" in str(summary.get("status", "")):
        lines.extend(
            [
                "",
                "## Abort Information",
                f"- Worker status: `{summary.get('status')}`",
                f"- Error type: `{metadata.get('error_type')}`",
                f"- Error message: `{metadata.get('error_message')}`",
            ]
        )

    if figure_paths:
        lines.extend(["", "## Figures"])
        for figure_name, figure_path in sorted(figure_paths.items()):
            rel = _relative_path(report_dir, figure_path)
            lines.append(f"- [{figure_name}]({rel})")

    report_path = report_dir / f"{summary.get('run_tag')}_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest_path = report_dir / "latest_report.md"
    latest_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "report_md": str(report_path),
        "latest_report_md": str(latest_path),
        "figure_paths": {key: str(path) for key, path in figure_paths.items()},
    }
    write_json(report_dir / "report_manifest.json", manifest)
    return {
        "report_md": report_path,
        "latest_report_md": latest_path,
        "report_manifest_json": report_dir / "report_manifest.json",
    }
