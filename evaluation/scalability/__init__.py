"""Scalability benchmark package.

This package contains the benchmark runner, report builder, artifact rebuild
helpers, configs, and focused sidecars for BandPilot scalability-latency
evidence.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
BENCHMARK_ARTIFACTS_DIR = ARTIFACTS_DIR / "benchmark"
CURRENT_BENCHMARK_ARTIFACT_DIR = (
    BENCHMARK_ARTIFACTS_DIR / "current" / "search_overhead_adaptive_main"
)
SMOKE_BENCHMARK_ARTIFACT_DIR = (
    BENCHMARK_ARTIFACTS_DIR / "smoke" / "search_overhead_post_activation_smoke"
)
RECOVERY_BENCHMARK_ARTIFACT_DIR = (
    BENCHMARK_ARTIFACTS_DIR / "recovery" / "search_overhead_het4mix_resume"
)
ARCHIVE_BENCHMARK_ARTIFACT_DIR = (
    BENCHMARK_ARTIFACTS_DIR / "archive" / "search_overhead"
)
WELL_SHOW_REPORT_DIR = PACKAGE_ROOT / "reports" / "well-show"

FULL_CONFIG_PATH = CONFIG_DIR / "full.yaml"
POST_ACTIVATION_SMOKE_CONFIG_PATH = CONFIG_DIR / "post_activation_smoke.yaml"
HET4MIX_RESUME_CONFIG_PATH = CONFIG_DIR / "het4mix_resume.yaml"
PTS_SIDECAR_CONFIG_PATH = CONFIG_DIR / "pts_sidecar.yaml"

__all__ = [
    "ARCHIVE_BENCHMARK_ARTIFACT_DIR",
    "ARTIFACTS_DIR",
    "BENCHMARK_ARTIFACTS_DIR",
    "CONFIG_DIR",
    "CURRENT_BENCHMARK_ARTIFACT_DIR",
    "FULL_CONFIG_PATH",
    "HET4MIX_RESUME_CONFIG_PATH",
    "PACKAGE_ROOT",
    "POST_ACTIVATION_SMOKE_CONFIG_PATH",
    "RECOVERY_BENCHMARK_ARTIFACT_DIR",
    "PTS_SIDECAR_CONFIG_PATH",
    "SMOKE_BENCHMARK_ARTIFACT_DIR",
    "WELL_SHOW_REPORT_DIR",
]
