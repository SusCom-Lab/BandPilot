"""Rebuild reviewer-facing scalability artifacts from regenerated raw CSVs.

The tool audits per-cluster raw CSV completeness, optionally overlays recovery
outputs, and regenerates summary CSV/TEX/PDF material under ignored artifact
directories.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from evaluation.scalability import FULL_CONFIG_PATH
from evaluation.scalability.benchmark import (
    PUBLIC_PREDICTOR_PROFILE_PLOT_FILENAME,
    PUBLIC_REAL_TRACE_PLOT_FILENAME,
    PUBLIC_SCALED_TRACE_PLOT_FILENAME,
    PUBLIC_SUMMARY_FILENAME,
    PUBLIC_SUMMARY_TEX_FILENAME,
    PUBLIC_SYNTH_LATENCY_PLOT_FILENAME,
    PUBLIC_TRIGGER_RATE_PLOT_FILENAME,
    _build_latency_summary_table,
    _normalize_benchmark_dataframe,
    _resolve_public_view_cfg,
    _save_extrapolation_plot,
    _save_inference_scaling_plot,
    _save_latency_real_plot,
    _save_scaled_latency_plot,
    _save_trigger_rate_plot,
    _write_latency_summary_table,
)
from utils.helpers import ensure_directory

logger = logging.getLogger(__name__)

# Per-cluster raw CSV prefixes used to rebuild reviewer-facing artifacts.
RAW_ARTIFACT_SPECS: Tuple[Tuple[str, str], ...] = (
    ("tier1", "tier1_"),
    ("scaled_search", "scaled_search_"),
    ("predictor_latency_profile", "predictor_latency_profile_"),
    ("synthesized_dispatch_latency", "synthesized_dispatch_latency_"),
)


@dataclass(frozen=True)
class ArtifactSelection:
    """Selected raw artifact for one benchmark stage and cluster."""

    stage_name: str
    cluster_type: str
    source_path: Path
    source_kind: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the artifact rebuild tool."""

    parser = argparse.ArgumentParser(
        description="Rebuild reviewer-facing scalability artifacts from per-cluster raw CSV."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing raw artifacts, for example "
            "evaluation/scalability/artifacts/benchmark/current/search_overhead_adaptive_main"
        ),
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Optional overlay directory; overlay raw CSVs take precedence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for rebuilt summaries and figures; defaults to --artifact-dir.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=FULL_CONFIG_PATH,
        help="Benchmark config used to resolve expected cluster types and public view.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check completeness; do not rebuild artifacts.",
    )
    parser.add_argument(
        "--sync-overlay-raw",
        action="store_true",
        help="Copy selected overlay raw CSVs back into the artifact directory.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level; defaults to INFO.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path]) -> dict:
    """Load a YAML config file, returning an empty dict when unset."""

    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_expected_clusters(config: dict, artifact_dir: Path, overlay_dir: Optional[Path]) -> List[str]:
    """Resolve expected cluster types from config or artifact filenames.

    Configured `cluster.cluster_types` is authoritative. If it is absent, infer
    clusters from primary and overlay raw CSV filenames.
    """

    configured = config.get("cluster", {}).get("cluster_types")
    if configured:
        return [str(item) for item in configured]

    inferred = set()
    for directory in [artifact_dir, overlay_dir]:
        if directory is None or not directory.exists():
            continue
        for _, prefix in RAW_ARTIFACT_SPECS:
            for path in directory.glob(f"{prefix}*.csv"):
                inferred.add(path.stem[len(prefix) :])
    return sorted(inferred)


def choose_raw_artifact(
    stage_name: str,
    prefix: str,
    cluster_type: str,
    artifact_dir: Path,
    overlay_dir: Optional[Path],
) -> Optional[ArtifactSelection]:
    """Choose the raw CSV for one stage and cluster.

    Overlay CSVs take precedence over primary artifacts so recovery outputs can
    be audited before they are synchronized.
    """

    filename = f"{prefix}{cluster_type}.csv"

    if overlay_dir is not None:
        overlay_path = overlay_dir / filename
        if overlay_path.exists():
            return ArtifactSelection(stage_name, cluster_type, overlay_path, "overlay")

    primary_path = artifact_dir / filename
    if primary_path.exists():
        return ArtifactSelection(stage_name, cluster_type, primary_path, "primary")

    return None


def collect_artifact_plan(
    expected_clusters: Sequence[str],
    artifact_dir: Path,
    overlay_dir: Optional[Path],
) -> Tuple[List[ArtifactSelection], List[str]]:
    """Collect available raw artifacts and report missing stage/cluster pairs."""

    selections: List[ArtifactSelection] = []
    missing: List[str] = []

    for cluster_type in expected_clusters:
        for stage_name, prefix in RAW_ARTIFACT_SPECS:
            selection = choose_raw_artifact(
                stage_name=stage_name,
                prefix=prefix,
                cluster_type=cluster_type,
                artifact_dir=artifact_dir,
                overlay_dir=overlay_dir,
            )
            if selection is None:
                missing.append(f"{stage_name}:{cluster_type}")
                continue
            selections.append(selection)

    return selections, missing


def sync_overlay_raws(selections: Iterable[ArtifactSelection], artifact_dir: Path) -> None:
    """Copy selected overlay raw CSVs into the primary artifact directory."""

    ensure_directory(artifact_dir)
    for selection in selections:
        if selection.source_kind != "overlay":
            continue
        destination = artifact_dir / selection.source_path.name
        shutil.copy2(selection.source_path, destination)
        logger.info("Synced overlay raw CSV | %s -> %s", selection.source_path, destination)


def group_by_stage(selections: Sequence[ArtifactSelection]) -> Dict[str, List[ArtifactSelection]]:
    """Group artifact selections by benchmark stage."""

    grouped: Dict[str, List[ArtifactSelection]] = {}
    for selection in selections:
        grouped.setdefault(selection.stage_name, []).append(selection)
    for stage_name in grouped:
        grouped[stage_name] = sorted(grouped[stage_name], key=lambda item: item.cluster_type)
    return grouped


def load_stage_dataframe(selections: Sequence[ArtifactSelection]) -> pd.DataFrame:
    """Load and concatenate all per-cluster raw CSVs for one stage."""

    frames: List[pd.DataFrame] = []
    for selection in selections:
        df = _normalize_benchmark_dataframe(pd.read_csv(selection.source_path))
        logger.info(
            "Loaded raw CSV | stage=%s | cluster=%s | source=%s | rows=%s",
            selection.stage_name,
            selection.cluster_type,
            selection.source_kind,
            len(df),
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def rebuild_public_artifacts(output_dir: Path, public_view_cfg: dict, selections: Sequence[ArtifactSelection]) -> None:
    """Rebuild public summary, table, and figure artifacts from raw CSVs.

    The function intentionally reuses the benchmark report builders so regenerated
    artifacts keep the same schema and reviewer-facing figure conventions.
    """

    grouped = group_by_stage(selections)

    # Load dataframes for Tier 1, Tier 2, predictor microbenchmark, and Tier 3.
    tier1_df = load_stage_dataframe(grouped.get("tier1", []))
    tier2_df = load_stage_dataframe(grouped.get("scaled_search", []))
    inference_df = load_stage_dataframe(grouped.get("predictor_latency_profile", []))
    tier3_df = load_stage_dataframe(grouped.get("synthesized_dispatch_latency", []))

    ensure_directory(output_dir)

    # Rebuild the same public figures used by the benchmark/report pipeline.
    _save_latency_real_plot(tier1_df, output_dir / PUBLIC_REAL_TRACE_PLOT_FILENAME, public_view_cfg)
    _save_scaled_latency_plot(tier2_df, output_dir / PUBLIC_SCALED_TRACE_PLOT_FILENAME, public_view_cfg)
    _save_inference_scaling_plot(inference_df, output_dir / PUBLIC_PREDICTOR_PROFILE_PLOT_FILENAME)
    _save_extrapolation_plot(tier3_df, output_dir / PUBLIC_SYNTH_LATENCY_PLOT_FILENAME, public_view_cfg)
    _save_trigger_rate_plot(tier2_df, output_dir / PUBLIC_TRIGGER_RATE_PLOT_FILENAME, public_view_cfg)

    # Rebuild the reviewer-facing summary table from the normalized dataframes.
    summary_df = _build_latency_summary_table(tier1_df, tier2_df, tier3_df, public_view_cfg)
    _write_latency_summary_table(
        summary_df,
        output_dir / PUBLIC_SUMMARY_FILENAME,
        output_dir / PUBLIC_SUMMARY_TEX_FILENAME,
    )
    logger.info(
        "Rebuild finished | output_dir=%s | real_rows=%s | scaled_rows=%s | synth_rows=%s | inference_rows=%s",
        output_dir,
        len(tier1_df),
        len(tier2_df),
        len(tier3_df),
        len(inference_df),
    )


def main() -> int:
    """CLI entry point: audit completeness, optionally sync overlays, and rebuild."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )

    artifact_dir = args.artifact_dir
    overlay_dir = args.overlay_dir
    output_dir = args.output_dir or artifact_dir

    config = load_config(args.config)
    benchmark_cfg = config.get("evaluation", {}).get("scalability_benchmark", {})
    public_view_cfg = _resolve_public_view_cfg(benchmark_cfg)
    expected_clusters = resolve_expected_clusters(config, artifact_dir, overlay_dir)
    selections, missing = collect_artifact_plan(expected_clusters, artifact_dir, overlay_dir)

    logger.info("Expected clusters: %s", expected_clusters)
    for selection in selections:
        logger.info(
            "Resolved raw CSV | stage=%s | cluster=%s | source=%s | path=%s",
            selection.stage_name,
            selection.cluster_type,
            selection.source_kind,
            selection.source_path,
        )

    if missing:
        logger.error("Missing raw CSVs: %s", missing)
        return 1

    if args.check_only:
        logger.info("Completeness check passed | artifact_dir=%s", artifact_dir)
        return 0

    if args.sync_overlay_raw:
        sync_overlay_raws(selections, artifact_dir)
        # Re-resolve from primary artifacts after overlay synchronization.
        selections, missing = collect_artifact_plan(expected_clusters, artifact_dir, None)
        if missing:
            logger.error("Raw sync finished but some files are still missing: %s", missing)
            return 1

    rebuild_public_artifacts(output_dir=output_dir, public_view_cfg=public_view_cfg, selections=selections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
