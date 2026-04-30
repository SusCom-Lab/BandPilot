"""Run the scalability PTS sidecar.

The sidecar scales the cluster template and compares `legacy-PTS` against
`PTS` on a narrow reviewer-facing slice. It writes raw rows,
summaries, figures, metadata, and reports under ignored artifact directories.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

# Make direct script execution work without requiring the caller to export
# PYTHONPATH from the repository root.
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
for candidate_path in [CURRENT_DIR, REPO_ROOT]:
    if str(candidate_path) not in sys.path:
        sys.path.insert(0, str(candidate_path))

from analyze import write_summary_artifacts
from plot import build_all_figures
from report_builder import build_report

from algorithms.search import hu_pts_only_search, legacy_pts_only_search
from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files, load_gpu_bw_dict
from evaluation.scalability import PTS_SIDECAR_CONFIG_PATH
from evaluation.scalability.benchmark import (
    ProfilingPredictor,
    _build_backend_record,
    _build_job_id,
    _build_bank_round_seed,
    _build_background_combo,
    _compute_pod_stats,
    _create_cluster_manager,
    _evaluate_combo_with_manager,
    _run_profiled_search,
    build_scaled_cluster_config,
    estimate_scaled_bandwidth,
    generate_realistic_avail_gpu,
    make_real_cluster_config,
)
from utils.helpers import build_artifact_filename, ensure_directory


logger = logging.getLogger(__name__)
LEGACY_PTS_ALGO = "legacy-PTS"
PTS_ALGO = "PTS"


def parse_args() -> argparse.Namespace:
    """Parse the PTS-sidecar runner CLI."""

    parser = argparse.ArgumentParser(description="Run PTS vs legacy-PTS sidecar")
    parser.add_argument(
        "--config",
        type=Path,
        default=PTS_SIDECAR_CONFIG_PATH,
        help="Path to the PTS-sidecar config",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    """Load a YAML config file."""

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible case streams."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_gpu_bandwidth_dicts(bandwidth_dir: Path, file_list: List[str]) -> list:
    """Load per-node bandwidth dictionaries for a cluster template."""

    return [load_gpu_bw_dict(bandwidth_dir / filename) for filename in file_list]


def _resolve_model_path(model_save_dir: Path, cluster_type: str, model_cfg: dict, num_train_samples: int) -> Path:
    """Resolve the predictor checkpoint used by a cluster template."""

    artifact_dir = model_save_dir / cluster_type
    base_name = "simple_bandwidth_predictor" if model_cfg.get("type") == "simple" else "bandwidth_predictor"
    return artifact_dir / build_artifact_filename(base_name, int(num_train_samples), ".pth")


def _build_cluster_configs(config: dict) -> List[dict]:
    """Build PTS-sidecar cluster templates from the shared config."""

    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    training_cfg = dict(config.get("training", {}))
    cluster_cfg = dict(config.get("cluster", {}))
    sidecar_cfg = dict(config.get("evaluation", {}).get("pts_sidecar", {}))

    total_gpu = int(cluster_cfg["total_gpu"])
    bandwidth_dir = Path(data_cfg["bandwidth_dict_dir"])
    model_save_dir = Path(data_cfg["model_save_dir"])
    device = torch.device(config.get("device", "cuda"))

    selected_clusters = sidecar_cfg.get("cluster_types") or cluster_cfg.get("cluster_types", [])
    cluster_configs: List[dict] = []
    for cluster_type in selected_clusters:
        switch_config = SwitchBandwidthConfig(
            num_machines=total_gpu // 8,
            cluster_type=cluster_type,
        )
        file_list = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
        gpu_bw_dict_list = _load_gpu_bandwidth_dicts(bandwidth_dir, file_list)
        model_path = _resolve_model_path(
            model_save_dir=model_save_dir,
            cluster_type=cluster_type,
            model_cfg=model_cfg,
            num_train_samples=int(training_cfg.get("num_train_samples", 0)),
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found for cluster {cluster_type}: {model_path}"
            )

        cluster_configs.append(
            make_real_cluster_config(
                cluster_type=cluster_type,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                model_path=model_path,
                model_cfg=model_cfg,
                training_data_path=str(data_cfg["h100_training_data_path"]),
                evaluation_data_path=str(
                    data_cfg.get("h100_evaluation_data_path", data_cfg["h100_training_data_path"])
                ),
                artifact_dir=model_save_dir / cluster_type,
                device=device,
                adaptive_runtime_policy=None,
                hu_unit_gate=sidecar_cfg.get("hu_unit_gate"),
            )
        )
    return cluster_configs


def _build_run_tag(sidecar_cfg: dict, random_seed: int) -> str:
    """Build a deterministic run tag from the PTS-sidecar configuration."""

    gpu_tag = "-".join(str(int(value)) for value in sidecar_cfg["gpu_counts"])
    cluster_tag = "2clusters" if len(sidecar_cfg.get("cluster_types", [])) > 1 else "1cluster"
    return (
        f"pts_sidecar_{cluster_tag}_{gpu_tag}gpu_k{int(sidecar_cfg['k_value'])}"
        f"_r{int(sidecar_cfg['repeat_num'])}_{sidecar_cfg['contention_mode']}"
        f"_a{float(sidecar_cfg['avail_ratio']):.1f}_f{float(sidecar_cfg['inter_pod_factor']):.1f}"
        f"_rs{int(random_seed)}"
    )


def _write_run_metadata(metadata_path: Path, metadata: Dict[str, Any]) -> None:
    """Write run metadata as pretty JSON.

    Metadata is updated during the run so interrupted executions still expose
    their status, row count, and configuration trace.
    """

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_raw_records(raw_csv_path: Path, pending_rows: List[Dict[str, Any]]) -> int:
    """Append pending raw rows to the CSV and return the number written.

    The runner flushes in small batches so long runs preserve partial progress
    without rewriting completed case rows.
    """

    if not pending_rows:
        return 0
    pending_df = pd.DataFrame(pending_rows)
    write_header = not raw_csv_path.exists()
    pending_df.to_csv(raw_csv_path, mode="a", header=write_header, index=False)
    return int(len(pending_df))


def _run_single_algorithm(
    *,
    algorithm: str,
    algo_fn: Callable[..., Any],
    total_gpu: int,
    avail_gpu: np.ndarray,
    gpu_need: int,
    scaled_cfg: dict,
    search_manager,
    real_manager,
    probe_job_id: int,
    aggressive: bool = False,
) -> Dict[str, Any]:
    """Run one algorithm for one case and collect raw-row metrics."""

    def _wrapped_search():
        search_manager.set_job_context(probe_job_id)
        try:
            return algo_fn(
                total_gpu,
                avail_gpu,
                model=None,
                gpu_need=gpu_need,
                total_gpu=total_gpu,
                gpu_bw_dict_list=scaled_cfg["gpu_bw_dict_list"],
                switch_config=scaled_cfg["switch_config"],
                training_data_path=scaled_cfg["training_data_path"],
                device=torch.device("cpu"),
                artifact_dir=Path("."),
                if_real_data=False,
                cluster_manager=search_manager,
                aggressive=bool(aggressive),
                return_metadata=True,
            )
        finally:
            search_manager.clear_job_context()

    (combo, search_meta), elapsed, _, _, contention_time = _run_profiled_search(_wrapped_search)
    final_bw = _evaluate_combo_with_manager(real_manager, combo, probe_job_id)
    num_active_pods, cross_pod_ratio = _compute_pod_stats(
        combo,
        total_gpu=total_gpu,
        pod_size=scaled_cfg["pod_size"],
    )
    return {
        "combo": combo,
        "search_meta": search_meta,
        "row_extra": {
            "measured_wall_time_s": float(elapsed),
            "predictor_time_s": float(search_manager.bandwidth_predictor.total_time),
            "predictor_calls": int(search_manager.bandwidth_predictor.call_count),
            "contention_time_s": float(contention_time),
            "final_bw": float(final_bw),
            "num_active_pods": int(num_active_pods),
            "cross_pod_ratio": float(cross_pod_ratio),
        },
    }


def run_pts_sidecar(config: dict, config_path: Path = PTS_SIDECAR_CONFIG_PATH) -> Dict[str, Path]:
    """Run the full PTS sidecar and build raw, summary, figure, and report artifacts."""

    sidecar_cfg = dict(config.get("evaluation", {}).get("pts_sidecar", {}))
    random_seed = int(config.get("random_seed", 1111))
    _set_seed(random_seed)

    output_root = Path(sidecar_cfg["output_dir"])
    ensure_directory(output_root)
    run_tag = _build_run_tag(sidecar_cfg, random_seed)
    run_dir = output_root / run_tag
    raw_dir = run_dir / "raw"
    summary_dir = run_dir / "summary"
    figure_dir = run_dir / "figures"
    report_dir = run_dir / "reports"
    for directory in [run_dir, raw_dir, summary_dir, figure_dir, report_dir]:
        ensure_directory(directory)

    cluster_configs = _build_cluster_configs(config)
    raw_rows: List[Dict[str, Any]] = []
    pending_rows: List[Dict[str, Any]] = []
    raw_csv_path = raw_dir / "raw_rows.csv"
    save_every_n_records = max(1, int(sidecar_cfg.get("save_every_n_records", 1)))
    if raw_csv_path.exists():
        raw_csv_path.unlink()

    metadata = {
        "run_tag": run_tag,
        "cluster_types": [cfg["cluster_type"] for cfg in cluster_configs],
        "gpu_counts": [int(value) for value in sidecar_cfg["gpu_counts"]],
        "k_value": int(sidecar_cfg["k_value"]),
        "repeat_num": int(sidecar_cfg["repeat_num"]),
        "contention_mode": str(sidecar_cfg["contention_mode"]),
        "avail_ratio": float(sidecar_cfg["avail_ratio"]),
        "inter_pod_factor": float(sidecar_cfg["inter_pod_factor"]),
        "hu_unit_gate": dict(sidecar_cfg.get("hu_unit_gate", {})),
        "evidence_type": "simulated",
        "config_path": str(config_path),
        "python_version": platform.python_version(),
        "device": str(config.get("device", "cuda")),
        "raw_row_count": 0,
        "save_every_n_records": int(save_every_n_records),
        "status": "running",
    }
    metadata_path = run_dir / "run_metadata.json"
    _write_run_metadata(metadata_path, metadata)

    logger.info(
        "PTS sidecar start | run_tag=%s | clusters=%s | gpu_counts=%s | k=%s | repeat=%s",
        run_tag,
        [cfg["cluster_type"] for cfg in cluster_configs],
        sidecar_cfg["gpu_counts"],
        sidecar_cfg["k_value"],
        sidecar_cfg["repeat_num"],
    )

    for cluster_cfg in cluster_configs:
        cluster_type = str(cluster_cfg["cluster_type"])
        for total_gpu in [int(value) for value in sidecar_cfg["gpu_counts"]]:
            scaled_cfg = build_scaled_cluster_config(
                total_gpu=total_gpu,
                cluster_template=cluster_cfg,
                inter_pod_factor=float(sidecar_cfg["inter_pod_factor"]),
            )

            # The sidecar reuses the benchmark scaled-bandwidth estimator so
            # its PTS-vs-legacy-PTS slice remains consistent with Tier 2.
            def predictor_fn(combo: np.ndarray, *, _scaled_cfg=scaled_cfg) -> float:
                return estimate_scaled_bandwidth(
                    gpu_config=combo,
                    total_gpu=_scaled_cfg["total_gpu"],
                    pod_size=_scaled_cfg["pod_size"],
                    gpu_bw_dict_list=_scaled_cfg["gpu_bw_dict_list"],
                    pod_bw_lookup=_scaled_cfg["pod_bw_lookup"],
                    inter_pod_factor=_scaled_cfg["inter_pod_factor"],
                )

            for repeat_idx in range(int(sidecar_cfg["repeat_num"])):
                scenario_group_id = (
                    f"{cluster_type}:pts_sidecar:{total_gpu}:"
                    f"a{float(sidecar_cfg['avail_ratio']):.2f}:f{float(sidecar_cfg['inter_pod_factor']):.2f}:"
                    f"m{sidecar_cfg['contention_mode']}"
                )
                seed = _build_bank_round_seed(
                    random_seed + total_gpu * 13,
                    scenario_group_id,
                    repeat_idx,
                )
                target_avail = max(
                    int(sidecar_cfg["k_value"]),
                    int(round(total_gpu * float(sidecar_cfg["avail_ratio"]))),
                )
                avail_gpu = generate_realistic_avail_gpu(
                    total_gpu=total_gpu,
                    target_avail=target_avail,
                    mode=str(sidecar_cfg.get("availability_mode", "mixed")),
                    seed=seed,
                )
                if len(avail_gpu) < int(sidecar_cfg["k_value"]):
                    continue

                background_combo = _build_background_combo(total_gpu, avail_gpu)
                background_gpu = np.where(background_combo == 1)[0].astype(int).tolist()
                occupancy_seed = int(seed + int(sidecar_cfg["k_value"]) * 97)
                probe_job_id = _build_job_id(int(sidecar_cfg["k_value"]), repeat_idx)

                case_context = {
                    "cluster_type": cluster_type,
                    "total_gpu": int(total_gpu),
                    "k": int(sidecar_cfg["k_value"]),
                    "avail_ratio": float(sidecar_cfg["avail_ratio"]),
                    "contention_mode": str(sidecar_cfg["contention_mode"]),
                    "inter_pod_factor": float(sidecar_cfg["inter_pod_factor"]),
                    "repeat_idx": int(repeat_idx),
                    "seed": int(seed),
                    "avail_gpu_count": int(len(avail_gpu)),
                    "avail_signature": ",".join(str(int(value)) for value in avail_gpu.tolist()),
                    "background_signature": ",".join(str(int(value)) for value in background_gpu),
                    "background_gpu_count": int(len(background_gpu)),
                    "probe_job_id": int(probe_job_id),
                    "scenario_group_id": scenario_group_id,
                    "bank_scope": "pts_sidecar_scale_slice",
                }

                for algorithm, algo_fn in [
                    (LEGACY_PTS_ALGO, legacy_pts_only_search),
                    (PTS_ALGO, hu_pts_only_search),
                ]:
                    search_predictor = ProfilingPredictor(predictor_fn)
                    real_predictor = ProfilingPredictor(predictor_fn)
                    search_manager = _create_cluster_manager(
                        total_gpu=total_gpu,
                        predictor=search_predictor,
                        contention_mode=str(sidecar_cfg["contention_mode"]),
                        background_combo=background_combo,
                        occupancy_seed=occupancy_seed,
                    )
                    real_manager = _create_cluster_manager(
                        total_gpu=total_gpu,
                        predictor=real_predictor,
                        contention_mode=str(sidecar_cfg["contention_mode"]),
                        background_combo=background_combo,
                        occupancy_seed=occupancy_seed,
                    )
                    result = _run_single_algorithm(
                        algorithm=algorithm,
                        algo_fn=algo_fn,
                        total_gpu=total_gpu,
                        avail_gpu=avail_gpu,
                        gpu_need=int(sidecar_cfg["k_value"]),
                        scaled_cfg=scaled_cfg,
                        search_manager=search_manager,
                        real_manager=real_manager,
                        probe_job_id=probe_job_id,
                        aggressive=bool(
                            algorithm == PTS_ALGO
                            and scaled_cfg.get("hu_unit_gate", {}).get("aggressive", False)
                        ),
                    )
                    row = _build_backend_record(
                        case_context=case_context,
                        algorithm=algorithm,
                        final_bw=float(result["row_extra"]["final_bw"]),
                        search_meta=result["search_meta"],
                        combo=result["combo"],
                        measured_wall_time_s=float(result["row_extra"]["measured_wall_time_s"]),
                        predictor_time_s=float(result["row_extra"]["predictor_time_s"]),
                        predictor_calls=int(result["row_extra"]["predictor_calls"]),
                        contention_time_s=float(result["row_extra"]["contention_time_s"]),
                        latency_evidence_kind="scaled_trace",
                        bandwidth_evidence_kind="scaled_estimated",
                        evidence_type="simulated",
                        extra_fields={
                            "num_active_pods": int(result["row_extra"]["num_active_pods"]),
                            "cross_pod_ratio": float(result["row_extra"]["cross_pod_ratio"]),
                            "pts_sidecar_label": "PTS_vs_legacy_PTS_speedup",
                        },
                    )
                    raw_rows.append(row)
                    pending_rows.append(row)
                    if len(pending_rows) >= save_every_n_records:
                        written = _append_raw_records(raw_csv_path, pending_rows)
                        metadata["raw_row_count"] = int(metadata["raw_row_count"]) + int(written)
                        _write_run_metadata(metadata_path, metadata)
                        logger.info(
                            "PTS sidecar raw flush | path=%s | new_rows=%s | total_rows=%s",
                            raw_csv_path,
                            written,
                            metadata["raw_row_count"],
                        )
                        pending_rows = []

    if pending_rows:
        written = _append_raw_records(raw_csv_path, pending_rows)
        metadata["raw_row_count"] = int(metadata["raw_row_count"]) + int(written)
        _write_run_metadata(metadata_path, metadata)
        logger.info(
            "PTS sidecar raw flush | path=%s | new_rows=%s | total_rows=%s",
            raw_csv_path,
            written,
            metadata["raw_row_count"],
        )

    raw_df = pd.read_csv(raw_csv_path).sort_values(
        ["cluster_type", "total_gpu", "repeat_idx", "algorithm"]
    ).reset_index(drop=True)
    raw_df.to_csv(raw_csv_path, index=False)

    analysis_paths = write_summary_artifacts(raw_df=raw_df, output_dir=summary_dir)
    summary_df = pd.read_csv(analysis_paths["summary_csv"])
    breakdown_df = pd.read_csv(analysis_paths["breakdown_csv"])
    figure_paths = build_all_figures(summary_df=summary_df, breakdown_df=breakdown_df, figure_dir=figure_dir)
    report_paths = build_report(
        summary_df=summary_df,
        breakdown_df=breakdown_df,
        metadata=metadata,
        figure_dir=figure_dir,
        report_dir=report_dir,
    )

    # Maintain a stable latest manifest while preserving run-tagged artifacts.
    latest_dir = output_root / "latest"
    ensure_directory(latest_dir)
    latest_manifest = {
        "run_tag": run_tag,
        "raw_csv": str(raw_csv_path),
        **{key: str(value) for key, value in analysis_paths.items()},
        **{key: str(value) for key, value in figure_paths.items()},
        **{key: str(value) for key, value in report_paths.items()},
        "metadata_json": str(metadata_path),
    }
    (latest_dir / "latest_manifest.json").write_text(
        json.dumps(latest_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata["status"] = "completed"
    metadata["raw_row_count"] = int(len(raw_df))
    _write_run_metadata(metadata_path, metadata)

    logger.info(
        "PTS sidecar done | raw_rows=%s | summary_rows=%s | report=%s",
        len(raw_df),
        len(summary_df),
        report_paths["latest_report_md"],
    )
    return {
        "run_dir": run_dir,
        "raw_csv": raw_csv_path,
        "summary_csv": analysis_paths["summary_csv"],
        "breakdown_csv": analysis_paths["breakdown_csv"],
        "metadata_json": metadata_path,
        "report_md": report_paths["report_md"],
        "latest_report_md": report_paths["latest_report_md"],
        "speedup_png": figure_paths["speedup_png"],
        "breakdown_png": figure_paths["breakdown_png"],
    }


def main() -> None:
    """CLI entry point for the full PTS-sidecar run."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    config = _load_config(args.config)
    run_pts_sidecar(config=config, config_path=args.config)


if __name__ == "__main__":
    main()
