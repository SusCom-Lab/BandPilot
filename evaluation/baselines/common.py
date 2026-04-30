"""Shared helpers for the isolated baseline suite.

Function:
- Load the local YAML config stored under ``evaluation/baselines``.
- Resolve cluster resources, artifact directories, and read-only external paths.
- Keep all new outputs inside the local baseline workspace.

Design:
- This file centralizes path and runtime conventions so the training script,
  runner, and report builder do not each re-encode the same assumptions.
- External repository files are treated as read-only inputs.
- New artifacts are materialized only under ``evaluation/baselines``.

Usage:
- Import ``load_suite_config`` to read the local config.
- Import ``load_cluster_resources`` to build ``switch_config`` and node-level
  bandwidth dictionaries for one cluster.
- Import the path helpers to resolve BandPilot inputs and local output dirs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import random

import numpy as np
import torch
import yaml

from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files, load_gpu_bw_dict
from algorithms.network_baselines import normalize_network_baseline_name
from evaluation.compare import build_max_bw_cache_filename
from utils.helpers import build_artifact_filename, ensure_directory


@dataclass(frozen=True)
class ClusterResources:
    """Stable read-only resources required to run one cluster configuration."""

    cluster_type: str
    total_gpu: int
    switch_config: SwitchBandwidthConfig
    gpu_bw_dict_list: object
    training_data_path: str
    evaluation_data_path: str


def load_suite_config(config_path: Path) -> Dict[str, Any]:
    """Load the isolated baseline-suite config from YAML."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_global_seed(seed: int) -> None:
    """Set Python / NumPy / Torch seeds for repeatable baseline runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: Dict[str, Any]) -> torch.device:
    """Resolve the desired torch device with a safe CPU fallback."""
    requested = str(config.get("device", "cuda")).strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cluster_resources(config: Dict[str, Any], cluster_type: str) -> ClusterResources:
    """Build switch config, lookup dictionaries, and data paths for one cluster."""
    total_gpu = int(config["cluster"]["total_gpu"])
    bandwidth_dir = Path(config["data"]["bandwidth_dict_dir"])
    file_list = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
    gpu_bw_dict_list = [load_gpu_bw_dict(bandwidth_dir / filename) for filename in file_list]
    switch_config = SwitchBandwidthConfig(
        num_machines=total_gpu // 8,
        cluster_type=cluster_type,
    )
    training_data_path = str(config["data"]["h100_training_data_path"])
    evaluation_data_path = str(
        config["data"].get("h100_evaluation_data_path", training_data_path)
    )
    return ClusterResources(
        cluster_type=cluster_type,
        total_gpu=total_gpu,
        switch_config=switch_config,
        gpu_bw_dict_list=gpu_bw_dict_list,
        training_data_path=training_data_path,
        evaluation_data_path=evaluation_data_path,
    )


def build_run_tag(config: Dict[str, Any]) -> str:
    """Create a deterministic run tag for artifact isolation."""
    suite_cfg = config["suite"]
    k_values = "-".join(str(value) for value in suite_cfg["test_num_values"])
    contention_modes = "-".join(str(mode) for mode in suite_cfg["contention_modes"])
    return (
        f"{config['random_seed']}RS_"
        f"{config['training']['num_train_samples']}TD_"
        f"{config['cluster']['total_gpu']}GPU_"
        f"{suite_cfg['repeat_num']}RN_"
        f"{k_values}K_"
        f"{contention_modes}CM_"
        "baseline_suite"
    )


def resolve_bandpilot_artifact_dir(config: Dict[str, Any], cluster_type: str) -> Path:
    """Return the existing BandPilot artifact directory for one cluster."""
    return Path(config["bandpilot"]["model_root"]) / cluster_type


def resolve_bandpilot_model_path(config: Dict[str, Any], cluster_type: str) -> Path:
    """Return the existing BandPilot model checkpoint path for one cluster."""
    artifact_dir = resolve_bandpilot_artifact_dir(config, cluster_type)
    num_train_samples = int(config["training"]["num_train_samples"])
    return artifact_dir / build_artifact_filename(
        "bandwidth_predictor",
        num_train_samples,
        ".pth",
    )


def resolve_linear_model_artifact_dir(config: Dict[str, Any], cluster_type: str) -> Path:
    """Return the local artifact directory that stores LinearBW model files."""
    artifact_dir = Path(config["outputs"]["linear_model_root"]) / cluster_type
    ensure_directory(artifact_dir)
    return artifact_dir


def resolve_linear_model_path(config: Dict[str, Any], cluster_type: str) -> Path:
    """Return the local LinearBW checkpoint path for one cluster."""
    artifact_dir = resolve_linear_model_artifact_dir(config, cluster_type)
    num_train_samples = int(config["training"]["num_train_samples"])
    return artifact_dir / build_artifact_filename(
        "bandwidth_predictor",
        num_train_samples,
        ".pth",
    )


def resolve_external_max_bw_cache_path(
    config: Dict[str, Any],
    cluster_type: str,
    contention_mode: str,
) -> Path:
    """Resolve the existing read-only max-bw cache path used by compare semantics."""
    max_bw_cfg = config["max_bw_offline"]
    filename = build_max_bw_cache_filename(
        random_seed=int(config["random_seed"]),
        num_train_samples=int(config["training"]["num_train_samples"]),
        total_gpu=int(config["cluster"]["total_gpu"]),
        repeat_num=int(max_bw_cfg["repeat_num"]),
        if_dynamic=bool(max_bw_cfg["if_dynamic"]),
        contention_mode=str(contention_mode),
        search_if_real_data=bool(max_bw_cfg["search_if_real_data"]),
        local_top_k=int(max_bw_cfg["local_top_k"]),
        max_combos_per_distribution=int(max_bw_cfg["max_combos_per_distribution"]),
        max_total_combos=int(max_bw_cfg["max_total_combos"]),
    )
    return Path(config["bandpilot"]["evaluation_root"]) / cluster_type / filename


def prepare_output_layout(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create and return the per-run artifact/report/figure directories."""
    run_tag = build_run_tag(config)
    artifact_dir = Path(config["outputs"]["artifact_root"]) / run_tag
    report_dir = Path(config["outputs"]["report_root"]) / run_tag
    figure_dir = Path(config["outputs"]["figure_root"]) / run_tag
    for path in (artifact_dir, report_dir, figure_dir):
        ensure_directory(path)
    return {
        "run_tag": run_tag,
        "artifact_dir": artifact_dir,
        "report_dir": report_dir,
        "figure_dir": figure_dir,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a small JSON artifact with deterministic formatting."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cluster_algorithm_order(config: Dict[str, Any]) -> List[str]:
    """Return the explicitly configured algorithm order for local suite outputs."""
    return [normalize_network_baseline_name(str(name)) for name in config["suite"]["algorithms"]]
