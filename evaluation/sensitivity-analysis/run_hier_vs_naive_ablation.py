#!/usr/bin/env python3
"""Het-4Mix hierarchical versus naive predictor ablation.

The ablation compares `BandwidthPredictor` against `SimpleBandwidthPredictor`
on Het-4Mix for selected sample sizes. It reuses the public training and
evaluation code paths and writes regenerated metrics, manifests, tables, and
prediction dumps under ignored artifact directories.

Examples:
    conda run -n gpu_dp_opt python evaluation/sensitivity-analysis/run_hier_vs_naive_ablation.py

    conda run -n gpu_dp_opt python evaluation/sensitivity-analysis/run_hier_vs_naive_ablation.py \
        --config config/default_config.yaml \
        --cluster Het-4Mix \
        --sample-sizes 100,250,1000 \
        --output-dir evaluation/sensitivity-analysis/artifacts/hier_vs_naive_het4mix_defaultcfg_rs1111_100-250-1000

Notes:
- This script produces `simulated` predictor-level evidence, not deployment measurements.
- The generated `.tex` table is a convenience artifact and should be reviewed before manuscript use.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Direct execution from `evaluation/sensitivity-analysis/` needs the repository
# root on `sys.path` so imports from `core/`, `models/`, and `training/` work.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files, load_gpu_bw_dict
from data_process.dataloader import (
    get_group_data_loader,
    get_group_test_loader,
    get_simple_group_data_loader,
    get_simple_group_test_loader,
)
from data_process.dataset import get_balanced_train_dataset, get_random_train_dataset
from models.bandwidth_predictor import BandwidthPredictor
from models.simple_predictor import SimpleBandwidthPredictor
from training.evaluator import compute_extra_metrics
from training.trainer import train_model, train_simple_model
from utils.helpers import build_artifact_filename, ensure_directory, record_active_num_train_samples


# Stable reviewer-facing labels used in CSV, LaTeX, and manifest outputs.
HIERARCHICAL_LABEL = "Hierarchical"
NAIVE_LABEL = "Naive"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the predictor ablation."""
    parser = argparse.ArgumentParser(
        description="Fair Het-4Mix hierarchical-vs-naive predictor ablation"
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_config.yaml"))
    parser.add_argument("--cluster", type=str, default="Het-4Mix")
    parser.add_argument("--sample-sizes", type=str, default="100,250,1000")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "evaluation/sensitivity-analysis/artifacts/"
            "hier_vs_naive_het4mix_defaultcfg_rs1111_100-250-1000"
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load a YAML config file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible data and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_base_random_seed(config: dict) -> int:
    """Resolve a single base random seed from scalar or list config values."""
    random_seed_cfg = config.get("random_seed", 123)
    if isinstance(random_seed_cfg, list):
        if not random_seed_cfg:
            raise ValueError("config.random_seed is an empty list.")
        return int(random_seed_cfg[0])
    return int(random_seed_cfg)


def resolve_cluster_resources(config: dict, cluster_type: str) -> Dict[str, object]:
    """Resolve cluster bandwidth dictionaries, topology settings, and paths."""
    data_cfg = config["data"]
    cluster_cfg = config["cluster"]
    total_gpu = int(cluster_cfg["total_gpu"])

    bandwidth_dir = Path(data_cfg["bandwidth_dict_dir"])
    training_data_path = str(data_cfg["h100_training_data_path"])

    switch_config = SwitchBandwidthConfig(
        num_machines=total_gpu // 8,
        cluster_type=cluster_type,
    )
    dict_files = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
    if not dict_files:
        raise RuntimeError(f"No bandwidth dictionaries found for cluster_type={cluster_type}.")
    gpu_bw_dict_list = [load_gpu_bw_dict(bandwidth_dir / name) for name in dict_files]

    return {
        "total_gpu": total_gpu,
        "training_data_path": training_data_path,
        "switch_config": switch_config,
        "gpu_bw_dict_list": gpu_bw_dict_list,
        "dict_files": dict_files,
    }


def build_seed_plan(base_seed: int, sample_sizes: List[int]) -> Dict[str, object]:
    """Build deterministic seeds for shared tests, datasets, and model training."""
    return {
        "base_random_seed": int(base_seed),
        "shared_test_seed": int(base_seed + 40000),
        "sample_size_plan": {
            str(sample_size): {
                "train_dataset_seed": int(base_seed + sample_size),
                "hierarchical_train_seed": int(base_seed + 20000 + sample_size),
                "naive_train_seed": int(base_seed + 30000 + sample_size),
            }
            for sample_size in sample_sizes
        },
    }


def hash_dataset(gpu_configs: np.ndarray, bandwidths: np.ndarray) -> str:
    """Hash GPU configs and targets for reproducibility manifests."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(gpu_configs).tobytes())
    hasher.update(np.ascontiguousarray(bandwidths).astype(np.float64).tobytes())
    return hasher.hexdigest()


def summarize_dataset(gpu_configs: np.ndarray, bandwidths: np.ndarray) -> Dict[str, object]:
    """Summarize dataset shape and value ranges for the manifest."""
    active_counts = np.sum(gpu_configs, axis=1)
    return {
        "sample_count": int(len(gpu_configs)),
        "hash_sha256": hash_dataset(gpu_configs, bandwidths),
        "active_gpu_count_min": int(np.min(active_counts)),
        "active_gpu_count_max": int(np.max(active_counts)),
        "active_gpu_count_mean": float(np.mean(active_counts)),
        "bandwidth_min": float(np.min(bandwidths)),
        "bandwidth_max": float(np.max(bandwidths)),
        "bandwidth_mean": float(np.mean(bandwidths)),
    }


def save_prediction_trace(
    path: Path,
    preds: np.ndarray,
    targets: np.ndarray,
    metadata: Dict[str, object],
) -> None:
    """Save predictions, targets, and metadata as a compressed trace."""
    np.savez_compressed(
        path,
        preds=np.asarray(preds, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def collect_full_model_predictions(
    model: BandwidthPredictor,
    *,
    artifact_dir: Path,
    num_train_samples: int,
    test_gpu_configs: np.ndarray,
    test_bandwidths: np.ndarray,
    total_gpu: int,
    gpu_bw_dict_list: list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect inverse-scaled predictions from the hierarchical predictor."""
    test_loader = get_group_test_loader(
        num_samples=len(test_gpu_configs),
        total_gpu=total_gpu,
        gpu_configs=test_gpu_configs,
        bandwidth_targets=test_bandwidths,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=num_train_samples,
    )

    with (artifact_dir / build_artifact_filename("y_scaler", num_train_samples, ".pkl")).open(
        "rb"
    ) as handle:
        y_scaler = pickle.load(handle)

    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    with torch.no_grad():
        for x_bws, x_node_counts, x_total_counts, y_batch in test_loader:
            x_bws = x_bws.to(device)
            x_node_counts = x_node_counts.to(device)
            x_total_counts = x_total_counts.to(device)
            outputs = model(x_bws, x_node_counts, x_total_counts)["final_bandwidth"].view(-1)

            pred_np = outputs.cpu().numpy().reshape(-1, 1)
            target_np = y_batch.cpu().numpy().reshape(-1, 1)
            all_preds.append(y_scaler.inverse_transform(pred_np).flatten())
            all_targets.append(y_scaler.inverse_transform(target_np).flatten())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def collect_simple_model_predictions(
    model: SimpleBandwidthPredictor,
    *,
    artifact_dir: Path,
    num_train_samples: int,
    test_gpu_configs: np.ndarray,
    test_bandwidths: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect inverse-scaled predictions from the naive/simple predictor."""
    test_loader = get_simple_group_test_loader(
        num_samples=len(test_gpu_configs),
        gpu_configs=test_gpu_configs,
        targets=test_bandwidths,
        artifact_dir=artifact_dir,
        num_train_samples=num_train_samples,
    )

    with (
        artifact_dir
        / build_artifact_filename("simple_y_scaler", num_train_samples, ".pkl")
    ).open("rb") as handle:
        y_scaler = pickle.load(handle)

    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    with torch.no_grad():
        for x_bws, y_batch in test_loader:
            x_bws = x_bws.to(device)
            outputs = model(x_bws)["final_bandwidth"].view(-1)

            pred_np = outputs.cpu().numpy().reshape(-1, 1)
            target_np = y_batch.cpu().numpy().reshape(-1, 1)
            all_preds.append(y_scaler.inverse_transform(pred_np).flatten())
            all_targets.append(y_scaler.inverse_transform(target_np).flatten())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def run_hierarchical_trial(
    *,
    sample_size: int,
    train_seed: int,
    train_gpu_configs: np.ndarray,
    train_bandwidths: np.ndarray,
    test_gpu_configs: np.ndarray,
    test_bandwidths: np.ndarray,
    config: dict,
    cluster_resource: Dict[str, object],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    """Train and evaluate one hierarchical-predictor trial."""
    set_seed(train_seed)

    training_cfg = config["training"]
    model_cfg = config["model"]
    trial_dir = output_dir / "models" / f"n{sample_size}_hierarchical"
    ensure_directory(trial_dir)

    # The full model consumes part bandwidths, node counts, and total counts.
    train_loader, val_loader = get_group_data_loader(
        train_gpu_configs,
        train_bandwidths,
        int(cluster_resource["total_gpu"]),
        cluster_resource["gpu_bw_dict_list"],
        cluster_resource["switch_config"],
        str(cluster_resource["training_data_path"]),
        artifact_dir=trial_dir,
        num_train_samples=sample_size,
        batch_size=int(training_cfg["batch_size"]),
    )

    model = BandwidthPredictor(
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
    )
    model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=int(training_cfg["num_epochs"]),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        patience=int(training_cfg["patience"]),
        lambda_ewc=float(training_cfg["lambda_ewc"]),
    )

    model_path = trial_dir / build_artifact_filename("bandwidth_predictor", sample_size, ".pth")
    torch.save(model.state_dict(), model_path)
    record_active_num_train_samples(trial_dir, sample_size)

    preds, targets = collect_full_model_predictions(
        model,
        artifact_dir=trial_dir,
        num_train_samples=sample_size,
        test_gpu_configs=test_gpu_configs,
        test_bandwidths=test_bandwidths,
        total_gpu=int(cluster_resource["total_gpu"]),
        gpu_bw_dict_list=cluster_resource["gpu_bw_dict_list"],
        switch_config=cluster_resource["switch_config"],
        training_data_path=str(cluster_resource["training_data_path"]),
        device=device,
    )
    metrics = compute_extra_metrics(preds, targets)

    save_prediction_trace(
        trial_dir / "predictions_hierarchical.npz",
        preds,
        targets,
        {
            "model_variant": HIERARCHICAL_LABEL,
            "sample_size": int(sample_size),
            "train_seed": int(train_seed),
        },
    )

    return {
        "model_variant": HIERARCHICAL_LABEL,
        "sample_size": int(sample_size),
        "train_seed": int(train_seed),
        "model_path": str(model_path),
        **metrics,
    }


def run_naive_trial(
    *,
    sample_size: int,
    train_seed: int,
    train_gpu_configs: np.ndarray,
    train_bandwidths: np.ndarray,
    test_gpu_configs: np.ndarray,
    test_bandwidths: np.ndarray,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    """Train and evaluate one naive/simple-predictor trial."""
    set_seed(train_seed)

    training_cfg = config["training"]
    model_cfg = config["model"]
    trial_dir = output_dir / "models" / f"n{sample_size}_naive"
    ensure_directory(trial_dir)

    # The naive baseline consumes only the flat 32-GPU mask via the simple loader.
    train_loader, val_loader = get_simple_group_data_loader(
        train_gpu_configs,
        train_bandwidths,
        artifact_dir=trial_dir,
        num_train_samples=sample_size,
        batch_size=int(training_cfg["batch_size"]),
    )

    model = SimpleBandwidthPredictor(
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
    )
    model, _ = train_simple_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=int(training_cfg["num_epochs"]),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        patience=int(training_cfg["patience"]),
    )

    model_path = trial_dir / build_artifact_filename(
        "simple_bandwidth_predictor", sample_size, ".pth"
    )
    torch.save(model.state_dict(), model_path)
    record_active_num_train_samples(trial_dir, sample_size)

    preds, targets = collect_simple_model_predictions(
        model,
        artifact_dir=trial_dir,
        num_train_samples=sample_size,
        test_gpu_configs=test_gpu_configs,
        test_bandwidths=test_bandwidths,
        device=device,
    )
    metrics = compute_extra_metrics(preds, targets)

    save_prediction_trace(
        trial_dir / "predictions_naive.npz",
        preds,
        targets,
        {
            "model_variant": NAIVE_LABEL,
            "sample_size": int(sample_size),
            "train_seed": int(train_seed),
        },
    )

    return {
        "model_variant": NAIVE_LABEL,
        "sample_size": int(sample_size),
        "train_seed": int(train_seed),
        "model_path": str(model_path),
        **metrics,
    }


def bold_if_best(value: str, is_best: bool) -> str:
    """Wrap a LaTeX table value in bold when it is best."""
    if is_best:
        return f"\\textbf{{{value}}}"
    return value


def build_latex_table(metrics_df: pd.DataFrame, cluster_type: str) -> str:
    """Build the manuscript-style LaTeX table from ablation metrics."""
    ordered_sizes = sorted(metrics_df["sample_size"].unique().tolist())
    pivot = metrics_df.pivot(index="sample_size", columns="model_variant")

    cluster_label = cluster_type
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.05}",
        (
            "  \\caption{Predictor ablation on "
            f"{cluster_label} cluster. The hierarchical structure is markedly more "
            "sample-efficient than a naive monolithic Transformer.}"
        ),
        r"  \label{tab:ablation-hier-vs-naive}",
        "",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}c cc cc}",
        r"    \toprule",
        r"    \multirow{2}{*}{Data} & \multicolumn{2}{c}{Hierarchical} & \multicolumn{2}{c}{Naive} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"     & $R^2\uparrow$ & MAPE(\%)$\downarrow$ & $R^2\uparrow$ & MAPE(\%)$\downarrow$ \\",
        r"    \midrule",
    ]

    for sample_size in ordered_sizes:
        hier_r2 = float(pivot.loc[sample_size, ("r2", HIERARCHICAL_LABEL)])
        hier_mape = float(pivot.loc[sample_size, ("mape_percent", HIERARCHICAL_LABEL)])
        naive_r2 = float(pivot.loc[sample_size, ("r2", NAIVE_LABEL)])
        naive_mape = float(pivot.loc[sample_size, ("mape_percent", NAIVE_LABEL)])

        hier_r2_s = bold_if_best(f"{hier_r2:.2f}", hier_r2 >= naive_r2)
        naive_r2_s = bold_if_best(f"{naive_r2:.2f}", naive_r2 > hier_r2)
        hier_mape_s = bold_if_best(f"{hier_mape:.2f}", hier_mape <= naive_mape)
        naive_mape_s = bold_if_best(f"{naive_mape:.2f}", naive_mape < hier_mape)

        lines.append(
            f"    {sample_size:<4} & {hier_r2_s} & {hier_mape_s}  & {naive_r2_s} & {naive_mape_s} \\\\"
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular*}",
            r"  % \vspace{-6pt}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the full hierarchical-versus-naive ablation workflow."""
    args = parse_args()
    config = load_config(args.config)
    cluster_type = args.cluster
    sample_sizes = [int(item.strip()) for item in args.sample_sizes.split(",") if item.strip()]
    output_dir = args.output_dir
    ensure_directory(output_dir)

    device_str = args.device or config.get("device", "cuda")
    device = torch.device(device_str)
    cluster_resource = resolve_cluster_resources(config, cluster_type)
    base_seed = resolve_base_random_seed(config)
    seed_plan = build_seed_plan(base_seed, sample_sizes)

    # Use one shared test set for every sample budget.
    set_seed(int(seed_plan["shared_test_seed"]))
    shared_test_gpu, shared_test_bw = get_random_train_dataset(
        num_samples=int(config["training"]["num_test_samples"]),
        total_gpu=int(cluster_resource["total_gpu"]),
        gpu_bw_dict_list=cluster_resource["gpu_bw_dict_list"],
        switch_config=cluster_resource["switch_config"],
        training_data_path=str(cluster_resource["training_data_path"]),
    )

    dataset_manifest: Dict[str, object] = {
        "cluster_type": cluster_type,
        "shared_test_set": {
            "seed": int(seed_plan["shared_test_seed"]),
            **summarize_dataset(shared_test_gpu, shared_test_bw),
        },
        "train_sets": {},
    }

    records: List[Dict[str, object]] = []
    for sample_size in sample_sizes:
        sample_seed_plan = seed_plan["sample_size_plan"][str(sample_size)]

        # Use the same training subset for hierarchical and naive models.
        set_seed(int(sample_seed_plan["train_dataset_seed"]))
        train_gpu, train_bw = get_balanced_train_dataset(
            num_samples=int(sample_size),
            total_gpu=int(cluster_resource["total_gpu"]),
            gpu_bw_dict_list=cluster_resource["gpu_bw_dict_list"],
            switch_config=cluster_resource["switch_config"],
            training_data_path=str(cluster_resource["training_data_path"]),
        )

        dataset_manifest["train_sets"][str(sample_size)] = {
            "seed": int(sample_seed_plan["train_dataset_seed"]),
            **summarize_dataset(train_gpu, train_bw),
        }

        hierarchical_result = run_hierarchical_trial(
            sample_size=sample_size,
            train_seed=int(sample_seed_plan["hierarchical_train_seed"]),
            train_gpu_configs=train_gpu,
            train_bandwidths=train_bw,
            test_gpu_configs=shared_test_gpu,
            test_bandwidths=shared_test_bw,
            config=config,
            cluster_resource=cluster_resource,
            device=device,
            output_dir=output_dir,
        )
        hierarchical_result["cluster_type"] = cluster_type
        records.append(hierarchical_result)

        naive_result = run_naive_trial(
            sample_size=sample_size,
            train_seed=int(sample_seed_plan["naive_train_seed"]),
            train_gpu_configs=train_gpu,
            train_bandwidths=train_bw,
            test_gpu_configs=shared_test_gpu,
            test_bandwidths=shared_test_bw,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        naive_result["cluster_type"] = cluster_type
        records.append(naive_result)

    metrics_df = pd.DataFrame(records).sort_values(
        by=["sample_size", "model_variant"]
    ).reset_index(drop=True)

    # Persist metrics and manifests for traceability.
    metrics_csv_path = output_dir / "ablation_metrics.csv"
    metrics_json_path = output_dir / "ablation_metrics.json"
    metrics_df.to_csv(metrics_csv_path, index=False)
    metrics_json_path.write_text(
        metrics_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    dataset_manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_manifest = {
        "experiment_name": "hierarchical_vs_naive_predictor_ablation",
        "cluster_type": cluster_type,
        "evidence_kind": "simulated",
        "config_path": str(args.config),
        "device": device_str,
        "sample_sizes": sample_sizes,
        "seed_plan": seed_plan,
        "config_snapshot": {
            "model": copy.deepcopy(config["model"]),
            "training": {
                "batch_size": int(config["training"]["batch_size"]),
                "num_epochs": int(config["training"]["num_epochs"]),
                "learning_rate": float(config["training"]["learning_rate"]),
                "weight_decay": float(config["training"]["weight_decay"]),
                "patience": int(config["training"]["patience"]),
                "lambda_ewc": float(config["training"]["lambda_ewc"]),
                "num_test_samples": int(config["training"]["num_test_samples"]),
            },
            "cluster": {
                "total_gpu": int(config["cluster"]["total_gpu"]),
                "cluster_type": cluster_type,
                "dict_files": cluster_resource["dict_files"],
            },
        },
        "fairness_contract": {
            "shared_test_set": True,
            "shared_train_set_per_sample_size": True,
            "shared_metric_function": "training.evaluator.compute_extra_metrics",
            "hierarchical_model": "models.bandwidth_predictor.BandwidthPredictor",
            "naive_model": "models.simple_predictor.SimpleBandwidthPredictor",
        },
        "output_files": {
            "metrics_csv": str(metrics_csv_path),
            "metrics_json": str(metrics_json_path),
            "dataset_manifest": str(dataset_manifest_path),
            "latex_table": str(output_dir / "ablation_table_het4mix.tex"),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    latex_table = build_latex_table(metrics_df, cluster_type)
    latex_path = output_dir / "ablation_table_het4mix.tex"
    latex_path.write_text(latex_table + "\n", encoding="utf-8")

    print("=" * 72)
    print("Hierarchical vs. Naive predictor ablation finished")
    print(f"Cluster: {cluster_type}")
    print(f"Output dir: {output_dir}")
    print("-" * 72)
    print(metrics_df[["sample_size", "model_variant", "r2", "mape_percent", "rmse"]].to_string(index=False))
    print("-" * 72)
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"LaTeX table: {latex_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
