"""Train the isolated ``LinearBW`` baseline model.

Function:
- Generate the local LinearBW predictor artifacts under
  ``evaluation/baselines/artifacts/models/<cluster>/``.
- Reuse the repository's existing data-loader and evaluator code without
  editing any files outside this directory.

Design:
- The script mirrors the current BandPilot training flow closely enough that
  the resulting scalers and checkpoint are compatible with
  ``predict_with_model(...)`` and the existing search helpers.
- The only model change is the predictor class itself, which stays linear.

Usage:
- ``python -m evaluation.baselines.train_linear_bw --config ...``
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from data_process.dataloader import get_group_data_loader, get_group_test_loader
from data_process.dataset import get_balanced_train_dataset, get_random_train_dataset
from evaluation.baselines.common import (
    load_cluster_resources,
    load_suite_config,
    resolve_device,
    resolve_linear_model_artifact_dir,
    set_global_seed,
    write_json,
)
from evaluation.baselines.linear_model import LinearBandwidthRegressor
from training.evaluator import evaluate_model
from training.trainer import train_model
from utils.helpers import build_artifact_filename, record_active_num_train_samples


def parse_args() -> argparse.Namespace:
    """Parse the local command line arguments."""
    parser = argparse.ArgumentParser(description="Train LinearBW baseline models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evaluation/baselines/config/baseline_suite.yaml"),
        help="Path to the isolated baseline-suite config",
    )
    return parser.parse_args()


def train_one_cluster(config: Dict[str, object], cluster_type: str, device: torch.device) -> Dict[str, object]:
    """Train and evaluate the linear baseline for one cluster."""
    resources = load_cluster_resources(config, cluster_type)
    artifact_dir = resolve_linear_model_artifact_dir(config, cluster_type)
    num_train_samples = int(config["training"]["num_train_samples"])

    # The local LinearBW model uses the exact same dataset budget as BandPilot
    # so the comparison isolates model capacity instead of sample count.
    gpu_train, bw_train = get_balanced_train_dataset(
        num_samples=num_train_samples,
        total_gpu=resources.total_gpu,
        gpu_bw_dict_list=resources.gpu_bw_dict_list,
        switch_config=resources.switch_config,
        training_data_path=resources.training_data_path,
    )
    train_loader, val_loader = get_group_data_loader(
        gpu_train=gpu_train,
        bw_train=bw_train,
        total_gpu=resources.total_gpu,
        gpu_bw_dict_list=resources.gpu_bw_dict_list,
        switch_config=resources.switch_config,
        training_data_path=resources.training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=num_train_samples,
        batch_size=int(config["training"]["batch_size"]),
    )

    model = LinearBandwidthRegressor()
    model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=int(config["training"]["num_epochs"]),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        patience=int(config["training"]["patience"]),
        lambda_ewc=float(config["training"]["lambda_ewc"]),
    )

    gpu_test, bw_test = get_random_train_dataset(
        num_samples=int(config["training"]["num_test_samples"]),
        total_gpu=resources.total_gpu,
        gpu_bw_dict_list=resources.gpu_bw_dict_list,
        switch_config=resources.switch_config,
        training_data_path=resources.training_data_path,
    )
    test_loader = get_group_test_loader(
        num_samples=len(gpu_test),
        total_gpu=resources.total_gpu,
        gpu_configs=gpu_test,
        bandwidth_targets=bw_test,
        gpu_bw_dict_list=resources.gpu_bw_dict_list,
        switch_config=resources.switch_config,
        training_data_path=resources.training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=num_train_samples,
        batch_size=int(config["training"]["batch_size"]),
    )
    mse, mae = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        total_gpu=resources.total_gpu,
        gpu_bw_dict_list=resources.gpu_bw_dict_list,
        switch_config=resources.switch_config,
        training_data_path=resources.training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=num_train_samples,
    )

    model_path = artifact_dir / build_artifact_filename(
        "bandwidth_predictor",
        num_train_samples,
        ".pth",
    )
    torch.save(model.state_dict(), model_path)
    record_active_num_train_samples(artifact_dir, num_train_samples)

    metrics = {
        "cluster_type": cluster_type,
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "num_train_samples": num_train_samples,
        "num_test_samples": int(config["training"]["num_test_samples"]),
        "mse": float(mse),
        "mae": float(mae),
    }
    write_json(artifact_dir / "linear_bw_metrics.json", metrics)
    return metrics


def main() -> None:
    """Train the linear baseline model for every configured cluster."""
    args = parse_args()
    config = load_suite_config(args.config)
    set_global_seed(int(config["random_seed"]))
    device = resolve_device(config)

    metrics: List[Dict[str, object]] = []
    for cluster_type in config["cluster"]["cluster_types"]:
        metrics.append(train_one_cluster(config, str(cluster_type), device))

    summary_path = Path(config["outputs"]["linear_model_root"]) / "training_summary.json"
    write_json(summary_path, {"clusters": metrics})
    print(f"LinearBW training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
