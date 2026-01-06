"""Variable-length capability study for the Transformer model.

This script:
1. Runs base training with 16-GPU / 24-GPU configs by reusing `trainer.model_train_pipeline`.
2. Finetunes on 32-GPU real data after excluding GPU configs seen in the 16/24 datasets.
3. Evaluates on the remaining 32-GPU samples and reports MAE/MAPE/MSE/RMSE/R².
4. Saves prediction-vs-target tables, model weights, and metric summaries for reproducibility.
"""
from __future__ import annotations

import argparse
import ast
import copy
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Ensure the script can run directly by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files, load_gpu_bw_dict
from data_process.dataloader import get_group_data_loader, get_group_test_loader
from models.bandwidth_predictor import BandwidthPredictor
from training.evaluator import evaluate_model
from training.trainer import model_train_pipeline, train_model
from utils.helpers import ensure_directory


# =============================== Common utilities =============================== #

def load_config(config_path: Path) -> dict:
    """Read YAML config file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random/np/torch seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_child_seeds(master_seed: int, num_runs: int) -> List[int]:
    """Derive multiple reproducible child seeds from a master seed."""
    rng = np.random.default_rng(master_seed)
    # 32-bit unsigned range covers torch random seed requirements
    return rng.integers(low=1, high=2**31 - 1, size=num_runs, dtype=np.int64).tolist()


def mapping_str_to_tuple(mapping_str: str, target_total_gpu: int) -> Tuple[int, ...]:
    """Parse a GPU_Mapping string into a fixed-length 0/1 vector key.

    Pads to target_total_gpu so different node counts remain comparable across files.
    """
    nodes = ast.literal_eval(mapping_str)
    flattened: List[int] = []
    for node in nodes:
        flattened.extend(int(x) for x in node)

    if len(flattened) > target_total_gpu:
        raise ValueError(
            f"Parsed GPU_Mapping length {len(flattened)} exceeds target_total_gpu={target_total_gpu}"
        )
    if len(flattened) < target_total_gpu:
        flattened.extend([0] * (target_total_gpu - len(flattened)))
    return tuple(flattened)


def load_gpu_dataframe(csv_path: Path, target_total_gpu: int) -> pd.DataFrame:
    """Read GPU CSV and add canonical keys for later deduplication/filtering."""
    df = pd.read_csv(csv_path)
    df["mapping_key"] = df["GPU_Mapping_Across_Nodes"].apply(
        lambda s: mapping_str_to_tuple(s, target_total_gpu)
    )
    return df


def build_filtered_32_dataset(
    data_32_path: Path,
    exclusion_files: Sequence[Path],
    target_total_gpu: int,
) -> pd.DataFrame:
    """Build a 32-GPU dataset after removing configs that appear in the 16/24 datasets."""
    exclusion_keys: set[Tuple[int, ...]] = set()
    for csv_path in exclusion_files:
        if not csv_path.exists():
            continue
        df = load_gpu_dataframe(csv_path, target_total_gpu)
        exclusion_keys.update(df["mapping_key"].tolist())

    df_32 = load_gpu_dataframe(data_32_path, target_total_gpu)
    before = len(df_32)
    df_32 = df_32[~df_32["mapping_key"].isin(exclusion_keys)].copy()
    df_32 = df_32.drop_duplicates(subset="mapping_key").reset_index(drop=True)
    after = len(df_32)
    if after == 0:
        raise RuntimeError(
            "Filtered 32-GPU dataset is empty. Check whether the 16/24 datasets already cover all samples."
        )
    print(
        f"[INFO] Loaded 32-GPU data: {before} rows original, {after} rows after filtering, "
        f"excluded {before - after} rows."
    )
    return df_32


def arrays_from_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract GPU configuration matrix and ground-truth bandwidth targets from a DataFrame."""
    configs = np.array(df["mapping_key"].tolist(), dtype=np.int64)
    targets = df["Bandwidth(GB/s)"].to_numpy(dtype=np.float32)
    return configs, targets


def split_indices(
    num_samples: int,
    train_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices by ratio while keeping both train and eval non-empty."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    perm = rng.permutation(num_samples)
    train_size = max(1, int(num_samples * train_ratio))
    if train_size >= num_samples:
        train_size = num_samples - 1
    train_idx = perm[:train_size]
    eval_idx = perm[train_size:]
    return train_idx, eval_idx


def compute_extra_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MAPE, RMSE, and R² in addition to MSE/MAE."""
    errors = preds - targets
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    denom = np.clip(np.abs(targets), 1e-6, None)
    mape = float(np.mean(np.abs(errors) / denom) * 100.0)
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "r2": r2,
    }


def load_prediction_pickle(pickle_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load prediction/target pickle produced by evaluate_model."""
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    return np.asarray(data["preds"]), np.asarray(data["targets"])


def build_switch_and_dicts(
    cluster_type: str,
    total_gpu: int,
    bandwidth_dir: Path,
) -> Tuple[SwitchBandwidthConfig, List[dict]]:
    """Build switch config and bandwidth dictionaries for a given cluster type and GPU count."""
    switch_config = SwitchBandwidthConfig(
        num_machines=total_gpu // 8,
        cluster_type=cluster_type,
    )
    dict_files = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
    if not dict_files:
        raise RuntimeError(f"Cannot find bandwidth dictionary files for cluster_type={cluster_type}.")
    gpu_bw_dict_list = [load_gpu_bw_dict(bandwidth_dir / name) for name in dict_files]
    return switch_config, gpu_bw_dict_list


@dataclass
class StageResource:
    """Bundle static resources for each total_gpu."""

    total_gpu: int
    training_data_path: Path
    switch_config: SwitchBandwidthConfig
    gpu_bw_dict_list: List[dict]


# ============================= Main execution flow ============================= #

def run_base_training(
    total_gpu: int,
    seed: int,
    config: dict,
    resource: StageResource,
    device: torch.device,
    artifact_dir: Path,
    model_tag: str,
    num_train_override: int | None = None,
    num_test_override: int | None = None,
) -> Tuple[Path, Dict[str, float]]:
    """Call the existing pipeline to finish base training."""
    set_seed(seed)
    ensure_directory(artifact_dir)
    config_copy = copy.deepcopy(config)
    training_cfg = config_copy.setdefault("training", {})
    if num_train_override is not None:
        training_cfg["num_train_samples"] = num_train_override
    if num_test_override is not None:
        training_cfg["num_test_samples"] = num_test_override
    mse, mae, model_path = model_train_pipeline(
        total_gpu=total_gpu,
        gpu_bw_dict_list=resource.gpu_bw_dict_list,
        switch_config=resource.switch_config,
        training_data_path=str(resource.training_data_path),
        artifact_dir=artifact_dir,
        device=device,
        config=config_copy,
    )
    final_model_path = artifact_dir / f"{model_tag}_seed{seed}.pth"
    shutil.copyfile(model_path, final_model_path)
    return final_model_path, {"mse": float(mse), "mae": float(mae)}


def run_finetune_on_32(
    base_model_path: Path,
    seed: int,
    config: dict,
    device: torch.device,
    resource_32: StageResource,
    train_data: Tuple[np.ndarray, np.ndarray],
    eval_data: Tuple[np.ndarray, np.ndarray],
    finetune_dir: Path,
    finetune_epochs: int,
    finetune_lr: float,
    model_tag: str,
) -> Tuple[Dict[str, float], Path, Path]:
    """Finetune the base model on a 32-GPU subset and evaluate on the remaining data."""
    set_seed(seed)
    ensure_directory(finetune_dir)
    num_train_samples = config["training"]["num_train_samples"]

    gpu_train, bw_train = train_data
    train_loader, val_loader = get_group_data_loader(
        gpu_train=gpu_train,
        bw_train=bw_train,
        total_gpu=resource_32.total_gpu,
        gpu_bw_dict_list=resource_32.gpu_bw_dict_list,
        switch_config=resource_32.switch_config,
        training_data_path=str(resource_32.training_data_path),
        artifact_dir=finetune_dir,
        num_train_samples=num_train_samples,
        batch_size=config["training"]["batch_size"],
    )

    model = BandwidthPredictor(
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    )
    state_dict = torch.load(base_model_path, map_location=device)
    model.load_state_dict(state_dict)

    model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=finetune_epochs,
        lr=finetune_lr,
        weight_decay=config["training"]["weight_decay"],
        patience=max(10, finetune_epochs // 5),
        lambda_ewc=config["training"]["lambda_ewc"],
    )

    finetuned_path = finetune_dir / f"{model_tag}_finetune32_seed{seed}.pth"
    torch.save(model.state_dict(), finetuned_path)

    eval_configs, eval_targets = eval_data
    test_loader = get_group_test_loader(
        num_samples=len(eval_configs),
        total_gpu=resource_32.total_gpu,
        gpu_configs=eval_configs,
        bandwidth_targets=eval_targets,
        gpu_bw_dict_list=resource_32.gpu_bw_dict_list,
        switch_config=resource_32.switch_config,
        training_data_path=str(resource_32.training_data_path),
        artifact_dir=finetune_dir,
        num_train_samples=num_train_samples,
    )
    mse, mae = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        total_gpu=resource_32.total_gpu,
        gpu_bw_dict_list=resource_32.gpu_bw_dict_list,
        switch_config=resource_32.switch_config,
        training_data_path=str(resource_32.training_data_path),
        artifact_dir=finetune_dir,
        num_train_samples=num_train_samples,
    )

    pickle_path = finetune_dir / f"test_loader_data_{len(eval_configs)}Data.pkl"
    preds, targets = load_prediction_pickle(pickle_path)
    metrics = compute_extra_metrics(preds, targets)
    metrics.update({"reported_mae": float(mae), "reported_mse": float(mse)})
    return metrics, finetuned_path, pickle_path


def save_prediction_table(
    df_subset: pd.DataFrame,
    preds: np.ndarray,
    targets: np.ndarray,
    seed: int,
    model_tag: str,
    output_csv: Path,
) -> None:
    """Save a comparison table between predicted and ground-truth bandwidth values."""
    eval_df = df_subset.copy().reset_index(drop=True)
    eval_df["prediction(GB/s)"] = preds
    eval_df["target(GB/s)"] = targets
    eval_df["abs_error"] = np.abs(preds - targets)
    eval_df["abs_pct_error(%)"] = np.where(
        np.abs(targets) < 1e-6,
        np.nan,
        eval_df["abs_error"] / np.abs(targets) * 100.0,
    )
    eval_df.insert(0, "seed", seed)
    eval_df.insert(1, "model_tag", model_tag)
    eval_df.to_csv(output_csv, index=False)


def append_metrics_summary(summary_path: Path, rows: List[Dict[str, float]]) -> None:
    """Append experiment metric rows to a CSV file for multi-run comparison."""
    summary_df = pd.DataFrame(rows)
    if summary_path.exists():
        summary_df.to_csv(summary_path, mode="a", index=False, header=False)
    else:
        summary_df.to_csv(summary_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Variable-Length Transformer experiment script")
    parser.add_argument("--config", type=Path, default=Path("config/default_config.yaml"))
    parser.add_argument("--master-seed", type=int, default=2025)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/Evaluation/VariableLengthStudy"),
        help="Root directory to save experiment artifacts (models/CSVs)",
    )
    parser.add_argument(
        "--base-num-train-samples",
        type=int,
        default=None,
        help="Override base training sample count (default uses config.training.num_train_samples)",
    )
    parser.add_argument(
        "--base-num-test-samples",
        type=int,
        default=None,
        help="Override random eval sample count for base training (default uses config.training.num_test_samples)",
    )
    parser.add_argument(
        "--finetune-train-ratio",
        type=float,
        default=0.6,
        help="If finetune sample counts are not set, split 32-GPU data by this ratio",
    )
    parser.add_argument(
        "--finetune-train-count",
        type=int,
        default=None,
        help="Explicit 32-GPU finetune train sample count (overrides --finetune-train-ratio)",
    )
    parser.add_argument(
        "--finetune-eval-count",
        type=int,
        default=None,
        help="Explicit 32-GPU eval sample count (must be provided with --finetune-train-count)",
    )
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument(
        "--data-16-path",
        type=Path,
        default=Path("Data/H100_16/Pune_H100_16M_binary.csv"),
    )
    parser.add_argument(
        "--data-24-path",
        type=Path,
        default=Path("Data/H100_24/Pune_H100_16M_binary.csv"),
    )
    parser.add_argument(
        "--data-32-path",
        type=Path,
        default=Path("Data/H100_Real/Pune_H100_16M_binary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config.get("device", "cuda"))
    cluster_type = config["cluster"]["cluster_types"][0]
    bandwidth_dir = Path(config["data"]["bandwidth_dict_dir"])

    output_dir = args.output_dir.resolve()
    ensure_directory(output_dir)

    seeds = derive_child_seeds(args.master_seed, args.num_runs)
    resource_cache: Dict[int, StageResource] = {}

    def get_resource(total_gpu: int, data_path: Path) -> StageResource:
        if total_gpu not in resource_cache:
            switch_config, gpu_bw_dict_list = build_switch_and_dicts(
                cluster_type=cluster_type,
                total_gpu=total_gpu,
                bandwidth_dir=bandwidth_dir,
            )
            resource_cache[total_gpu] = StageResource(
                total_gpu=total_gpu,
                training_data_path=data_path,
                switch_config=switch_config,
                gpu_bw_dict_list=gpu_bw_dict_list,
            )
        return resource_cache[total_gpu]

    # Pre-load 32-GPU real-data samples after excluding all configurations
    # that already appear in the 16-GPU and 24-GPU datasets.
    df_32 = build_filtered_32_dataset(
        data_32_path=args.data_32_path,
        exclusion_files=[args.data_16_path, args.data_24_path],
        target_total_gpu=32,
    )
    configs_32, targets_32 = arrays_from_dataframe(df_32)

    metrics_rows: List[Dict[str, float]] = []
    summary_path = output_dir / "variable_length_metrics.csv"

    for run_id, seed in enumerate(seeds, start=1):
        print(f"\n========== Start experiment run#{run_id} seed={seed} ==========")
        rng = np.random.default_rng(seed)
        total_available = len(df_32)
        if args.finetune_train_count is None and args.finetune_eval_count is None:
            train_idx, eval_idx = split_indices(
                num_samples=total_available,
                train_ratio=args.finetune_train_ratio,
                rng=rng,
            )
        else:
            if args.finetune_train_count is None or args.finetune_eval_count is None:
                raise ValueError(
                    "Finetune sample counts require both --finetune-train-count and --finetune-eval-count."
                )
            total_needed = args.finetune_train_count + args.finetune_eval_count
            if total_needed > total_available:
                raise ValueError(
                    f"Finetune sample demand ({total_needed}) exceeds available 32-GPU samples ({total_available})."
                )
            perm = rng.permutation(total_available)
            train_idx = perm[: args.finetune_train_count]
            eval_idx = perm[
                args.finetune_train_count : args.finetune_train_count + args.finetune_eval_count
            ]
        train_data = (configs_32[train_idx], targets_32[train_idx])
        eval_data = (configs_32[eval_idx], targets_32[eval_idx])

        base_model_paths: Dict[int, Path] = {}
        for total_gpu, data_path in [(16, args.data_16_path), (24, args.data_24_path)]:
            model_tag = f"H100_{total_gpu}GPU"
            artifact_dir = (
                output_dir / model_tag / f"seed_{seed}" / "base_model"
            ).resolve()
            resource = get_resource(total_gpu, data_path)
            base_model_path, base_metrics = run_base_training(
                total_gpu=total_gpu,
                seed=seed,
                config=config,
                resource=resource,
                device=device,
                artifact_dir=artifact_dir,
                model_tag=model_tag,
                num_train_override=args.base_num_train_samples,
                num_test_override=args.base_num_test_samples,
            )
            base_model_paths[total_gpu] = base_model_path
            base_metrics.update(
                {
                    "seed": seed,
                    "stage": "base_train_random_eval",
                    "model_tag": model_tag,
                    "num_train_samples": (
                        args.base_num_train_samples
                        if args.base_num_train_samples is not None
                        else config["training"]["num_train_samples"]
                    ),
                    "num_test_samples": (
                        args.base_num_test_samples
                        if args.base_num_test_samples is not None
                        else config["training"]["num_test_samples"]
                    ),
                }
            )
            metrics_rows.append(base_metrics)

        resource_32 = get_resource(32, args.data_32_path)
        for total_gpu in (16, 24):
            model_tag = f"H100_{total_gpu}GPU"
            finetune_dir = (
                output_dir / model_tag / f"seed_{seed}" / "finetune32"
            ).resolve()
            finetune_metrics, finetuned_model_path, pickle_path = run_finetune_on_32(
                base_model_path=base_model_paths[total_gpu],
                seed=seed,
                config=config,
                device=device,
                resource_32=resource_32,
                train_data=train_data,
                eval_data=eval_data,
                finetune_dir=finetune_dir,
                finetune_epochs=args.finetune_epochs,
                finetune_lr=args.finetune_lr,
                model_tag=model_tag,
            )
            preds, targets = load_prediction_pickle(pickle_path)
            pred_csv = (
                finetune_dir / f"pred_vs_actual_{model_tag}_seed{seed}.csv"
            ).resolve()
            save_prediction_table(
                df_subset=df_32.iloc[eval_idx],
                preds=preds,
                targets=targets,
                seed=seed,
                model_tag=model_tag,
                output_csv=pred_csv,
            )
            finetune_metrics.update(
                {
                    "seed": seed,
                    "stage": "finetune_eval32",
                    "model_tag": model_tag,
                    "num_finetune_samples": len(train_idx),
                    "num_eval_samples": len(eval_idx),
                    "finetuned_model_path": str(finetuned_model_path),
                }
            )
            metrics_rows.append(finetune_metrics)

        append_metrics_summary(summary_path, metrics_rows)
        metrics_rows.clear()

    print(f"\nExperiment finished. Metrics summary written to: {summary_path}")


if __name__ == "__main__":
    main()

