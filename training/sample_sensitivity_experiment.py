"""Predictor-level sample-size sensitivity analysis.

The script reports `R^2`, `MAPE`, and `RMSE` for different training sample
budgets. The `independent` protocol samples each budget separately. The
`nested` protocol builds one mother pool per `(cluster, strategy, seed)` and
derives cumulative subsets such as `100 subset 250 subset 500`.

Examples:
    python -m training.sample_sensitivity_experiment --config config/default_config.yaml
    python -m training.sample_sensitivity_experiment --config config/default_config.yaml --cluster H100_26H100_27H100_28H100_29
    python -m training.sample_sensitivity_experiment --config config/default_config.yaml --sampling-protocol nested --sample-sizes 100,250,500
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import shutil
import sys
import time
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files, load_gpu_bw_dict
from data_process.dataloader import get_group_data_loader, get_group_test_loader
from data_process.dataset import (
    _compute_bandwidths,
    get_balanced_train_dataset,
    get_random_train_dataset,
    get_stratified_train_dataset,
    get_worst_case_train_dataset,
)
from models.bandwidth_predictor import BandwidthPredictor
from training.evaluator import compute_extra_metrics
from training.sensitivity_sampling_protocol import (
    build_nested_dataset_family,
    extend_nested_dataset_family_from_manifest,
)
from training.trainer import train_model
from utils.helpers import build_artifact_filename, ensure_directory

logger = logging.getLogger(__name__)

# ====================== Configuration & Constants ====================== #

# Default sample sizes to evaluate (covers the saturation curve)
DEFAULT_SAMPLE_SIZES = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500]

# Sampling strategy name => dataset generator function
STRATEGY_GENERATORS = {
    "Random": get_balanced_train_dataset,       # balanced allocation = default "Random" in paper
    "Stratified": get_stratified_train_dataset,
    "Worst-Case": get_worst_case_train_dataset,
}

DEFAULT_NUM_SEEDS = 5
DEFAULT_TEST_SAMPLES = 2500


# ========================== Utility Functions ========================== #

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_child_seeds(master_seed: int, n: int) -> List[int]:
    rng = np.random.default_rng(master_seed)
    return rng.integers(low=1, high=2**31 - 1, size=n, dtype=np.int64).tolist()


def compute_sparsity_ratio(total_gpu: int, num_train_samples: int) -> float:
    """Compute the sampling ratio: num_train_samples / total number of possible GPU subsets."""
    total_combos = sum(comb(total_gpu, k) for k in range(1, total_gpu + 1))
    return num_train_samples / total_combos


# ========================= Core Experiment Loop ========================= #

def _train_and_evaluate_single(
    *,
    sample_size: int,
    strategy_name: str,
    seed: int,
    total_gpu: int,
    gpu_bw_dict_list: list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    test_configs: np.ndarray,
    test_bandwidths: np.ndarray,
    model_cfg: dict,
    training_cfg: dict,
    work_dir: Path,
    device: torch.device,
    gpu_train: np.ndarray | None = None,
    bw_train: np.ndarray | None = None,
) -> Dict[str, float]:
    """Train one model and evaluate it - the inner loop of the experiment.

    Returns a dict with keys: r2, mape_percent, rmse, mse, mae.
    """
    set_seed(seed)

    # 1. Generate training data using the specified strategy unless the caller
    # already prepared a nested/cumulative subset for this sample budget.
    if gpu_train is None or bw_train is None:
        generator_fn = STRATEGY_GENERATORS[strategy_name]
        gpu_train, bw_train = generator_fn(
            num_samples=sample_size,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
        )

    # 2. Create data loaders (persists scalers to work_dir required by test loader)
    artifact_dir = work_dir / f"n{sample_size}_{strategy_name}_s{seed}"
    ensure_directory(artifact_dir)

    train_loader, val_loader = get_group_data_loader(
        gpu_train, bw_train,
        total_gpu, gpu_bw_dict_list, switch_config, training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=sample_size,
        batch_size=training_cfg.get("batch_size", 100),
    )

    # 3. Build & train model
    model = BandwidthPredictor(
        hidden_dim=model_cfg.get("hidden_dim", 32),
        num_layers=model_cfg.get("num_layers", 6),
        num_heads=model_cfg.get("num_heads", 8),
        dropout=model_cfg.get("dropout", 0.05),
    )
    model, _ = train_model(
        model, train_loader, val_loader, device,
        num_epochs=training_cfg.get("num_epochs", 300),
        lr=training_cfg.get("learning_rate", 0.001),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
        patience=training_cfg.get("patience", 80),
        lambda_ewc=training_cfg.get("lambda_ewc", 2.0),
    )

    # 4. Evaluate on the shared fixed test set
    test_loader = get_group_test_loader(
        num_samples=len(test_configs),
        total_gpu=total_gpu,
        gpu_configs=test_configs,
        bandwidth_targets=test_bandwidths,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=sample_size,
    )

    model.eval()
    import pickle
    y_scaler = pickle.loads(
        (artifact_dir / build_artifact_filename("y_scaler", sample_size, ".pkl")).read_bytes()
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_bws, x_node_counts, x_total_counts, y_batch in test_loader:
            x_bws = x_bws.to(device)
            x_node_counts = x_node_counts.to(device)
            x_total_counts = x_total_counts.to(device)
            outputs = model(x_bws, x_node_counts, x_total_counts)["final_bandwidth"].view(-1)
            pred_np = outputs.cpu().numpy().reshape(-1, 1)
            target_np = y_batch.numpy().reshape(-1, 1)
            all_preds.append(y_scaler.inverse_transform(pred_np).flatten())
            all_targets.append(y_scaler.inverse_transform(target_np).flatten())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_extra_metrics(preds, targets)

    # 5. Clean up per-run artifacts to save disk space (only keep metrics)
    shutil.rmtree(artifact_dir, ignore_errors=True)

    return metrics


def run_sensitivity_experiment(
    *,
    cluster_type: str,
    total_gpu: int,
    gpu_bw_dict_list: list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    model_cfg: dict,
    training_cfg: dict,
    sample_sizes: List[int],
    strategies: List[str],
    num_seeds: int,
    master_seed: int,
    num_test_samples: int,
    device: torch.device,
    output_dir: Path,
    sampling_protocol: str = "independent",
    nested_max_sample_size: int | None = None,
    nested_extend_from_artifact_dir: Path | None = None,
    existing_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full sensitivity sweep for a single cluster type.

    Returns a DataFrame with columns:
        cluster_type, sample_size, strategy, seed, sampling_protocol,
        r2, mape_percent, rmse, mse, mae
    """
    work_dir = output_dir / "tmp_artifacts" / cluster_type
    ensure_directory(work_dir)

    child_seeds = derive_child_seeds(master_seed, num_seeds)
    print(f"\n{'='*70}")
    print(f"  Sensitivity Analysis - {cluster_type}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Strategies: {strategies}")
    print(f"  Seeds ({num_seeds}): {child_seeds}")
    print(f"  Test samples: {num_test_samples}")
    print(f"  Sampling protocol: {sampling_protocol}")
    sparsity = compute_sparsity_ratio(total_gpu, 250)
    print(f"  Sparsity: 250 / (2^{total_gpu}-1) ≈ {sparsity:.2e}")
    print(f"{'='*70}\n")

    # --- Generate shared fixed test set ---
    print("[INFO] Generating shared test set ...")
    set_seed(master_seed + 9999)  # deterministic but distinct from training seeds
    test_configs, test_bandwidths = get_random_train_dataset(
        num_samples=num_test_samples,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
    )
    print(f"[INFO] Shared test set ready: {len(test_configs)} samples\n")

    records: List[dict] = []
    reusable_rows = (
        existing_results_df.copy()
        if existing_results_df is not None and not existing_results_df.empty
        else pd.DataFrame()
    )
    total_runs = len(sample_sizes) * len(strategies) * num_seeds
    run_idx = 0

    nested_manifest_root = output_dir / "nested_manifests"
    for strat in strategies:
        for seed in child_seeds:
            nested_family = None
            if sampling_protocol == "nested":
                if nested_extend_from_artifact_dir is not None:
                    nested_family = extend_nested_dataset_family_from_manifest(
                        cluster_type=cluster_type,
                        strategy_name=strat,
                        sample_sizes=sample_sizes,
                        seed=seed,
                        total_gpu=total_gpu,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        training_data_path=training_data_path,
                        generator_fn=STRATEGY_GENERATORS[strat],
                        compute_bandwidths_fn=_compute_bandwidths,
                        existing_manifest_root=nested_extend_from_artifact_dir / "nested_manifests",
                        manifest_root=nested_manifest_root,
                        mother_pool_size=nested_max_sample_size,
                        protocol_name=sampling_protocol,
                    )
                else:
                    nested_family = build_nested_dataset_family(
                        cluster_type=cluster_type,
                        strategy_name=strat,
                        sample_sizes=sample_sizes,
                        seed=seed,
                        total_gpu=total_gpu,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        training_data_path=training_data_path,
                        generator_fn=STRATEGY_GENERATORS[strat],
                        compute_bandwidths_fn=_compute_bandwidths,
                        manifest_root=nested_manifest_root,
                        mother_pool_size=nested_max_sample_size,
                        protocol_name=sampling_protocol,
                    )

            for n in sample_sizes:
                run_idx += 1
                tag = f"[{run_idx}/{total_runs}] n={n}, strategy={strat}, seed={seed}"
                print(f"--- {tag} ---")
                if not reusable_rows.empty:
                    existing_match = reusable_rows[
                        (reusable_rows["cluster_type"] == cluster_type)
                        & (reusable_rows["sample_size"] == int(n))
                        & (reusable_rows["strategy"] == strat)
                        & (reusable_rows["seed"] == int(seed))
                        & (reusable_rows["sampling_protocol"] == sampling_protocol)
                    ]
                    if len(existing_match) == 1:
                        reused_row = existing_match.iloc[0].to_dict()
                        print(
                            "    Reuse existing metrics: "
                            f"R^2={float(reused_row['r2']):.4f}  "
                            f"MAPE={float(reused_row['mape_percent']):.2f}%  "
                            f"RMSE={float(reused_row['rmse']):.2f}"
                        )
                        records.append(reused_row)
                        continue

                t0 = time.time()

                metrics = _train_and_evaluate_single(
                    sample_size=n,
                    strategy_name=strat,
                    seed=seed,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    training_data_path=training_data_path,
                    test_configs=test_configs,
                    test_bandwidths=test_bandwidths,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    work_dir=work_dir,
                    device=device,
                    gpu_train=None if nested_family is None else nested_family[int(n)][0],
                    bw_train=None if nested_family is None else nested_family[int(n)][1],
                )

                elapsed = time.time() - t0
                print(f"    R^2={metrics['r2']:.4f}  MAPE={metrics['mape_percent']:.2f}%  "
                      f"RMSE={metrics['rmse']:.2f}  ({elapsed:.1f}s)")

                records.append({
                    "cluster_type": cluster_type,
                    "sample_size": n,
                    "strategy": strat,
                    "seed": seed,
                    "sampling_protocol": sampling_protocol,
                    **metrics,
                })

    # Clean up temp working directory
    shutil.rmtree(work_dir, ignore_errors=True)

    df = pd.DataFrame(records)
    return df


# ========================= Plotting Functions ========================= #

def plot_sensitivity_figures(
    df: pd.DataFrame,
    output_dir: Path,
    total_gpu: int,
) -> None:
    """Generate TPDS-quality sensitivity figures.

    Produces:
      - sensitivity_R2.pdf:   R^2 vs sample size (one subplot per cluster)
      - sensitivity_MAPE.pdf: MAPE vs sample size (one subplot per cluster)
      - sensitivity_RMSE.pdf: RMSE vs sample size (one subplot per cluster)
      - sensitivity_combined.pdf: 2x2 panel (R^2 + MAPE for each cluster)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ensure_directory(output_dir)

    # ---- Style configuration for IEEE TPDS ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })

    clusters = df["cluster_type"].unique().tolist()
    strategies = df["strategy"].unique().tolist()

    # Formatting helpers
    CLUSTER_LABELS = {
        "H100_26H100_27H100_28H100_29": "H100 Cluster",
        "Het-4Mix": "Het-4Mix Cluster",
    }
    STRATEGY_MARKERS = {
        "Random": ("o", "#2078B4"),      # blue
        "Stratified": ("s", "#FF7F0E"),  # orange
        "Worst-Case": ("^", "#D62728"),  # red
    }

    def _cluster_label(ct: str) -> str:
        return CLUSTER_LABELS.get(ct, ct)

    def _aggregate(sub_df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (sample_sizes, means, stds) sorted by sample_size."""
        agg = sub_df.groupby("sample_size")[metric].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("sample_size")
        return agg["sample_size"].values, agg["mean"].values, agg["std"].values

    # =========== Individual metric plots =========== #
    for metric, ylabel, threshold, thresh_label in [
        ("r2", "$R^2$", 0.95, "$R^2 = 0.95$"),
        ("mape_percent", "MAPE (%)", 5.0, "MAPE = 5%"),
        ("rmse", "RMSE (GB/s)", None, None),
    ]:
        ncols = len(clusters)
        fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 2.8), squeeze=False)

        for col_idx, ct in enumerate(clusters):
            ax = axes[0, col_idx]
            ct_df = df[df["cluster_type"] == ct]

            for strat in strategies:
                strat_df = ct_df[ct_df["strategy"] == strat]
                xs, means, stds = _aggregate(strat_df, metric)
                marker, color = STRATEGY_MARKERS.get(strat, ("D", "#888888"))
                ax.plot(xs, means, marker=marker, color=color, label=strat, zorder=3)
                ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color, zorder=2)

            if threshold is not None:
                ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
                # Place text near the right edge
                ax.text(
                    xs[-1], threshold, f"  {thresh_label}",
                    va="bottom" if metric == "r2" else "top",
                    ha="right", fontsize=7.5, color="gray",
                )

            ax.set_xlabel("Number of Training Samples")
            ax.set_ylabel(ylabel)
            ax.set_title(_cluster_label(ct))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if col_idx == 0:
                ax.legend(loc="best", framealpha=0.8)

        fig.tight_layout()
        fig_path = output_dir / f"sensitivity_{metric}.pdf"
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))
        plt.close(fig)
        print(f"[Figure] Saved {fig_path}")

    # =========== Combined 2-row x N-column panel (R^2 + MAPE) =========== #
    ncols = len(clusters)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 5.0), squeeze=False)

    metric_info = [
        ("r2", "$R^2$", 0.95, "$R^2 = 0.95$"),
        ("mape_percent", "MAPE (%)", 5.0, "MAPE = 5%"),
    ]

    for row_idx, (metric, ylabel, threshold, thresh_label) in enumerate(metric_info):
        for col_idx, ct in enumerate(clusters):
            ax = axes[row_idx, col_idx]
            ct_df = df[df["cluster_type"] == ct]

            for strat in strategies:
                strat_df = ct_df[ct_df["strategy"] == strat]
                xs, means, stds = _aggregate(strat_df, metric)
                marker, color = STRATEGY_MARKERS.get(strat, ("D", "#888888"))
                ax.plot(xs, means, marker=marker, color=color, label=strat, zorder=3)
                ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color, zorder=2)

            if threshold is not None:
                ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
                ax.text(
                    xs[-1], threshold, f"  {thresh_label}",
                    va="bottom" if metric == "r2" else "top",
                    ha="right", fontsize=7.5, color="gray",
                )

            ax.set_xlabel("Number of Training Samples")
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(_cluster_label(ct))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if col_idx == 0 and row_idx == 0:
                ax.legend(loc="best", framealpha=0.8)

    fig.tight_layout()
    combined_path = output_dir / "sensitivity_combined.pdf"
    fig.savefig(combined_path)
    fig.savefig(combined_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[Figure] Saved {combined_path}")

    # =========== Summary table (LaTeX) =========== #
    _generate_latex_table(df, output_dir, total_gpu)


def _generate_latex_table(df: pd.DataFrame, output_dir: Path, total_gpu: int) -> None:
    """Generate a concise LaTeX-formatted results table.

    Table format: sample_size | Strategy | R^2 | MAPE(%) | RMSE
    Grouped by cluster_type, with 250-sample row highlighted.
    """
    seed_counts = (
        df.groupby(["cluster_type", "strategy", "sample_size"])["seed"]
        .nunique()
        .tolist()
    )
    display_seed_count = seed_counts[0] if seed_counts and len(set(seed_counts)) == 1 else "multiple"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sample-size sensitivity analysis. "
        rf"Each entry shows $\text{{mean}} \pm \text{{std}}$ over {display_seed_count} random seeds.}}",
        r"\label{tab:sensitivity}",
        r"\small",
    ]

    for ct in df["cluster_type"].unique():
        ct_df = df[df["cluster_type"] == ct]
        strategies = ct_df["strategy"].unique().tolist()

        lines.append(r"\begin{tabular}{c l c c c}")
        lines.append(r"\toprule")
        lines.append(r"$n_{\text{train}}$ & Strategy & $R^2$ & MAPE (\%) & RMSE (GB/s) \\")
        lines.append(r"\midrule")

        for n in sorted(ct_df["sample_size"].unique()):
            for strat in strategies:
                sub = ct_df[(ct_df["sample_size"] == n) & (ct_df["strategy"] == strat)]
                r2_m, r2_s = sub["r2"].mean(), sub["r2"].std()
                mape_m, mape_s = sub["mape_percent"].mean(), sub["mape_percent"].std()
                rmse_m, rmse_s = sub["rmse"].mean(), sub["rmse"].std()
                bold = r"\bfseries " if n == 250 else ""
                lines.append(
                    f"  {bold}{n} & {bold}{strat} & "
                    f"{bold}{r2_m:.3f}$\\pm${r2_s:.3f} & "
                    f"{bold}{mape_m:.1f}$\\pm${mape_s:.1f} & "
                    f"{bold}{rmse_m:.1f}$\\pm${rmse_s:.1f} \\\\"
                )
            lines.append(r"\cmidrule(lr){1-5}")

        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")

        sparsity = compute_sparsity_ratio(total_gpu, 250)
        lines.append(f"% sparsity (250 samples): {sparsity:.2e}")
        lines.append("")

    lines.append(r"\end{table}")

    tex_path = output_dir / "sensitivity_table.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[LaTeX] Saved {tex_path}")


# ============================= CLI Entry Point ============================= #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training sample-size sensitivity analysis for BandPilot"
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_config.yaml"))
    parser.add_argument(
        "--cluster", type=str, default=None,
        help="Run for a single cluster_type only (default: all in config)",
    )
    parser.add_argument(
        "--sample-sizes", type=str, default=None,
        help="Comma-separated sample sizes, e.g. '50,100,250,500'",
    )
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--num-test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument(
        "--sampling-protocol",
        type=str,
        choices=["independent", "nested"],
        default="independent",
        help="Use historical independent resampling or cumulative nested budgets.",
    )
    parser.add_argument(
        "--nested-max-sample-size",
        type=int,
        default=None,
        help="Mother-pool size for nested protocol (default: max(sample_sizes)).",
    )
    parser.add_argument(
        "--nested-extend-from-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Reuse an existing nested predictor artifact and only extend to a larger "
            "sample size while preserving old budgets exactly."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/sensitivity-analysis/artifacts/predictor-level"),
        help="Directory for output figures and CSV",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    cluster_cfg = config["cluster"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    total_gpu = cluster_cfg["total_gpu"]

    device_str = args.device or config.get("device", "cuda")
    device = torch.device(device_str)

    sample_sizes = (
        [int(x) for x in args.sample_sizes.split(",")]
        if args.sample_sizes
        else DEFAULT_SAMPLE_SIZES
    )

    cluster_types = (
        [args.cluster] if args.cluster else cluster_cfg["cluster_types"]
    )

    strategies = list(STRATEGY_GENERATORS.keys())
    bandwidth_dir = Path(data_cfg["bandwidth_dict_dir"])
    training_data_path = data_cfg["h100_training_data_path"]

    output_dir = args.output_dir
    ensure_directory(output_dir)
    existing_results_df: pd.DataFrame | None = None
    if args.nested_extend_from_artifact_dir is not None:
        existing_results_csv = args.nested_extend_from_artifact_dir / "sensitivity_results.csv"
        if not existing_results_csv.exists():
            raise FileNotFoundError(f"Missing existing sensitivity results: {existing_results_csv}")
        existing_results_df = pd.read_csv(existing_results_csv)

    all_dfs: List[pd.DataFrame] = []

    for cluster_type in cluster_types:
        switch_config = SwitchBandwidthConfig(
            num_machines=total_gpu // 8,
            cluster_type=cluster_type,
        )
        file_list = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
        gpu_bw_dict_list = [load_gpu_bw_dict(bandwidth_dir / name) for name in file_list]

        df = run_sensitivity_experiment(
            cluster_type=cluster_type,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            sample_sizes=sample_sizes,
            strategies=strategies,
            num_seeds=args.num_seeds,
            master_seed=args.master_seed,
            num_test_samples=args.num_test_samples,
            device=device,
            output_dir=output_dir,
            sampling_protocol=args.sampling_protocol,
            nested_max_sample_size=args.nested_max_sample_size,
            nested_extend_from_artifact_dir=args.nested_extend_from_artifact_dir,
            existing_results_df=existing_results_df,
        )
        all_dfs.append(df)

    # Merge all cluster results and save
    full_df = pd.concat(all_dfs, ignore_index=True)
    csv_path = output_dir / "sensitivity_results.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\n[Results] Full CSV saved to {csv_path}")

    # Also save as JSON for machine readability
    json_path = output_dir / "sensitivity_results.json"
    full_df.to_json(json_path, orient="records", indent=2)
    print(f"[Results] Full JSON saved to {json_path}")

    # Generate plots
    plot_sensitivity_figures(full_df, output_dir, total_gpu)

    # Print summary
    print("\n" + "=" * 70)
    print("  SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    for ct in full_df["cluster_type"].unique():
        ct_df = full_df[full_df["cluster_type"] == ct]
        print(f"\n  Cluster: {ct}")
        for strat in strategies:
            strat_df = ct_df[ct_df["strategy"] == strat]
            ref = strat_df[strat_df["sample_size"] == 250]
            if len(ref) > 0:
                print(f"    [{strat}] @ n=250: "
                      f"R^2={ref['r2'].mean():.4f}±{ref['r2'].std():.4f}, "
                      f"MAPE={ref['mape_percent'].mean():.2f}±{ref['mape_percent'].std():.2f}%")
    print()
    sparsity = compute_sparsity_ratio(total_gpu, 250)
    print(f"  Sampling sparsity (n=250, {total_gpu} GPUs): {sparsity:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
