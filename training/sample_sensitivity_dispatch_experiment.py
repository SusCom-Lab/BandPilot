"""Dispatch-level sensitivity sidecar for BandPilot sample-size analysis.

The sidecar replays a fixed single-contention case stream with predictors
trained under different sample budgets and sampling protocols. `independent`
resamples each budget separately; `nested` derives cumulative budgets from one
mother pool so smaller budgets are strict subsets of larger budgets.

Example:
```bash
conda run -n gpu_dp_opt python -m training.sample_sensitivity_dispatch_experiment \
  --config config/default_config.yaml \
  --sample-sizes 100,250,500 \
  --strategies Random,Stratified,Worst-Case \
  --num-seeds 5 \
  --repeat-indices 0,1,2,3,4,5,6,7,8,9 \
  --sampling-protocol nested \
  --output-dir evaluation/sensitivity-analysis/artifacts/dispatch_sidecar/<run-tag>
```

- `rows.csv`: per-case dispatch rows.
- `seed_summary.csv`: per `(cluster, strategy, sample_size, seed)` summaries.
- `summary.csv`: cross-seed aggregate summary.
- `summary_by_mode.csv`: contention-mode aggregates across seeds.
- `nested_manifests/`: nested protocol mother-pool membership manifests.
- `dispatch_sensitivity_utilization.pdf/.png`
- `dispatch_sensitivity_latency.pdf/.png`
- `report.md`
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.search import legacy_improved_searching_algo
from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files, load_gpu_bw_dict
from data_process.dataloader import get_group_data_loader, get_group_test_loader
from data_process.dataset import _compute_bandwidths, get_random_train_dataset
from evaluation.compare import (
    build_max_bw_cache_filename,
    build_single_contention_real_manager,
    iter_single_contention_case_contexts,
    prepare_single_contention_runtime_context,
    run_single_contention_search_algorithm,
)
from models.bandwidth_predictor import BandwidthPredictor
from training.evaluator import compute_extra_metrics
from training.sensitivity_sampling_protocol import build_nested_dataset_family
from training.sample_sensitivity_experiment import STRATEGY_GENERATORS
from training.trainer import train_model
from utils.helpers import build_artifact_filename, ensure_directory, record_active_num_train_samples

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZES = [100, 250, 500]
DEFAULT_STRATEGIES = ["Random", "Stratified", "Worst-Case"]
DEFAULT_NUM_SEEDS = 3
DEFAULT_REPEAT_INDICES = [0, 1, 2, 3, 4]
DEFAULT_K_VALUES = [4, 8, 12, 16, 20, 24, 28]
DEFAULT_CONTENTION_MODES = ["idle", "common", "intensive"]
DEFAULT_NUM_TEST_SAMPLES = 2500


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_child_seeds(master_seed: int, n: int) -> List[int]:
    rng = np.random.default_rng(master_seed)
    return rng.integers(low=1, high=2**31 - 1, size=n, dtype=np.int64).tolist()


def _parse_csv_list(raw: str | None, value_type=int) -> List:
    if raw is None:
        return []
    return [value_type(item.strip()) for item in raw.split(",") if item.strip()]


def _load_cluster_runtime(
    *,
    config: dict,
    cluster_type: str,
) -> Tuple[int, SwitchBandwidthConfig, List[dict], str, str, Path]:
    """Resolve cluster runtime resources for the dispatch sidecar.

    Returns GPU count, switch config, bandwidth dictionaries, data paths, and
    the evaluation artifact directory used by compare-compatible helpers.
    """

    data_cfg = config["data"]
    cluster_cfg = config["cluster"]
    total_gpu = int(cluster_cfg["total_gpu"])

    bandwidth_dir = Path(data_cfg["bandwidth_dict_dir"])
    file_list = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
    gpu_bw_dict_list = [load_gpu_bw_dict(bandwidth_dir / name) for name in file_list]

    switch_config = SwitchBandwidthConfig(
        num_machines=total_gpu // 8,
        cluster_type=cluster_type,
    )
    training_data_path = data_cfg["h100_training_data_path"]
    evaluation_data_path = data_cfg.get("h100_evaluation_data_path", training_data_path)
    evaluation_base_dir = Path(data_cfg["evaluation_dir"]) / cluster_type
    return total_gpu, switch_config, gpu_bw_dict_list, training_data_path, evaluation_data_path, evaluation_base_dir


def _build_model(
    *,
    model_cfg: dict,
) -> BandwidthPredictor:
    """Build the hierarchical bandwidth predictor from config values."""
    return BandwidthPredictor(
        hidden_dim=model_cfg.get("hidden_dim", 32),
        num_layers=model_cfg.get("num_layers", 6),
        num_heads=model_cfg.get("num_heads", 8),
        dropout=model_cfg.get("dropout", 0.05),
    )


def _evaluate_predictor_on_shared_test_set(
    *,
    model: BandwidthPredictor,
    device: torch.device,
    artifact_dir: Path,
    sample_size: int,
    total_gpu: int,
    gpu_bw_dict_list: list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    test_configs: np.ndarray,
    test_bandwidths: np.ndarray,
) -> Dict[str, float]:
    """Evaluate predictor-level metrics on the shared test set.

    These metrics accompany dispatch-level utilization so the sidecar can relate
    predictor quality to scheduling behavior.
    """

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

    scaler_path = artifact_dir / build_artifact_filename("y_scaler", sample_size, ".pkl")
    y_scaler = pickle.loads(scaler_path.read_bytes())

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_bws, x_node_counts, x_total_counts, y_batch in test_loader:
            x_bws = x_bws.to(device)
            x_node_counts = x_node_counts.to(device)
            x_total_counts = x_total_counts.to(device)
            outputs = model(x_bws, x_node_counts, x_total_counts)["final_bandwidth"].view(-1)
            pred_np = outputs.detach().cpu().numpy().reshape(-1, 1)
            target_np = y_batch.numpy().reshape(-1, 1)
            all_preds.append(y_scaler.inverse_transform(pred_np).flatten())
            all_targets.append(y_scaler.inverse_transform(target_np).flatten())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_extra_metrics(preds, targets)


def train_predictor_variant(
    *,
    cluster_type: str,
    strategy_name: str,
    sample_size: int,
    seed: int,
    total_gpu: int,
    gpu_bw_dict_list: list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    model_cfg: dict,
    training_cfg: dict,
    test_configs: np.ndarray,
    test_bandwidths: np.ndarray,
    device: torch.device,
    model_root: Path,
    gpu_train: np.ndarray | None = None,
    bw_train: np.ndarray | None = None,
) -> Tuple[Path, Path, Dict[str, float]]:
    """Train one predictor variant and return paths plus predictor metrics."""

    set_seed(seed)
    strategy_dir = strategy_name.lower().replace(" ", "-")
    artifact_dir = model_root / cluster_type / strategy_dir / f"n{sample_size}_seed{seed}"
    ensure_directory(artifact_dir)

    if gpu_train is None or bw_train is None:
        generator_fn = STRATEGY_GENERATORS[strategy_name]
        gpu_train, bw_train = generator_fn(
            num_samples=sample_size,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
        )

    # The dataloader writes scalers into artifact_dir for later inverse scaling.
    train_loader, val_loader = get_group_data_loader(
        gpu_train,
        bw_train,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=sample_size,
        batch_size=int(training_cfg.get("batch_size", 100)),
    )

    model = _build_model(model_cfg=model_cfg)
    model, _ = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=int(training_cfg.get("num_epochs", 300)),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        patience=int(training_cfg.get("patience", 80)),
        lambda_ewc=float(training_cfg.get("lambda_ewc", 2.0)),
    )
    model_path = artifact_dir / build_artifact_filename("bandwidth_predictor", sample_size, ".pth")
    torch.save(model.state_dict(), model_path)
    record_active_num_train_samples(artifact_dir, sample_size)

    predictor_metrics = _evaluate_predictor_on_shared_test_set(
        model=model,
        device=device,
        artifact_dir=artifact_dir,
        sample_size=sample_size,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
        test_configs=test_configs,
        test_bandwidths=test_bandwidths,
    )
    return model_path, artifact_dir, predictor_metrics


def _resolve_real_max_bw_cache(
    *,
    evaluation_cluster_dir: Path,
    cache_seed: int,
    cache_num_train_samples: int,
    total_gpu: int,
    repeat_num: int,
    contention_mode: str,
    if_dynamic: bool,
    local_top_k: int,
    max_combos_per_distribution: int,
    max_total_combos: int,
) -> Path:
    """Resolve the real-data `realSM` max-bandwidth cache.

    The cache is shared across predictor variants so dispatch comparisons use
    the same oracle denominator and remain fair.
    """

    file_name = build_max_bw_cache_filename(
        random_seed=cache_seed,
        num_train_samples=cache_num_train_samples,
        total_gpu=total_gpu,
        repeat_num=repeat_num,
        if_dynamic=if_dynamic,
        contention_mode=contention_mode,
        search_if_real_data=True,
        local_top_k=local_top_k,
        max_combos_per_distribution=max_combos_per_distribution,
        max_total_combos=max_total_combos,
    )
    cache_path = evaluation_cluster_dir / file_name
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing realSM max_bw cache for mode={contention_mode}: {cache_path}"
        )
    return cache_path


def evaluate_dispatch_variant(
    *,
    cluster_type: str,
    strategy_name: str,
    sample_size: int,
    seed: int,
    model_path: Path,
    artifact_dir: Path,
    predictor_metrics: Dict[str, float],
    total_gpu: int,
    gpu_bw_dict_list: list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    evaluation_data_path: str,
    evaluation_cluster_dir: Path,
    model_cfg: dict,
    contention_modes: Sequence[str],
    k_values: Sequence[int],
    repeat_indices: Sequence[int],
    case_seed: int,
    real_cache_anchor_num_train_samples: int,
    max_bw_cfg: dict,
    adaptive_threshold_policy: dict | None,
    sampling_protocol: str,
) -> pd.DataFrame:
    """Evaluate one predictor variant on a BandPilot dispatch case stream."""

    records: List[Dict[str, object]] = []
    repeat_num = int(max(repeat_indices) + 1) if repeat_indices else 1
    for contention_mode in contention_modes:
        max_bw_cache_file = _resolve_real_max_bw_cache(
            evaluation_cluster_dir=evaluation_cluster_dir,
            cache_seed=case_seed,
            cache_num_train_samples=real_cache_anchor_num_train_samples,
            total_gpu=total_gpu,
            repeat_num=int(max_bw_cfg.get("repeat_num", 50)),
            contention_mode=contention_mode,
            if_dynamic=bool(max_bw_cfg.get("if_dynamic", True)),
            local_top_k=int(max_bw_cfg.get("local_top_k", 10)),
            max_combos_per_distribution=int(max_bw_cfg.get("max_combos_per_distribution", 2048)),
            max_total_combos=int(max_bw_cfg.get("max_total_combos", 200000)),
        )
        runtime_context = prepare_single_contention_runtime_context(
            repeat_num=repeat_num,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            model_path=model_path,
            model_cfg=model_cfg,
            cluster_type=cluster_type,
            training_data_path=training_data_path,
            evaluation_data_path=evaluation_data_path,
            bw_type="contention",
            artifact_dir=artifact_dir,
            if_dynamic=bool(max_bw_cfg.get("if_dynamic", True)),
            random_seed=case_seed,
            contention_mode=contention_mode,
            search_if_real_data=False,
            max_bw_cache_file=max_bw_cache_file,
            adaptive_threshold_policy=adaptive_threshold_policy,
        )

        for case_context in iter_single_contention_case_contexts(
            runtime_context,
            test_num_values=k_values,
            repeat_indices=repeat_indices,
        ):
            real_manager = build_single_contention_real_manager(runtime_context, case_context)
            result = run_single_contention_search_algorithm(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm="BandPilot",
                algo_fn=legacy_improved_searching_algo,
                use_real_data=False,
                real_manager=real_manager,
                job_id=case_context.probe_job_id,
            )
            if result["record"] is None:
                continue

            record = dict(result["record"])
            record.update(
                {
                    "strategy": strategy_name,
                    "sample_size": int(sample_size),
                    "seed": int(seed),
                    "sampling_protocol": sampling_protocol,
                    "predictor_r2": float(predictor_metrics["r2"]),
                    "predictor_mape_percent": float(predictor_metrics["mape_percent"]),
                    "predictor_rmse": float(predictor_metrics["rmse"]),
                }
            )
            records.append(record)

    return pd.DataFrame(records)


def build_summaries(rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate case rows into per-seed, overall, and per-mode summaries."""

    rows = rows.copy()
    rows["elapsed_time_ms"] = rows["elapsed_time"] * 1000.0
    group_prefix = ["sampling_protocol"] if "sampling_protocol" in rows.columns else []

    seed_summary = (
        rows.groupby(
            group_prefix + ["cluster_type", "strategy", "sample_size", "seed"],
            as_index=False,
        )
        .agg(
            case_count=("final_utilization", "size"),
            mean_final_utilization=("final_utilization", "mean"),
            p05_final_utilization=("final_utilization", lambda series: series.quantile(0.05)),
            min_final_utilization=("final_utilization", "min"),
            mean_elapsed_time_ms=("elapsed_time_ms", "mean"),
            predictor_r2=("predictor_r2", "mean"),
            predictor_mape_percent=("predictor_mape_percent", "mean"),
            predictor_rmse=("predictor_rmse", "mean"),
        )
    )

    summary = (
        seed_summary.groupby(
            group_prefix + ["cluster_type", "strategy", "sample_size"],
            as_index=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            mean_final_utilization=("mean_final_utilization", "mean"),
            std_final_utilization=("mean_final_utilization", "std"),
            mean_p05_final_utilization=("p05_final_utilization", "mean"),
            mean_min_final_utilization=("min_final_utilization", "mean"),
            mean_elapsed_time_ms=("mean_elapsed_time_ms", "mean"),
            std_elapsed_time_ms=("mean_elapsed_time_ms", "std"),
            mean_predictor_r2=("predictor_r2", "mean"),
            mean_predictor_mape_percent=("predictor_mape_percent", "mean"),
            mean_predictor_rmse=("predictor_rmse", "mean"),
        )
    )

    seed_mode_summary = (
        rows.groupby(
            group_prefix + ["cluster_type", "contention_mode", "strategy", "sample_size", "seed"],
            as_index=False,
        )
        .agg(
            case_count=("final_utilization", "size"),
            mean_final_utilization=("final_utilization", "mean"),
            mean_elapsed_time_ms=("elapsed_time_ms", "mean"),
        )
    )
    summary_by_mode = (
        seed_mode_summary.groupby(
            group_prefix + ["cluster_type", "contention_mode", "strategy", "sample_size"],
            as_index=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            mean_final_utilization=("mean_final_utilization", "mean"),
            std_final_utilization=("mean_final_utilization", "std"),
            mean_elapsed_time_ms=("mean_elapsed_time_ms", "mean"),
            std_elapsed_time_ms=("mean_elapsed_time_ms", "std"),
        )
    )

    return seed_summary, summary, summary_by_mode


def plot_dispatch_sensitivity(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot dispatch-level sensitivity curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_directory(output_dir)
    plt.rcParams.update(
        {
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
        }
    )

    cluster_labels = {
        "H100_26H100_27H100_28H100_29": "H100 Cluster",
        "Het-4Mix": "Het-4Mix Cluster",
    }
    strategy_styles = {
        "Random": ("o", "#2078B4"),
        "Stratified": ("s", "#FF7F0E"),
        "Worst-Case": ("^", "#D62728"),
    }
    clusters = list(summary["cluster_type"].unique())

    for metric, ylabel, output_name in [
        ("mean_final_utilization", "Mean Final Utilization (%)", "dispatch_sensitivity_utilization"),
        ("mean_elapsed_time_ms", "Mean Search Latency (ms)", "dispatch_sensitivity_latency"),
    ]:
        fig, axes = plt.subplots(1, len(clusters), figsize=(3.6 * len(clusters), 2.9), squeeze=False)
        for idx, cluster_type in enumerate(clusters):
            ax = axes[0, idx]
            cluster_df = summary[summary["cluster_type"] == cluster_type]
            for strategy in cluster_df["strategy"].unique():
                sub = cluster_df[cluster_df["strategy"] == strategy].sort_values("sample_size")
                marker, color = strategy_styles.get(strategy, ("D", "#888888"))
                std_col = "std_final_utilization" if metric == "mean_final_utilization" else "std_elapsed_time_ms"
                std_values = sub[std_col].fillna(0.0).values
                ax.plot(sub["sample_size"], sub[metric], marker=marker, color=color, label=strategy)
                ax.fill_between(
                    sub["sample_size"],
                    sub[metric] - std_values,
                    sub[metric] + std_values,
                    alpha=0.15,
                    color=color,
                )
            ax.set_title(cluster_labels.get(cluster_type, cluster_type))
            ax.set_xlabel("Number of Training Samples")
            ax.set_ylabel(ylabel)
            if idx == 0:
                ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        pdf_path = output_dir / f"{output_name}.pdf"
        fig.savefig(pdf_path)
        fig.savefig(pdf_path.with_suffix(".png"))
        plt.close(fig)


def build_report(
    *,
    summary: pd.DataFrame,
    output_dir: Path,
    sample_sizes: Sequence[int],
) -> Path:
    """Build a Markdown report for the dispatch-level sensitivity sidecar."""

    lines = [
        "# Dispatch-Level Sensitivity Sidecar",
        "",
        "- Evidence type: `simulated`",
        f"- Sampling protocol: `{summary['sampling_protocol'].iloc[0]}`" if "sampling_protocol" in summary.columns else "- Sampling protocol: `independent`",
        "- Case protocol: fixed `single_contention` case stream + BandPilot only",
        "- This sidecar complements predictor-level `R^2/MAPE` with direct `final_utilization` evidence.",
        "",
    ]

    max_sample_size = max(int(value) for value in sample_sizes)
    min_sample_size = min(int(value) for value in sample_sizes)
    for cluster_type in summary["cluster_type"].unique():
        cluster_df = summary[summary["cluster_type"] == cluster_type].copy()
        lines.append(f"## {cluster_type}")
        lines.append("")
        for strategy in cluster_df["strategy"].unique():
            sub = cluster_df[cluster_df["strategy"] == strategy].sort_values("sample_size")
            ref_min = sub[sub["sample_size"] == min_sample_size]
            ref_250 = sub[sub["sample_size"] == 250]
            ref_max = sub[sub["sample_size"] == max_sample_size]
            if ref_min.empty or ref_250.empty or ref_max.empty:
                continue
            row_min = ref_min.iloc[0]
            row_250 = ref_250.iloc[0]
            row_max = ref_max.iloc[0]
            util_delta_100_to_250 = float(row_250["mean_final_utilization"] - row_min["mean_final_utilization"])
            util_delta = float(row_max["mean_final_utilization"] - row_250["mean_final_utilization"])
            latency_delta = float(row_max["mean_elapsed_time_ms"] - row_250["mean_elapsed_time_ms"])
            lines.append(
                f"- `{strategy}`: `n={min_sample_size} -> 250` utilization changes by `{util_delta_100_to_250:+.2f} pt`, "
                f"`250 -> {max_sample_size}` changes by `{util_delta:+.2f} pt`; "
                f"`n=250` mean utilization `{row_250['mean_final_utilization']:.2f}%`,"
                f"`n={max_sample_size}` mean utilization `{row_max['mean_final_utilization']:.2f}%`, "
                f"mean latency changes by `{latency_delta:+.2f} ms`."
            )
            if abs(util_delta) <= 1.0:
                lines.append(f"  Reading: `{strategy}` shows little cluster-level change from `250 -> {max_sample_size}`.")
            elif util_delta > 1.0:
                lines.append(f"  Reading: `{strategy}` improves from `250 -> {max_sample_size}`, so `250` may underfit this slice.")
            else:
                lines.append(f"  Reading: `{strategy}` degrades at the largest budget; inspect per-seed rows before drawing conclusions.")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dispatch-level sidecar for sample-size sensitivity analysis"
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_config.yaml"))
    parser.add_argument("--cluster", type=str, default=None, help="Run only one cluster type")
    parser.add_argument("--sample-sizes", type=str, default="100,250,500")
    parser.add_argument("--strategies", type=str, default="Random,Stratified,Worst-Case")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--num-test-samples", type=int, default=DEFAULT_NUM_TEST_SAMPLES)
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
    parser.add_argument("--contention-modes", type=str, default="idle,common,intensive")
    parser.add_argument("--k-values", type=str, default="4,8,12,16,20,24,28")
    parser.add_argument("--repeat-indices", type=str, default="0,1,2,3,4")
    parser.add_argument("--case-seed", type=int, default=1111)
    parser.add_argument("--real-cache-anchor-num-train-samples", type=int, default=250)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/sensitivity-analysis/artifacts/dispatch_sidecar"),
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
    eval_cfg = config["evaluation"]
    max_bw_cfg = dict(eval_cfg.get("max_bw_offline", {}))
    adaptive_threshold_policy = eval_cfg.get("adaptive_threshold_policy")

    sample_sizes = _parse_csv_list(args.sample_sizes, int) or list(DEFAULT_SAMPLE_SIZES)
    strategies = _parse_csv_list(args.strategies, str) or list(DEFAULT_STRATEGIES)
    repeat_indices = _parse_csv_list(args.repeat_indices, int) or list(DEFAULT_REPEAT_INDICES)
    k_values = _parse_csv_list(args.k_values, int) or list(DEFAULT_K_VALUES)
    contention_modes = _parse_csv_list(args.contention_modes, str) or list(DEFAULT_CONTENTION_MODES)

    cluster_types = [args.cluster] if args.cluster else list(cluster_cfg["cluster_types"])
    child_seeds = derive_child_seeds(args.master_seed, args.num_seeds)

    device_str = args.device or config.get("device", "cuda")
    device = torch.device(device_str)

    ensure_directory(args.output_dir)
    model_root = args.output_dir / "models"
    nested_manifest_root = args.output_dir / "nested_manifests"
    rows_list: List[pd.DataFrame] = []

    for cluster_type in cluster_types:
        total_gpu, switch_config, gpu_bw_dict_list, training_data_path, evaluation_data_path, evaluation_cluster_dir = _load_cluster_runtime(
            config=config,
            cluster_type=cluster_type,
        )

        # Use one shared test set for predictor-level metrics within this cluster.
        set_seed(args.master_seed + 9999)
        test_configs, test_bandwidths = get_random_train_dataset(
            num_samples=args.num_test_samples,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
        )

        for strategy_name in strategies:
            if strategy_name not in STRATEGY_GENERATORS:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            for seed in child_seeds:
                nested_family = None
                if args.sampling_protocol == "nested":
                    nested_family = build_nested_dataset_family(
                        cluster_type=cluster_type,
                        strategy_name=strategy_name,
                        sample_sizes=sample_sizes,
                        seed=int(seed),
                        total_gpu=total_gpu,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        training_data_path=training_data_path,
                        generator_fn=STRATEGY_GENERATORS[strategy_name],
                        compute_bandwidths_fn=_compute_bandwidths,
                        manifest_root=nested_manifest_root,
                        mother_pool_size=args.nested_max_sample_size,
                        protocol_name=args.sampling_protocol,
                    )
                for sample_size in sample_sizes:
                    logger.info(
                        "Dispatch sensitivity | cluster=%s | strategy=%s | n=%s | seed=%s",
                        cluster_type,
                        strategy_name,
                        sample_size,
                        seed,
                    )
                    model_path, artifact_dir, predictor_metrics = train_predictor_variant(
                        cluster_type=cluster_type,
                        strategy_name=strategy_name,
                        sample_size=int(sample_size),
                        seed=int(seed),
                        total_gpu=total_gpu,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        training_data_path=training_data_path,
                        model_cfg=model_cfg,
                        training_cfg=training_cfg,
                        test_configs=test_configs,
                        test_bandwidths=test_bandwidths,
                        device=device,
                        model_root=model_root,
                        gpu_train=None if nested_family is None else nested_family[int(sample_size)][0],
                        bw_train=None if nested_family is None else nested_family[int(sample_size)][1],
                    )
                    rows = evaluate_dispatch_variant(
                        cluster_type=cluster_type,
                        strategy_name=strategy_name,
                        sample_size=int(sample_size),
                        seed=int(seed),
                        model_path=model_path,
                        artifact_dir=artifact_dir,
                        predictor_metrics=predictor_metrics,
                        total_gpu=total_gpu,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        training_data_path=training_data_path,
                        evaluation_data_path=evaluation_data_path,
                        evaluation_cluster_dir=evaluation_cluster_dir,
                        model_cfg=model_cfg,
                        contention_modes=contention_modes,
                        k_values=k_values,
                        repeat_indices=repeat_indices,
                        case_seed=args.case_seed,
                        real_cache_anchor_num_train_samples=args.real_cache_anchor_num_train_samples,
                        max_bw_cfg=max_bw_cfg,
                        adaptive_threshold_policy=adaptive_threshold_policy,
                        sampling_protocol=args.sampling_protocol,
                    )
                    rows_list.append(rows)

    full_rows = pd.concat(rows_list, ignore_index=True)
    rows_csv = args.output_dir / "rows.csv"
    full_rows.to_csv(rows_csv, index=False)

    seed_summary, summary, summary_by_mode = build_summaries(full_rows)
    seed_summary.to_csv(args.output_dir / "seed_summary.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    summary_by_mode.to_csv(args.output_dir / "summary_by_mode.csv", index=False)

    plot_dispatch_sensitivity(summary, args.output_dir)
    report_path = build_report(summary=summary, output_dir=args.output_dir, sample_sizes=sample_sizes)

    print(f"[Rows] {rows_csv}")
    print(f"[Summary] {args.output_dir / 'summary.csv'}")
    print(f"[ByMode] {args.output_dir / 'summary_by_mode.csv'}")
    print(f"[Report] {report_path}")


if __name__ == "__main__":
    main()
