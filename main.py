"""Public BandPilot training and evaluation entry point.

The module keeps heavyweight ML and data-processing imports inside `main()` so
`python main.py --help` remains a lightweight self-documenting command. This is
important for artifact evaluators who first inspect available CLI options before
installing GPU-specific runtime dependencies.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the public training/evaluation entry point."""

    parser = argparse.ArgumentParser(description="GPU Bandwidth Dispatcher")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train"],
        default="train",
        help="Run mode",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gpu_bandwidth_dicts(bandwidth_dir: Path, file_list: List[str]):
    from core.bandwidth import load_gpu_bw_dict

    dicts = []
    for filename in file_list:
        dicts.append(load_gpu_bw_dict(bandwidth_dir / filename))
    return dicts



def _ensure_list(value):
    """Helper: wrap scalar config into a list for sequential execution."""
    if isinstance(value, list):
        return value
    return [value]


def main() -> None:
    args = parse_args()

    import torch

    from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files
    from core.cluster_state import normalize_contention_mode
    from evaluation.compare import (
        build_max_bw_cache_filename,
        build_single_experiment_filename,
        collect_single_contention_max_bw_data,
        get_single_dispatch_with_contention_data,
    )
    from evaluation.scalability import CURRENT_BENCHMARK_ARTIFACT_DIR
    from evaluation.scalability.benchmark import (
        make_real_cluster_config,
        run_scalability_benchmark_suite,
    )
    from training.sample_sensitivity_experiment import (
        DEFAULT_SAMPLE_SIZES,
        STRATEGY_GENERATORS,
        plot_sensitivity_figures,
        run_sensitivity_experiment,
    )
    from training.trainer import model_train_pipeline
    from utils.helpers import (
        build_artifact_filename,
        ensure_directory,
        record_active_num_train_samples,
    )

    config = load_config(args.config)
    
    # Support random_seed as single value or list
    random_seed_cfg = config.get("random_seed", 123)
    if isinstance(random_seed_cfg, list):
        random_seeds = random_seed_cfg
    else:
        random_seeds = [random_seed_cfg]

    data_cfg = config["data"]
    cluster_cfg = config["cluster"]
    model_cfg = config["model"]
    eval_cfg = config.get("evaluation", {})
    training_cfg = config.get("training", {})

    h100_training_data_path = data_cfg.get("h100_training_data_path")
    h100_evaluation_data_path = data_cfg.get(
        "h100_evaluation_data_path",
        h100_training_data_path,
    )

    bandwidth_dir = Path(data_cfg["bandwidth_dict_dir"])
    model_save_dir = Path(data_cfg["model_save_dir"])
    model_save_dir.mkdir(parents=True, exist_ok=True)
    evaluation_base_dir = Path(data_cfg["evaluation_dir"])
    
    total_gpu = cluster_cfg["total_gpu"]

    # Iterate over random_seeds
    for random_seed in random_seeds:
        # Set seed per experiment group
        set_seed(random_seed)
        logger.info(f"========== Start experiment random_seed={random_seed} ==========")
        scalability_cfg = eval_cfg.get("scalability_benchmark", {})
        benchmark_cluster_configs = []

        for cluster_type in cluster_cfg["cluster_types"]:
            is_h100_cluster = "H100" in cluster_type
            training_data_path = h100_training_data_path
            evaluation_data_path = h100_evaluation_data_path

            switch_config = SwitchBandwidthConfig(
                num_machines=total_gpu // 8,
                cluster_type=cluster_type,
            )
            file_list = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
            gpu_bw_dict_list = load_gpu_bandwidth_dicts(bandwidth_dir, file_list)

            artifact_dir = model_save_dir / cluster_type
            artifact_dir.mkdir(parents=True, exist_ok=True)
            cluster_eval_dir = evaluation_base_dir / cluster_type
            ensure_directory(cluster_eval_dir)

            device = torch.device(config.get("device", "cuda"))
            
            # Train or load existing model based on enable_training
            enable_training = training_cfg.get("enable_training", True)
            num_train_samples = int(training_cfg.get("num_train_samples", 0))
            if enable_training:
                mse, mae, model_path = model_train_pipeline(
                    total_gpu,
                    gpu_bw_dict_list,
                    switch_config,
                    training_data_path,
                    artifact_dir,
                    device,
                    config,
                )
                print(f"[seed={random_seed}] {cluster_type} training done, MSE={mse:.4f}, MAE={mae:.4f}")
            else:
                # Skip training; use existing model from default path
                if model_cfg["type"] == "simple":
                    model_path = artifact_dir / build_artifact_filename(
                        "simple_bandwidth_predictor", num_train_samples, ".pth"
                    )
                else:
                    model_path = artifact_dir / build_artifact_filename(
                        "bandwidth_predictor", num_train_samples, ".pth"
                    )
                
                # Ensure model file exists
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Model file not found: {model_path}\n"
                        f"Ensure the model is trained at the default path or set training.enable_training=true."
                    )
                logger.info(
                    f"[seed={random_seed}] {cluster_type} skip training, using existing model: {model_path}"
                )
                print(f"[seed={random_seed}] {cluster_type} using existing model: {model_path}")
                record_active_num_train_samples(artifact_dir, num_train_samples)

            if scalability_cfg.get("enable", False):
                benchmark_cfg = make_real_cluster_config(
                    cluster_type=cluster_type,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    model_path=model_path,
                    model_cfg=model_cfg,
                    training_data_path=training_data_path,
                    evaluation_data_path=evaluation_data_path,
                    artifact_dir=artifact_dir,
                    device=device,
                    adaptive_runtime_policy=scalability_cfg.get("adaptive_runtime_policy"),
                    hu_unit_gate=scalability_cfg.get("hu_unit_gate"),
                )
                benchmark_cluster_configs.append(benchmark_cfg)

            # ==================== Sensitivity Analysis ====================
            sens_cfg = eval_cfg.get("sensitivity_analysis", {})
            if sens_cfg.get("enable", False):
                import pandas as pd
                sens_output = Path(
                    sens_cfg.get(
                        "output_dir",
                        "./evaluation/sensitivity-analysis/artifacts/predictor-level",
                    )
                )
                ensure_directory(sens_output)
                sens_df = run_sensitivity_experiment(
                    cluster_type=cluster_type,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    training_data_path=training_data_path,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    sample_sizes=sens_cfg.get("sample_sizes", DEFAULT_SAMPLE_SIZES),
                    strategies=sens_cfg.get("strategies", list(STRATEGY_GENERATORS.keys())),
                    num_seeds=sens_cfg.get("num_seeds", 5),
                    master_seed=sens_cfg.get("master_seed", 42),
                    num_test_samples=sens_cfg.get("num_test_samples", 2500),
                    device=device,
                    output_dir=sens_output,
                )
                csv_path = sens_output / f"sensitivity_{cluster_type}.csv"
                sens_df.to_csv(csv_path, index=False)
                print(f"[seed={random_seed}] Sensitivity results saved to {csv_path}")
                # Plot after all clusters finish (handled outside the loop)

            max_bw_offline_cfg = eval_cfg.get("max_bw_offline", {})
            max_bw_cache_files = {}
            if max_bw_offline_cfg:
                # Support repeat_num / contention_mode as lists; iterate in order
                max_bw_repeat_list = _ensure_list(
                    max_bw_offline_cfg.get("repeat_num", eval_cfg.get("repeat_num", 1))
                )
                # Normalize casing/whitespace to keep cache filenames consistent
                max_bw_contention_list = [
                    normalize_contention_mode(mode)
                    for mode in _ensure_list(max_bw_offline_cfg.get("contention_mode", "common"))
                ]
                max_bw_if_dynamic = max_bw_offline_cfg.get("if_dynamic", eval_cfg.get("if_dynamic", True))
                max_bw_search_real = max_bw_offline_cfg.get("search_if_real_data", False)
                local_top_k = int(max_bw_offline_cfg.get("local_top_k", 3))
                max_combos_per_distribution = int(max_bw_offline_cfg.get("max_combos_per_distribution", 2048))
                max_total_combos = int(max_bw_offline_cfg.get("max_total_combos", 100000))
                num_train_samples = int(training_cfg.get("num_train_samples", 0))

                for max_bw_contention in max_bw_contention_list:
                    for max_bw_repeat in max_bw_repeat_list:
                        max_bw_filename = build_max_bw_cache_filename(
                            random_seed=random_seed,
                            num_train_samples=num_train_samples,
                            total_gpu=total_gpu,
                            repeat_num=max_bw_repeat,
                            if_dynamic=max_bw_if_dynamic,
                            contention_mode=max_bw_contention,
                            search_if_real_data=max_bw_search_real,
                            local_top_k=local_top_k,
                            max_combos_per_distribution=max_combos_per_distribution,
                            max_total_combos=max_total_combos,
                        )
                        max_bw_cache_file = cluster_eval_dir / max_bw_filename
                        max_bw_cache_files[(max_bw_contention, max_bw_repeat)] = max_bw_cache_file

                        if max_bw_offline_cfg.get("enable"):
                            logger.info(
                                "========== Start offline max_bw collection (cluster=%s, seed=%s, contention=%s, repeat=%s) ==========",
                                cluster_type,
                                random_seed,
                                max_bw_contention,
                                max_bw_repeat,
                            )
                            max_bw_df = collect_single_contention_max_bw_data(
                                repeat_num=max_bw_repeat,
                                total_gpu=total_gpu,
                                gpu_bw_dict_list=gpu_bw_dict_list,
                                switch_config=switch_config,
                                model_path=model_path,
                                model_cfg=model_cfg,
                                cluster_type=cluster_type,
                                training_data_path=training_data_path,
                                evaluation_data_path=evaluation_data_path,
                                artifact_dir=artifact_dir,
                                if_dynamic=max_bw_if_dynamic,
                                random_seed=random_seed,
                                contention_mode=max_bw_contention,
                                search_if_real_data=max_bw_search_real,
                                max_bw_options=max_bw_offline_cfg,
                            )
                            max_bw_df.to_csv(max_bw_cache_file, index=False)
                            logger.info("max_bw cache written to %s", max_bw_cache_file)


            # ==================== Utilization evaluation ====================
            if eval_cfg.get("enable_utilization") or eval_cfg.get("enable_accumulation") or eval_cfg.get("enable_single_contention"):
                if model_cfg["type"] == "simple":
                    print("Model type is simple; skip single-GPU comparison evaluations (util/acc/contention).")
                else:
                    cluster_eval_dir = evaluation_base_dir / cluster_type
                    ensure_directory(cluster_eval_dir)
                    if eval_cfg.get("enable_single_contention"):
                        single_cont_cfg = eval_cfg.get("single_contention", {})
                        if not max_bw_cache_files:
                            raise ValueError(
                                "single_dispatch_with_contention requires max_bw cache; "
                                "configure and run config.evaluation.max_bw_offline first."
                            )

                        single_repeat_list = _ensure_list(
                            single_cont_cfg.get("repeat_num", eval_cfg.get("repeat_num", 1))
                        )
                        # Keep labels aligned with max_bw cache to match correctly
                        single_contention_list = [
                            normalize_contention_mode(mode)
                            for mode in _ensure_list(single_cont_cfg.get("contention_mode", "common"))
                        ]
                        single_if_dynamic = single_cont_cfg.get("if_dynamic", eval_cfg.get("if_dynamic", True))
                        single_search_real = single_cont_cfg.get("search_if_real_data", False)
                        adaptive_runtime_policy = dict(
                            single_cont_cfg.get("adaptive_runtime_policy", {})
                        )
                        num_train_samples = int(training_cfg.get("num_train_samples", 0))

                        for single_repeat in single_repeat_list:
                            for single_contention in single_contention_list:
                                max_bw_cache_file = max_bw_cache_files.get((single_contention, single_repeat))
                                if max_bw_cache_file is None:
                                    # If not generated this run, locate existing cache by naming rule
                                    max_bw_cache_file = cluster_eval_dir / build_max_bw_cache_filename(
                                        random_seed=random_seed,
                                        num_train_samples=num_train_samples,
                                        total_gpu=total_gpu,
                                        repeat_num=single_repeat,
                                        if_dynamic=single_if_dynamic,
                                        contention_mode=single_contention,
                                        search_if_real_data=max_bw_offline_cfg.get("search_if_real_data", False),
                                        local_top_k=int(max_bw_offline_cfg.get("local_top_k", 3)),
                                        max_combos_per_distribution=int(max_bw_offline_cfg.get("max_combos_per_distribution", 2048)),
                                        max_total_combos=int(max_bw_offline_cfg.get("max_total_combos", 100000)),
                                    )

                                if not max_bw_cache_file.exists():
                                    raise FileNotFoundError(
                                        f"No max_bw cache found for contention_mode={single_contention}, repeat_num={single_repeat}: {max_bw_cache_file}"
                                    )

                                single_cont_df = get_single_dispatch_with_contention_data(
                                    repeat_num=single_repeat,
                                    total_gpu=total_gpu,
                                    gpu_bw_dict_list=gpu_bw_dict_list,
                                    switch_config=switch_config,
                                    model_path=model_path,
                                    model_cfg=model_cfg,
                                    cluster_type=cluster_type,
                                    training_data_path=training_data_path,
                                    evaluation_data_path=evaluation_data_path,
                                    bw_type=single_contention,
                                    artifact_dir=artifact_dir,
                                    if_dynamic=single_if_dynamic,
                                    random_seed=random_seed,
                                    contention_mode=single_contention,
                                    search_if_real_data=single_search_real,
                                    max_bw_cache_file=max_bw_cache_file,
                                    adaptive_threshold_policy=single_cont_cfg.get(
                                        "adaptive_threshold_policy",
                                        eval_cfg.get("adaptive_threshold_policy"),
                                    ),
                                    adaptive_runtime_policy=adaptive_runtime_policy,
                                    hu_unit_gate=single_cont_cfg.get("hu_unit_gate"),
                                    baseline_config=single_cont_cfg.get("baselines"),
                                )

                                single_cont_stem = build_single_experiment_filename(
                                    metric_type="contention",
                                    random_seed=random_seed,
                                    contention_mode=single_contention,
                                    num_train_samples=num_train_samples,
                                    if_dynamic=single_if_dynamic,
                                    total_gpu=total_gpu,
                                    repeat_num=single_repeat,
                                )
                                single_cont_file = cluster_eval_dir / f"{single_cont_stem}.csv"
                                single_cont_df.to_csv(single_cont_file, index=False)
                                print(
                                    f"single_dispatch_with_contention results saved to {single_cont_file} "
                                    f"(contention_mode={single_contention}, repeat_num={single_repeat})"
                                )

        if scalability_cfg.get("enable", False) and benchmark_cluster_configs:
            output_dir = Path(
                scalability_cfg.get("output_dir", str(CURRENT_BENCHMARK_ARTIFACT_DIR))
            )
            ensure_directory(output_dir)
            # The scalability benchmark may build larger virtual clusters
            # from each configured cluster template.
            run_scalability_benchmark_suite(
                cluster_configs=benchmark_cluster_configs,
                benchmark_cfg=scalability_cfg,
                output_dir=output_dir,
                random_seed=random_seed,
            )
            print(f"[seed={random_seed}] scalability benchmark artifacts saved to {output_dir}")


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
