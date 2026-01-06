"""Refactored GPU scheduling entrypoint."""
from __future__ import annotations
"""python main.py --config config/default_config.yaml"""
import argparse
import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

from core.bandwidth import SwitchBandwidthConfig, get_gpu_dict_files, load_gpu_bw_dict
from core.cluster_state import normalize_contention_mode
from evaluation.compare import (

    get_single_dispatch_with_contention_data,
    collect_single_contention_max_bw_data,
    build_single_experiment_filename,
    build_max_bw_cache_filename,
)
from training.trainer import model_train_pipeline
from utils.helpers import ensure_directory, build_artifact_filename, record_active_num_train_samples

# Import algorithm functions
from algorithms.baseline import default_algo, random_algo
from algorithms.eha import eha_search
from algorithms.search import improved_searching_algo, tree_search_only
from algorithms.slurm import slurm_best_fit_algo


def parse_args() -> argparse.Namespace:
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
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gpu_bandwidth_dicts(bandwidth_dir: Path, file_list: List[str]):
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
                        num_train_samples = int(training_cfg.get("num_train_samples", 0))

                        for single_contention in single_contention_list:
                            for single_repeat in single_repeat_list:
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


if __name__ == "__main__":
    #python main.py --config config/default_config.yaml
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
