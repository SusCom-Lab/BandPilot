"""重构后的GPU调度主入口。"""
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
from evaluation.compare import (
    get_compare_accumulation_data,
    get_compare_utilization_data,
    get_multi_tenant_compare_data,
    build_multi_tenant_filename,
)
from training.trainer import model_train_pipeline, simple_model_train_pipeline
from utils.helpers import ensure_directory

# 导入所有需要的算法函数
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
        help="配置文件路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train"],
        default="train",
        help="运行模式",
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


def _create_multi_tenant_algorithm_configs() -> List[dict]:
    """创建多租户仿真评估所需的算法配置列表。
    
    返回包含所有7个算法配置的列表，每个配置包含：
    - 'name': 算法名称（字符串）
    - 'algo': 算法函数（Callable）
    - 'search_if_real_data': 搜索阶段使用的评估模式（bool，可选，默认False）
    
    Returns:
        算法配置列表，包含以下算法：
        - BandDisp: improved_searching_algo (模型预测模式)
        - Default: default_algo (简单基线算法)
        - Tree: tree_search_only (模型预测模式)
        - EHA: eha_search (模型预测模式)
        - Topo: slurm_best_fit_algo (拓扑感知算法)
        - Random: random_algo (随机基线算法)
        - UpperBandDisp: improved_searching_algo (真实数据模式，作为上界)
    """
    return [
        {
            "name": "BandDisp",
            "algo": improved_searching_algo,
            "search_if_real_data": False,
            "global_mode": True,
        },
        {
            "name": "Default",
            "algo": default_algo,
            "search_if_real_data": False,  # default_algo 不使用预测，但保持一致性
        },
        {
            "name": "Topo",
            "algo": slurm_best_fit_algo,
            "search_if_real_data": False,  # slurm_best_fit_algo 不使用预测，但保持一致性
        },
        {
            "name": "Random",
            "algo": random_algo,
            "search_if_real_data": False,  # random_algo 不使用预测，但保持一致性
        },
        {
            "name": "UpperBandDisp",
            "algo": improved_searching_algo,
            "search_if_real_data": True,  # 使用真实数据作为上界
            "global_mode": True,
        },
        # {
        #     "name": "GroundTruth",
        #     "algo": "MINLP",
        #     "search_if_real_data": True,
        # },
        {
            "name": "BandDisp_Greedy",
            "algo": improved_searching_algo,
            "search_if_real_data": False,
            "global_mode": False,
        },
        # {
        #     "name": "EHA",
        #     "algo": eha_search,
        #     "search_if_real_data": False,
        #     "global_mode": True,
        # },
        # {
        #     "name": "Tree_search",
        #     "algo": tree_search_only,
        #     "search_if_real_data": False,
        #     "global_mode": True,
        # },
        # 新增：暴力搜索 GroundTruth（用于小规模问题验证）
        # {
        #     "name": "GroundTruth_BruteForce",
        #     "algo": "BRUTE_FORCE",
        #     "search_if_real_data": True,
        # },
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # 支持 random_seed 为单个值或列表
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

    data_path = data_cfg["h100_data_path"]
    bandwidth_dir = Path(data_cfg["bandwidth_dict_dir"])
    model_save_dir = Path(data_cfg["model_save_dir"])
    model_save_dir.mkdir(parents=True, exist_ok=True)
    evaluation_base_dir = Path(data_cfg["evaluation_dir"])
    
    total_gpu = cluster_cfg["total_gpu"]
    bw_switch = cluster_cfg["bw_switch"]
    bw_type = str(bw_switch)

    # 遍历所有 random_seed
    for random_seed in random_seeds:
        # 为每组实验设置随机种子
        set_seed(random_seed)
        logger.info(f"========== 开始 random_seed={random_seed} 的实验 ==========")

        for cluster_type in cluster_cfg["cluster_types"]:
            switch_config = SwitchBandwidthConfig(
                num_machines=total_gpu // 8,
                cluster_type=cluster_type,
            )
            file_list = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)
            gpu_bw_dict_list = load_gpu_bandwidth_dicts(bandwidth_dir, file_list)

            artifact_dir = model_save_dir / cluster_type
            artifact_dir.mkdir(parents=True, exist_ok=True)

            device = torch.device(config.get("device", "cuda"))
            if model_cfg["type"] == "simple":
                mse, mae, model_path = simple_model_train_pipeline(
                    total_gpu,
                    gpu_bw_dict_list,
                    switch_config,
                    data_path,
                    artifact_dir,
                    device,
                    config,
                )
            else:
                mse, mae, model_path = model_train_pipeline(
                    total_gpu,
                    gpu_bw_dict_list,
                    switch_config,
                    data_path,
                    artifact_dir,
                    device,
                    config,
                )
            print(f"[seed={random_seed}] {cluster_type} 训练完成，MSE={mse:.4f}, MAE={mae:.4f}")

            # ==================== 多租户仿真评估 ====================
            # 仅在模型类型为 'full' 且启用多租户仿真时执行
            if eval_cfg.get("enable_multi_tenant") and model_cfg["type"] == "full":
                logger.info("========== 开始多租户仿真评估（多算法对比） ==========")
                cluster_eval_dir = evaluation_base_dir / cluster_type
                ensure_directory(cluster_eval_dir)
                
                # 获取多租户仿真配置
                multi_tenant_cfg = eval_cfg.get("multi_tenant", {})
                workload_mode = multi_tenant_cfg.get("workload_mode", "fixed_sum")
                total_gpu_sum = multi_tenant_cfg.get("total_gpu_sum", total_gpu)
                num_jobs = multi_tenant_cfg.get("num_jobs", 10)
                job_sizes = multi_tenant_cfg.get("job_sizes", [1, 2, 4, 8])

                # 支持 repeat_num 与 contention_mode 为单个值或列表
                repeat_cfg = multi_tenant_cfg.get("repeat_num", 1)
                if isinstance(repeat_cfg, list):
                    repeat_nums = repeat_cfg
                else:
                    repeat_nums = [repeat_cfg]

                contention_cfg = multi_tenant_cfg.get("contention_mode", "intensive")
                if isinstance(contention_cfg, list):
                    contention_modes = contention_cfg
                else:
                    contention_modes = [contention_cfg]
                
                # 创建算法配置列表
                algorithm_configs = _create_multi_tenant_algorithm_configs()
                logger.info(f"将对比 {len(algorithm_configs)} 个算法: {[cfg['name'] for cfg in algorithm_configs]}")

                # 按照 random_seed × contention_mode × repeat_num 依次运行实验
                for contention_mode in contention_modes:
                    for repeat_num in repeat_nums:
                        logger.info(
                            f"开始多租户仿真: seed={random_seed}, "
                            f"contention_mode={contention_mode}, repeat_num={repeat_num}"
                        )
                        # 运行多算法对比的多租户仿真
                        # 传递 random_seed 确保所有算法使用相同的workload序列进行公平对比
                        multi_tenant_df = get_multi_tenant_compare_data(
                            total_gpu=total_gpu,
                            gpu_bw_dict_list=gpu_bw_dict_list,
                            switch_config=switch_config,
                            model_path=model_path,
                            model_cfg=model_cfg,
                            cluster_type=cluster_type,
                            data_path=data_path,
                            artifact_dir=artifact_dir,
                            device=device,
                            contention_mode=contention_mode,
                            workload_mode=workload_mode,
                            total_gpu_sum=total_gpu_sum,
                            num_jobs=num_jobs,
                            job_sizes=job_sizes,
                            repeat_num=repeat_num,
                            algorithm_configs=algorithm_configs,
                            random_seed=random_seed,
                        )
                        
                        # 保存 CSV 文件，按照约定命名规则：
                        # MTS_{random_seed}RS_{num_train_samples}TD_{contention_mode}CM_{repeat_num}RN
                        num_train_samples = int(training_cfg.get("num_train_samples", 0))
                        file_stem = build_multi_tenant_filename(
                            random_seed=random_seed,
                            num_train_samples=num_train_samples,
                            contention_mode=contention_mode,
                            repeat_num=repeat_num,
                        )
                        multi_tenant_file = cluster_eval_dir / f"{file_stem}.csv"
                        multi_tenant_df.to_csv(multi_tenant_file, index=False)
                        print(
                            f"多租户仿真结果已保存至 {multi_tenant_file}，"
                            f"共 {len(multi_tenant_df)} 条记录，"
                            f"{multi_tenant_df['algorithm_name'].nunique()} 个算法"
                        )
                logger.info("========== 多租户仿真评估完成 ==========")

            # ==================== 利用率/累积差距评估 ====================
            if eval_cfg.get("enable_utilization") or eval_cfg.get("enable_accumulation"):
                if model_cfg["type"] == "simple":
                    print("当前模型类型为 simple，跳过利用率/累积比较评估。")
                else:
                    cluster_eval_dir = evaluation_base_dir / cluster_type
                    ensure_directory(cluster_eval_dir)
                    if eval_cfg.get("enable_utilization"):
                        util_df = get_compare_utilization_data(
                            repeat_num=eval_cfg["repeat_num"],
                            total_gpu=total_gpu,
                            gpu_bw_dict_list=gpu_bw_dict_list,
                            switch_config=switch_config,
                            model_path=model_path,
                            model_cfg=model_cfg,
                            cluster_type=cluster_type,
                            data_path=data_path,
                            bw_type=bw_type,
                            artifact_dir=artifact_dir,
                            if_dynamic=eval_cfg.get("if_dynamic", True),
                            random_seed=random_seed,
                        )
                        util_file = (
                            cluster_eval_dir
                            / f"Part_mean_{bw_type}_bw_{total_gpu}dim_dynamic{eval_cfg.get('if_dynamic', True)}.csv"
                        )
                        util_df.to_csv(util_file, index=False)
                        print(f"利用率比较结果已保存至 {util_file}")

                    if eval_cfg.get("enable_accumulation"):
                        acc_df = get_compare_accumulation_data(
                            repeat_num=eval_cfg["repeat_num"],
                            total_gpu=total_gpu,
                            gpu_bw_dict_list=gpu_bw_dict_list,
                            switch_config=switch_config,
                            model_path=model_path,
                            model_cfg=model_cfg,
                            cluster_type=cluster_type,
                            data_path=data_path,
                            bw_type=bw_type,
                            artifact_dir=artifact_dir,
                            if_dynamic=eval_cfg.get("if_dynamic", True),
                            random_seed=random_seed,
                        )
                        acc_file = (
                            cluster_eval_dir
                            / f"Part_sum_{bw_type}_bw_{total_gpu}dim_dynamic{eval_cfg.get('if_dynamic', True)}.csv"
                        )
                        acc_df.to_csv(acc_file, index=False)
                        print(f"累积差距结果已保存至 {acc_file}")


if __name__ == "__main__":
    #python main.py --config config/default_config.yaml
    main()
