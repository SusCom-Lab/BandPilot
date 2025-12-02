"""算法比较与统计。"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 是可选依赖
    tqdm = None

from algorithms.baseline import default_algo, random_algo
from algorithms.eha import eha_search,eha_search_old
from algorithms.search import improved_searching_algo, tree_search_only
from algorithms.slurm import slurm_best_fit_algo
from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values
from core.topology import (
    build_composite_topo_matrix,
    convert_cluster_type_to_node_configs,
    create_gpu_to_node_map,
)
from evaluation.metrics import find_max_bw_for_k_gpus
from models.bandwidth_predictor import BandwidthPredictor

logger = logging.getLogger(__name__)


def _load_predictor(model_path: Path, device: torch.device, model_cfg: dict) -> BandwidthPredictor:
    model = BandwidthPredictor(
        input_dim=model_cfg.get("input_dim", 1),
        node_count_embedding_dim=model_cfg.get("node_count_embedding_dim", 8),
        total_count_feature_dim=model_cfg.get("total_count_feature_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 3),
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.05),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _run_and_record(algo_fn, result_dict: Dict[str, list], *args, **kwargs):
    start = time.time()
    combo = algo_fn(*args, **kwargs)
    elapsed = time.time() - start
    result_dict["time"].append(elapsed)
    return combo


def _sample_available_gpu(total_gpu: int, test_num: int, if_dynamic: bool, random_seed: Optional[int] = None) -> np.ndarray:
    """采样可用GPU。
    
    Args:
        total_gpu: 总GPU数量
        test_num: 测试的GPU数量下限
        if_dynamic: 是否动态采样
        random_seed: 随机种子（可选），如果提供则设置随机种子
    
    Returns:
        可用GPU索引数组
    """
    if if_dynamic:
        if random_seed is not None:
            np.random.seed(random_seed)
        avail_gpu_num = np.random.randint(test_num, total_gpu + 1)
        return np.random.choice(total_gpu, avail_gpu_num, replace=False)
    return np.arange(total_gpu)


def get_compare_utilization_data(
    repeat_num: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    cluster_type: str,
    data_path: str,
    bw_type: str,
    artifact_dir: Path,
    if_dynamic: bool = True,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """比较多种算法的利用率表现。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_predictor(model_path, device, model_cfg)

    node_configs = convert_cluster_type_to_node_configs(cluster_type, total_gpu)
    topo_matrix, _ = build_composite_topo_matrix(node_configs)
    gpu_to_node_map = create_gpu_to_node_map(node_configs)

    results = []
    for test_num in range(2, total_gpu):
        print(f"--------------------------------当前测试的GPU数量为: {test_num}--------------------------------")
        algo_results = {
            name: {"bw": [], "time": []}
            for name in ["BandDisp", "Default", "Tree", "EHA", "Topo", "Random", "UpperBandDisp"]
        }
        for repeat_idx in range(repeat_num):
            # if repeat_idx % 10 == 0:
            #     print(f"  重复测试 {repeat_idx + 1}/{repeat_num}")
            # 为每次 repeat 使用不同的随机种子
            current_seed = None
            if random_seed is not None:
                current_seed = random_seed + repeat_idx
                # 设置 numpy 和 python random 的随机种子
                np.random.seed(current_seed)
                import random
                random.seed(current_seed)
            
            avail_gpu = _sample_available_gpu(total_gpu, test_num, if_dynamic, random_seed=current_seed)

            max_bw, _ = find_max_bw_for_k_gpus(
                test_num, gpu_bw_dict_list, total_gpu, switch_config, avail_gpu, data_path
            )
            if max_bw <= 0:
                continue

            # BandDisp: 使用模型预测（if_real_data=False，默认值）
            combo = _run_and_record(
                lambda *a, **kw: improved_searching_algo(*a, **kw),
                algo_results["BandDisp"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["BandDisp"]["bw"].append(bw / max_bw * 100)
                logger.info(f"BandDisp 算法完成，当前利用率: {bw / max_bw * 100:.2f}%")

            combo = _run_and_record(default_algo, algo_results["Default"], total_gpu, avail_gpu, test_num)
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Default"]["bw"].append(bw / max_bw * 100)
                logger.info(f"Default 算法完成，当前利用率: {bw / max_bw * 100:.2f}%")

            # Tree: 使用模型预测（if_real_data=False，默认值）
            combo = _run_and_record(
                lambda *a, **kw: tree_search_only(*a, **kw),
                algo_results["Tree"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Tree"]["bw"].append(bw / max_bw * 100)
                logger.info(f"Tree 算法完成，当前利用率: {bw / max_bw * 100:.2f}%")

            # EHA: 使用模型预测（if_real_data=False，默认值）
            combo = _run_and_record(
                lambda *a, **kw: eha_search(*a, **kw),
                algo_results["EHA"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["EHA"]["bw"].append(bw / max_bw * 100)
                logger.info(f"EHA 算法完成，当前利用率: {bw / max_bw * 100:.2f}%")

            combo = _run_and_record(
                lambda *a, **kw: slurm_best_fit_algo(*a, **kw),
                algo_results["Topo"],
                total_gpu,
                avail_gpu,
                test_num,
                topo_matrix,
                gpu_to_node_map,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Topo"]["bw"].append(bw / max_bw * 100)
                logger.info(f"Topo 算法完成，当前利用率: {bw / max_bw * 100:.2f}%")

            combo = _run_and_record(random_algo, algo_results["Random"], total_gpu, avail_gpu, test_num)
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Random"]["bw"].append(bw / max_bw * 100)
                logger.info(f"Random 算法完成，当前利用率: {bw / max_bw * 100:.2f}%")

            # UpperBandDisp: 使用真实数据计算（if_real_data=True）
            combo = _run_and_record(
                lambda *a, **kw: improved_searching_algo(*a, **kw, if_real_data=True),
                algo_results["UpperBandDisp"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["UpperBandDisp"]["bw"].append(bw / max_bw * 100)
                logger.info(f"UpperBandDisp 算法完成，当前利用率: {bw / max_bw * 100:.2f}%")

        summary = {
            "test_num": test_num,
            "total_gpu": total_gpu,
            "bw_type": bw_type,
            "cluster_type": cluster_type,
            "if_dynamic": if_dynamic,
        }
        for key, values in algo_results.items():
            summary[key] = np.mean(values["bw"]) if values["bw"] else 0
            summary[f"{key}_time"] = np.mean(values["time"]) if values["time"] else 0
        results.append(summary)

    return pd.DataFrame(results)


def get_compare_accumulation_data(
    repeat_num: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    cluster_type: str,
    data_path: str,
    bw_type: str,
    artifact_dir: Path,
    if_dynamic: bool = True,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """统计不同算法与最优带宽之间的差距。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_predictor(model_path, device, model_cfg)

    node_configs = convert_cluster_type_to_node_configs(cluster_type, total_gpu)
    topo_matrix, _ = build_composite_topo_matrix(node_configs)
    gpu_to_node_map = create_gpu_to_node_map(node_configs)

    results = []
    for test_num in range(2, total_gpu):
        logger.info(f"当前测试的GPU数量为: {test_num}")
        algo_results = {
            name: {"gap": [], "time": []}
            for name in ["BandDisp", "Default", "Tree", "EHA", "Topo", "Random", "UpperBandDisp"]
        }
        for repeat_idx in range(repeat_num):
            logger.info(f"  重复测试 {repeat_idx + 1}/{repeat_num}")
            # 为每次 repeat 使用不同的随机种子
            current_seed = None
            if random_seed is not None:
                current_seed = random_seed + repeat_idx
                # 设置 numpy 和 python random 的随机种子
                np.random.seed(current_seed)
                import random
                random.seed(current_seed)
            
            avail_gpu = _sample_available_gpu(total_gpu, test_num, if_dynamic, random_seed=current_seed)

            max_bw, _ = find_max_bw_for_k_gpus(
                test_num, gpu_bw_dict_list, total_gpu, switch_config, avail_gpu, data_path
            )
            if max_bw <= 0:
                continue

            # BandDisp: 使用模型预测（if_real_data=False，默认值）
            combo = _run_and_record(
                lambda *a, **kw: improved_searching_algo(*a, **kw),
                algo_results["BandDisp"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["BandDisp"]["gap"].append(max_bw - bw)
                logger.info(f"BandDisp 算法完成，当前累积差距: {max_bw - bw:.2f}")

            combo = _run_and_record(default_algo, algo_results["Default"], total_gpu, avail_gpu, test_num)
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Default"]["gap"].append(max_bw - bw)
                logger.info(f"Default 算法完成，当前累积差距: {max_bw - bw:.2f}")

            # Tree: 使用模型预测（if_real_data=False，默认值）
            combo = _run_and_record(
                lambda *a, **kw: tree_search_only(*a, **kw),
                algo_results["Tree"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Tree"]["gap"].append(max_bw - bw)
                logger.info(f"Tree 算法完成，当前累积差距: {max_bw - bw:.2f}")

            # EHA: 使用模型预测（if_real_data=False，默认值）
            combo = _run_and_record(
                lambda *a, **kw: eha_search(*a, **kw),
                algo_results["EHA"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["EHA"]["gap"].append(max_bw - bw)
                logger.info(f"EHA 算法完成，当前累积差距: {max_bw - bw:.2f}")

            combo = _run_and_record(
                lambda *a, **kw: slurm_best_fit_algo(*a, **kw),
                algo_results["Topo"],
                total_gpu,
                avail_gpu,
                test_num,
                topo_matrix,
                gpu_to_node_map,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Topo"]["gap"].append(max_bw - bw)
                logger.info(f"Topo 算法完成，当前累积差距: {max_bw - bw:.2f}")

            combo = _run_and_record(random_algo, algo_results["Random"], total_gpu, avail_gpu, test_num)
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["Random"]["gap"].append(max_bw - bw)
                logger.info(f"Random 算法完成，当前累积差距: {max_bw - bw:.2f}")

            # UpperBandDisp: 使用真实数据计算（if_real_data=True）
            combo = _run_and_record(
                lambda *a, **kw: improved_searching_algo(*a, **kw, if_real_data=True),
                algo_results["UpperBandDisp"],
                total_gpu,
                avail_gpu,
                model,
                test_num,
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                data_path,
                device,
                artifact_dir,
            )
            if combo is not None:
                bw, _, _ = calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)
                algo_results["UpperBandDisp"]["gap"].append(max_bw - bw)
                logger.info(f"UpperBandDisp 算法完成，当前累积差距: {max_bw - bw:.2f}")

        summary = {
            "test_num": test_num,
            "total_gpu": total_gpu,
            "bw_type": bw_type,
            "cluster_type": cluster_type,
            "if_dynamic": if_dynamic,
        }
        for key, values in algo_results.items():
            summary[key] = np.mean(values["gap"]) if values["gap"] else 0
            summary[f"{key}_time"] = np.mean(values["time"]) if values["time"] else 0
        results.append(summary)

    return pd.DataFrame(results)


def get_multi_tenant_compare_data(
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    cluster_type: str,
    data_path: str,
    artifact_dir: Path,
    device: torch.device,
    contention_mode: str = "intensive",
    workload_mode: str = "fixed_sum",
    total_gpu_sum: int = 32,
    num_jobs: int = 10,
    job_sizes: List[int] = None,
    repeat_num: int = 1,
    algorithm_configs: List[dict] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """运行多个算法的多租户仿真并对比结果。
    
    该函数接受多个算法配置，对每个算法运行多租户仿真，并合并结果以便对比。
    
    Args:
        total_gpu: 集群总GPU数量
        gpu_bw_dict_list: GPU带宽字典列表
        switch_config: 交换机配置
        model_path: 模型权重文件路径
        model_cfg: 模型配置字典
        cluster_type: 集群类型
        data_path: 带宽数据文件路径
        artifact_dir: 模型和scaler文件所在目录
        device: PyTorch设备（CPU或CUDA）
        workload_mode: 工作负载生成模式（'fixed_sum' 或 'random'）
        total_gpu_sum: fixed_sum 模式下的总GPU数
        num_jobs: random 模式下的任务数量
        job_sizes: 允许的任务大小列表，默认为 [1, 2, 4, 8]
        repeat_num: 重复仿真次数
        algorithm_configs: 算法配置列表，每个配置包含：
            - 'name': 算法名称（字符串）
            - 'algo': 算法函数（Callable）
            - 'search_if_real_data': 搜索阶段使用的评估模式（bool，可选，默认False）
        contention_mode: 争用模式（intensive/common/idle），透传给所有 ClusterStateManager
        random_seed: 随机种子（可选），用于生成统一的workload序列，确保所有算法使用相同的workload进行公平对比
    
    Returns:
        pandas DataFrame，包含所有算法的结果，额外添加以下列：
        - algorithm_name: 算法名称
        - search_mode: 搜索阶段使用的评估模式（'real_data' 或 'model_prediction'）
    """
    from evaluation.multi_tenant_sim import (
        run_multi_tenant_simulation,
        run_multi_tenant_simulation_offline_minlp,
        create_search_algo_adapter,
        _generate_workload_fixed_sum,
        _generate_workload_random,
    )
    import random

    if tqdm is None:
        raise ImportError("tqdm 未安装，无法显示进度条。请运行 `pip install tqdm` 后重试。")
    
    if algorithm_configs is None:
        # 默认算法配置：使用 improved_searching_algo（模型预测和真实数据两个版本）
        from algorithms.search import improved_searching_algo
        
        algorithm_configs = [
            {
                "name": "BandDisp_Model",
                "algo": improved_searching_algo,
                "search_if_real_data": False,
            },
            {
                "name": "BandDisp_GroundTruth",
                "algo": improved_searching_algo,
                "search_if_real_data": True,
            },
            # 新增：离线全局最优 GroundTruth（MINLP/B&B）
            {
                "name": "GroundTruth",
                "algo": "MINLP",
                "search_if_real_data": True,
            },
            # 新增：暴力搜索 GroundTruth（用于小规模问题验证）
            # {
            #     "name": "GroundTruth_BruteForce",
            #     "algo": "BRUTE_FORCE",
            #     "search_if_real_data": True,
            # },
        ]
    
    if job_sizes is None:
        job_sizes = [1, 2, 4, 8]
    
    # ==================== 关键修复：预先生成统一的workload序列 ====================
    # 为了确保所有算法在完全相同的workload下进行公平对比，必须预先生成统一的workload序列
    # 这样所有算法都会使用相同的任务序列，消除workload差异对结果的影响
    logger.info("========== 预生成统一的workload序列（确保所有算法使用相同的workload） ==========")
    workload_sequences = []
    
    for repeat_id in range(repeat_num):
        # 为每个repeat生成workload序列
        current_seed = None
        if random_seed is not None:
            current_seed = random_seed + repeat_id
            # 设置随机种子，确保可复现
            np.random.seed(current_seed)
            random.seed(current_seed)
        
        if workload_mode == "fixed_sum":
            workload_seq = _generate_workload_fixed_sum(
                total_gpu_sum, job_sizes, random_seed=current_seed
            )
        elif workload_mode == "random":
            workload_seq = _generate_workload_random(
                num_jobs, job_sizes, random_seed=current_seed
            )
        else:
            raise ValueError(f"未知的工作负载模式: {workload_mode}")
        
        workload_sequences.append(workload_seq)
        logger.info(
            f"Repeat {repeat_id + 1}/{repeat_num} workload序列: {workload_seq} "
            f"(总计: {sum(workload_seq)} GPUs)"
        )
    
    logger.info("========== workload序列预生成完成，所有算法将使用相同的workload ==========")
    
    # 加载模型（如果需要）
    model = _load_predictor(model_path, device, model_cfg)
    
    # 准备拓扑相关参数（某些算法需要）
    node_configs = convert_cluster_type_to_node_configs(cluster_type, total_gpu)
    topo_matrix, _ = build_composite_topo_matrix(node_configs)
    gpu_to_node_map = create_gpu_to_node_map(node_configs)
    
    all_results = []
    
    repeat_master_bar = tqdm(total=repeat_num, desc="Repeat 进度", position=0, leave=True)

    try:
        for algo_idx, algo_cfg in enumerate(algorithm_configs):
            algo_name = algo_cfg["name"]
            algo_func = algo_cfg["algo"]
            search_if_real_data = algo_cfg.get("search_if_real_data", False)
            
            logger.info(f"========== 运行算法: {algo_name} (搜索模式: {'真实数据' if search_if_real_data else '模型预测'}) ==========")
            
            algo_bar = tqdm(
                total=repeat_num,
                desc=f"{algo_name} repeat 进度",
                position=1,
                leave=False,
            )

            try:
                def _progress_callback(_):
                    """tqdm 进度更新回调。"""
                    if algo_bar is not None:
                        algo_bar.update(1)
                    if repeat_master_bar is not None and algo_idx == 0:
                        repeat_master_bar.update(1)

                def _run_with_timing(run_callable):
                    """运行仿真并记录算法级别耗时。"""
                    start_time = time.perf_counter()
                    df_result = run_callable()
                    elapsed = time.perf_counter() - start_time
                    df_result["algorithm_time"] = round(elapsed, 2)
                    logger.info(f"算法 {algo_name} 仿真总耗时: {elapsed:.2f}s")
                    return df_result
                
                # --------- 新增：离线 MINLP/BruteForce GroundTruth special-case ---------
                if isinstance(algo_func, str) and algo_func.upper() == "MINLP":
                    df = _run_with_timing(
                        lambda: run_multi_tenant_simulation_offline_minlp(
                            total_gpu=total_gpu,
                            gpu_bw_dict_list=gpu_bw_dict_list,
                            switch_config=switch_config,
                            model_path=model_path,
                            model_cfg=model_cfg,
                            data_path=data_path,
                            artifact_dir=artifact_dir,
                            device=device,
                            contention_mode=contention_mode,
                            workload_mode=workload_mode,
                            total_gpu_sum=total_gpu_sum,
                            num_jobs=num_jobs,
                            job_sizes=job_sizes,
                            repeat_num=repeat_num,
                            search_if_real_data=search_if_real_data,  # 这里默认 True
                            cluster_type=cluster_type,
                            random_seed=random_seed,
                            workload_sequences=workload_sequences,  # 关键：与其它算法共享完全相同的 workload
                            use_brute_force=False,  # 使用B&B算法
                            progress_callback=_progress_callback,
                        )
                    )

                    df["algorithm_name"] = algo_name
                    df["search_mode"] = "real_data" if search_if_real_data else "model_prediction"
                    all_results.append(df)
                    continue
                elif isinstance(algo_func, str) and algo_func.upper() == "BRUTE_FORCE":
                    df = _run_with_timing(
                        lambda: run_multi_tenant_simulation_offline_minlp(
                            total_gpu=total_gpu,
                            gpu_bw_dict_list=gpu_bw_dict_list,
                            switch_config=switch_config,
                            model_path=model_path,
                            model_cfg=model_cfg,
                            data_path=data_path,
                            artifact_dir=artifact_dir,
                            device=device,
                            contention_mode=contention_mode,
                            workload_mode=workload_mode,
                            total_gpu_sum=total_gpu_sum,
                            num_jobs=num_jobs,
                            job_sizes=job_sizes,
                            repeat_num=repeat_num,
                            search_if_real_data=search_if_real_data,  # 这里默认 True
                            cluster_type=cluster_type,
                            random_seed=random_seed,
                            workload_sequences=workload_sequences,  # 关键：与其它算法共享完全相同的 workload
                            use_brute_force=True,  # 使用暴力搜索
                            progress_callback=_progress_callback,
                        )
                    )

                    df["algorithm_name"] = algo_name
                    df["search_mode"] = "real_data" if search_if_real_data else "model_prediction"
                    all_results.append(df)
                    continue
                # ----------------------------------------------------------        
                # 创建算法适配器
                import inspect
                sig = inspect.signature(algo_func)
                param_names = list(sig.parameters.keys())
                # 仅当底层算法签名中包含 global_mode 时，才从配置中读取并透传
                extra_kwargs = {}
                if "global_mode" in param_names:
                    extra_kwargs["global_mode"] = algo_cfg.get("global_mode", True)
                
                if "topo_matrix" in param_names or "gpu_to_node_map" in param_names:
                    # 需要拓扑参数的算法（如 slurm_best_fit_algo）
                    from algorithms.slurm import slurm_best_fit_algo
                    search_algo = create_search_algo_adapter(
                        algo_func=algo_func,
                        algo_name=algo_name,
                        total_gpu=total_gpu,
                        topo_matrix=topo_matrix,
                        gpu_to_node_map=gpu_to_node_map,
                        **extra_kwargs,
                    )
                elif "model" in param_names:
                    # 需要模型参数的算法
                    search_algo = create_search_algo_adapter(
                        algo_func=algo_func,
                        algo_name=algo_name,
                        total_gpu=total_gpu,
                        model=model,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        data_path=data_path,
                        device=device,
                        artifact_dir=artifact_dir,
                        if_real_data=search_if_real_data,
                        **extra_kwargs,
                    )
                else:
                    # 简单算法（如 default_algo, random_algo）
                    search_algo = create_search_algo_adapter(
                        algo_func=algo_func,
                        algo_name=algo_name,
                        total_gpu=total_gpu,
                        **extra_kwargs,
                    )
                
                # 运行多租户仿真（使用预生成的统一workload序列）
                df = _run_with_timing(
                    lambda: run_multi_tenant_simulation(
                        total_gpu=total_gpu,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        model_path=model_path,
                        model_cfg=model_cfg,
                        data_path=data_path,
                        artifact_dir=artifact_dir,
                        device=device,
                        search_algo=search_algo,
                        contention_mode=contention_mode,
                        workload_mode=workload_mode,
                        total_gpu_sum=total_gpu_sum,
                        num_jobs=num_jobs,
                        job_sizes=job_sizes,
                        repeat_num=repeat_num,
                        search_if_real_data=search_if_real_data,
                        cluster_type=cluster_type,
                        random_seed=random_seed,  # 传递random_seed（虽然会被workload_sequences覆盖，但保持接口一致性）
                        workload_sequences=workload_sequences,  # 关键：传递预生成的统一workload序列
                        progress_callback=_progress_callback,
                    )
                )
                
                # 添加算法标识列
                df["algorithm_name"] = algo_name
                df["search_mode"] = "real_data" if search_if_real_data else "model_prediction"
                
                all_results.append(df)
            finally:
                algo_bar.close()
    finally:
        repeat_master_bar.close()
    
    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 重新排列列的顺序，将 algorithm_name 和 search_mode 放在前面
    cols = ["algorithm_name", "search_mode", "algorithm_time"]
    if "repeat_id" in combined_df.columns:
        cols.insert(0, "repeat_id")
    cols.extend([col for col in combined_df.columns if col not in cols])
    combined_df = combined_df[cols]
    
    return combined_df


def build_multi_tenant_filename(
    random_seed: int,
    num_train_samples: int,
    contention_mode: str,
    repeat_num: int,
) -> str:
    """根据配置构造多租户仿真结果文件名。
    
    命名规则：
        MTS_{random_seed}RS_{num_train_samples}TD_{contention_mode}CM_{repeat_num}RN
    
    其中：
        - random_seed: 顶层 random_seed（config.random_seed）
        - num_train_samples: 训练样本数（training.num_train_samples）
        - contention_mode: 争用模式（evaluation.multi_tenant.contention_mode）
        - repeat_num: 多租户仿真重复次数（evaluation.multi_tenant.repeat_num）
    """
    return f"MTS_{random_seed}RS_{num_train_samples}TD_{contention_mode}CM_{repeat_num}RN"
