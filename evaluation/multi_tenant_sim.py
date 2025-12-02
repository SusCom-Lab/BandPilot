"""多租户带宽预测与Placement仿真模块。

该模块实现了多租户场景下的GPU调度仿真，能够：
1. 模拟多个任务依次到达并分配GPU资源
2. 检测和计算资源争用对带宽的影响
3. 记录每个任务的分配结果和集群状态变化

支持多种搜索算法的对比，通过算法适配器统一不同算法的接口。
"""
from __future__ import annotations

import logging
import random
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from algorithms.baseline import default_algo, random_algo
from algorithms.search import improved_searching_algo, tree_search_only
from algorithms.eha import eha_search
from core.bandwidth import SwitchBandwidthConfig
from core.cluster_state import ClusterStateManager, create_bandwidth_predictor
from core.topology import (
    build_composite_topo_matrix,
    convert_cluster_type_to_node_configs,
    create_gpu_to_node_map,
)
from models.bandwidth_predictor import BandwidthPredictor

# 配置日志
logger = logging.getLogger(__name__)


# ==================== 算法适配器 ====================
# 为了统一不同算法的接口，创建适配器函数
# 统一的算法函数签名：(num_dimensions, avail_gpu, gpu_need, ..., cluster_manager=None) -> Optional[np.ndarray]


def create_search_algo_adapter(
    algo_func: Callable,
    algo_name: str,
    total_gpu: int,
    model=None,
    gpu_bw_dict_list=None,
    switch_config=None,
    data_path: str = "",
    device=None,
    artifact_dir: Optional[Path] = None,
    if_real_data: bool = False,
    **extra_kwargs,
) -> Callable:
    """创建搜索算法的适配器，统一不同算法的接口。
    
    该函数接受一个原始算法函数，返回一个符合统一接口的适配器函数。
    适配器函数签名：`(num_dimensions, avail_gpu, gpu_need, cluster_manager=None) -> Optional[np.ndarray]`
    
    Args:
        algo_func: 原始算法函数
        algo_name: 算法名称（用于日志）
        total_gpu: 总GPU数量
        model: PyTorch模型（某些算法需要）
        gpu_bw_dict_list: GPU带宽字典列表（某些算法需要）
        switch_config: 交换机配置（某些算法需要）
        data_path: 数据路径（某些算法需要）
        device: PyTorch设备（某些算法需要）
        artifact_dir: 模型目录（某些算法需要）
        if_real_data: 是否使用真实数据（某些算法需要）
        **extra_kwargs: 其他算法特定的参数（如 topo_matrix, gpu_to_node_map 等）
    
    Returns:
        适配器函数，符合统一接口：(num_dimensions, avail_gpu, gpu_need, cluster_manager=None) -> Optional[np.ndarray]
    """
    # 检查算法函数的签名，判断需要哪些参数
    import inspect
    
    sig = inspect.signature(algo_func)
    param_names = list(sig.parameters.keys())
    
    # 判断算法类型
    needs_model = "model" in param_names
    needs_topo = "topo_matrix" in param_names or "gpu_to_node_map" in param_names
    
    def adapter(
        num_dimensions: int,
        avail_gpu: List[int],
        gpu_need: int,
        cluster_manager: Optional[ClusterStateManager] = None,
    ) -> Optional[np.ndarray]:
        """统一的算法接口适配器。"""
        try:
            if needs_topo:
                # 需要拓扑参数的算法（如 slurm_best_fit_algo）
                # 这些参数应该在创建适配器时通过 extra_kwargs 传入
                return algo_func(
                    total_gpu,
                    avail_gpu,
                    gpu_need,
                    **extra_kwargs,
                )
            elif needs_model:
                # 需要模型参数的算法（如 improved_searching_algo, tree_search_only, eha_search）
                return algo_func(
                    num_dimensions=num_dimensions,
                    avail_gpu=avail_gpu,
                    model=model,
                    gpu_need=gpu_need,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    data_path=data_path,
                    device=device,
                    artifact_dir=artifact_dir,
                    if_real_data=if_real_data,
                    cluster_manager=cluster_manager,
                    **extra_kwargs,
                )
            else:
                # 简单算法（如 default_algo, random_algo）
                return algo_func(
                    total_gpu=total_gpu,
                    avail_gpu=avail_gpu,
                    gpu_need=gpu_need,
                )
        except Exception as e:
            logger.error(f"算法 {algo_name} 执行失败: {e}")
            return None
    
    return adapter


def _load_predictor(model_path: Path, device: torch.device, model_cfg: dict) -> BandwidthPredictor:
    """加载 BandwidthPredictor 模型（与 compare.py 保持一致）。"""
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


def _generate_workload_fixed_sum(
    total_gpu_sum: int, job_sizes: List[int], random_seed: Optional[int] = None
) -> List[int]:
    """生成总和为指定值的随机任务序列（fixed_sum 模式）。
    
    Args:
        total_gpu_sum: 任务GPU需求的总和
        job_sizes: 允许的任务大小列表
        random_seed: 随机种子（可选）
    
    Returns:
        任务大小列表，总和等于 total_gpu_sum
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    workload_requests = []
    remaining = total_gpu_sum
    
    while remaining > 0:
        # 如果剩余数量小于等于最小任务大小，直接分配
        if remaining <= min(job_sizes):
            size = remaining
        else:
            # 从允许的大小中选择不超过剩余数量的值
            valid_sizes = [s for s in job_sizes if s <= remaining]
            if not valid_sizes:
                # 如果没有合适的，分配剩余的全部
                size = remaining
            else:
                size = random.choice(valid_sizes)
        workload_requests.append(size)
        remaining -= size
    
    # 打乱顺序模拟真实到达
    random.shuffle(workload_requests)
    return workload_requests


def _generate_workload_random(
    num_jobs: int, job_sizes: List[int], random_seed: Optional[int] = None
) -> List[int]:
    """随机生成指定数量的任务（random 模式）。
    
    Args:
        num_jobs: 任务数量
        job_sizes: 允许的任务大小列表
        random_seed: 随机种子（可选）
    
    Returns:
        任务大小列表
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    return [random.choice(job_sizes) for _ in range(num_jobs)]


def _format_combo_as_node_counts(combo: np.ndarray, total_gpu: int, node_size: int = 8) -> str:
    """将 GPU 组合格式化为每个节点的 GPU 数量列表。
    
    例如，如果 GPU [13,14,22,23] 被选中：
    - 节点 0 (GPU 0-7): 0 个 GPU
    - 节点 1 (GPU 8-15): GPU 13,14 → 2 个 GPU
    - 节点 2 (GPU 16-23): GPU 22,23 → 2 个 GPU
    - 节点 3 (GPU 24-31): 0 个 GPU
    结果: "[0,2,2,0]"
    
    Args:
        combo: GPU组合（0/1向量）
        total_gpu: 总GPU数量
        node_size: 每个节点的GPU数量（默认8）
    
    Returns:
        格式化的字符串，如 "[0,2,2,0]"
    """
    num_nodes = total_gpu // node_size
    node_counts = [0] * num_nodes
    
    # 获取选中的 GPU 索引
    selected_gpus = np.where(combo == 1)[0]
    
    # 统计每个节点上的 GPU 数量
    for gpu_idx in selected_gpus:
        node_id = gpu_idx // node_size
        if 0 <= node_id < num_nodes:
            node_counts[node_id] += 1
    
    # 格式化为字符串
    return "[" + ",".join(map(str, node_counts)) + "]"


def _ensure_combo_feasible(
    combo: np.ndarray,
    total_gpu: int,
    node_size: int,
    avail_gpu: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """在提交 ClusterStateManager 前做防御式检查，避免非法资源组合。

    - combo 需与 total_gpu 同维度，并且元素只允许 {0,1}
    - 任意节点的使用量不能超过 node_size
    - 如果提供 avail_gpu，则组合中的 GPU 必须全部来自可用列表
    """
    combo_arr = np.asarray(combo)
    if combo_arr.ndim != 1 or combo_arr.size != total_gpu:
        raise ValueError(
            f"非法 combo 维度，期望 {total_gpu}，实际 {combo_arr.size}"
        )

    if not np.all((combo_arr == 0) | (combo_arr == 1)):
        raise ValueError("combo 中存在非 {0,1} 元素，无法映射到 GPU")

    combo_int = combo_arr.astype(int, copy=False)

    reshaped = combo_int.reshape(-1, node_size)
    node_usage = reshaped.sum(axis=1)
    if np.any(node_usage > node_size):
        raise ValueError(
            f"combo 触发单节点超额：节点需求 {node_usage.tolist()}，单节点容量 {node_size}"
        )

    if avail_gpu is not None:
        avail_set = set(avail_gpu)
        selected = np.where(combo_int == 1)[0]
        unavailable = [idx for idx in selected if idx not in avail_set]
        if unavailable:
            raise ValueError(
                f"combo 使用了不可用 GPU（示例: {unavailable[:5]}）"
            )

    return combo_int


def run_multi_tenant_simulation(
    total_gpu: int,
    gpu_bw_dict_list: List,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    data_path: str,
    artifact_dir: Path,
    device: torch.device,
    search_algo: Callable,
    contention_mode: str = "intensive",
    workload_mode: str = "fixed_sum",
    total_gpu_sum: int = 32,
    num_jobs: int = 10,
    job_sizes: List[int] = None,
    repeat_num: int = 1,
    search_if_real_data: bool = False,
    random_seed: Optional[int] = None,
    cluster_type: Optional[str] = None,
    workload_sequences: Optional[List[List[int]]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> pd.DataFrame:
    """运行多租户仿真，模拟多个任务依次到达并分配GPU资源。
    
    该函数实现了完整的多租户仿真流程，分为两个阶段：
    
    **阶段一：搜索阶段（可配置评估模式）**
    1. 使用 search_if_real_data 指定的评估模式创建 ClusterStateManager
    2. 根据配置生成工作负载序列
    3. 对每个任务使用 search_algo 搜索最优GPU组合（考虑多租户争用）
    4. 分配资源并记录GPU组合
    
    **阶段二：评估阶段（使用真实数据）**
    5. 使用真实数据（if_real_data=True）创建新的 ClusterStateManager
    6. 按照相同的顺序重新分配所有任务，使用真实数据计算带宽
    7. 记录真实数据下的带宽值、争用情况等
    
    Args:
        total_gpu: 集群总GPU数量
        gpu_bw_dict_list: GPU带宽字典列表
        switch_config: 交换机配置
        model_path: 模型权重文件路径
        model_cfg: 模型配置字典
        data_path: 带宽数据文件路径
        artifact_dir: 模型和scaler文件所在目录
        device: PyTorch设备（CPU或CUDA）
        search_algo: 搜索算法函数，签名应为 (num_dimensions, avail_gpu, gpu_need, cluster_manager=None) -> Optional[np.ndarray]
        contention_mode: 争用模式。
            - "intensive": 假设各任务满负载，跨节点任务按照瓶颈容量瓜分带宽
            - "common": 模拟实时中等带宽占用，每个任务随机取 25%~50% 峰值作为需求再做争用
            - "idle": 认为所有任务错峰运行，不发生争用
        workload_mode: 工作负载生成模式（'fixed_sum' 或 'random'）
        total_gpu_sum: fixed_sum 模式下的总GPU数
        num_jobs: random 模式下的任务数量
        job_sizes: 允许的任务大小列表，默认为 [1, 2, 4, 8]
        repeat_num: 重复仿真次数
        search_if_real_data: 搜索阶段使用的评估模式（True=真实数据，False=模型预测）
        random_seed: 随机种子（可选，如果提供了 workload_sequences 则忽略）
        cluster_type: 集群类型（可选，某些算法需要）
        workload_sequences: 预生成的工作负载序列列表（可选），如果提供则使用，否则内部生成。
            应该是一个长度为 repeat_num 的列表，每个元素是一个任务大小列表。
        progress_callback: 可选回调函数，在每次 repeat 完成后调用，用于更新外部进度条。
    
    Returns:
        pandas DataFrame，包含以下列：
        - job_id: 任务ID
        - gpu_need: 需要的GPU数量
        - combo: GPU组合（转为字符串，格式如 "0,1,2,3"）
        - predicted_standalone_bw: 搜索阶段的独占带宽
        - predicted_final_bw: 搜索阶段的最终带宽（考虑争用）
        - real_standalone_bw: 评估阶段的独占带宽（真实数据）
        - real_final_bw: 评估阶段的最终带宽（真实数据，考虑争用）
        - real_contention_ratio: 真实数据下的争用比例 (real_final_bw / real_standalone_bw)
        - real_cluster_throughput: 真实数据下的集群总吞吐量
        - num_active_jobs: 当前活跃任务数
        - repeat_id: 重复仿真的ID（当 repeat_num > 1 时）
    """
    if job_sizes is None:
        job_sizes = [1, 2, 4, 8]
    
    # 加载模型（如果搜索阶段使用模型预测，或者评估阶段需要）
    model = None
    if not search_if_real_data:
        model = _load_predictor(model_path, device, model_cfg)
        logger.info(f"模型加载成功: {model_path}")
    
    # 存储所有重复仿真的结果
    all_results = []
    
    for repeat_id in range(repeat_num):
        logger.info(f"========== 开始第 {repeat_id + 1}/{repeat_num} 次仿真 ==========")
        
        # ==================== 阶段一：搜索阶段（可配置评估模式） ====================
        search_mode_str = "真实数据" if search_if_real_data else "模型预测"
        logger.info(f"--- 阶段一：搜索阶段（使用{search_mode_str}） ---")
        
        # 重要：无论是否使用预提供的workload，都需要设置随机种子，确保算法内部的随机操作一致
        # 这对于公平对比不同算法至关重要
        current_seed = None
        if random_seed is not None:
            current_seed = random_seed + repeat_id
            # 设置 numpy 和 python random 的随机种子，确保算法内部的随机操作也使用正确的种子
            np.random.seed(current_seed)
            random.seed(current_seed)
        
        # 为本次 repeat 生成一个占用比率种子，确保搜索与评估阶段共享相同的随机序列
        occupancy_seed = current_seed if current_seed is not None else int(np.random.randint(0, 2**31 - 1))
        
        # 创建用于搜索的 ClusterStateManager（根据 search_if_real_data 配置）
        search_predictor = create_bandwidth_predictor(
            if_real_data=search_if_real_data,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            model=model if not search_if_real_data else None,
            device=device if not search_if_real_data else None,
            artifact_dir=artifact_dir if not search_if_real_data else None,
        )
        
        search_manager = ClusterStateManager(
            total_gpu=total_gpu,
            bandwidth_predictor=search_predictor,
            contention_mode=contention_mode,
            occupancy_seed=occupancy_seed,
        )
        
        # 生成或使用预提供的工作负载序列
        if workload_sequences is not None:
            # 使用预提供的工作负载序列（确保所有算法使用相同的workload）
            if repeat_id >= len(workload_sequences):
                raise ValueError(
                    f"workload_sequences 长度 ({len(workload_sequences)}) 小于 repeat_num ({repeat_num})"
                )
            workload_requests = workload_sequences[repeat_id]
            logger.info(
                f"使用预提供的工作负载序列 (repeat {repeat_id}): {workload_requests} "
                f"(总计: {sum(workload_requests)} GPUs)"
            )
        else:
            # 内部生成工作负载序列（random_seed 已在上面设置）
            if workload_mode == "fixed_sum":
                workload_requests = _generate_workload_fixed_sum(
                    total_gpu_sum, job_sizes, random_seed=current_seed
                )
            elif workload_mode == "random":
                workload_requests = _generate_workload_random(
                    num_jobs, job_sizes, random_seed=current_seed
                )
            else:
                raise ValueError(f"未知的工作负载模式: {workload_mode}")
            
            logger.info(f"生成的工作负载序列: {workload_requests} (总计: {sum(workload_requests)} GPUs)")
        
        # 存储搜索阶段的结果（GPU组合）
        search_results = []  # List of (job_idx, gpu_need, combo, predicted_standalone_bw, predicted_final_bw)
        
        # 搜索阶段：使用模型预测搜索最优GPU组合
        for job_idx, gpu_need in enumerate(workload_requests):
            logger.info(f"[搜索阶段] 处理任务 {job_idx}: 需要 {gpu_need} 个GPU")
            
            # 1. 获取当前可用资源
            avail_gpu = search_manager.get_available_gpus()
            
            if len(avail_gpu) < gpu_need:
                logger.error(f"资源不足！任务 {job_idx} 需要 {gpu_need}, 可用 {len(avail_gpu)}")
                break
            
            # 2. 搜索最优 Placement（使用传入的搜索算法，感知多租户争用）
            search_manager.set_job_context(job_idx)
            try:
                best_combo = search_algo(
                    num_dimensions=total_gpu,
                    avail_gpu=avail_gpu,
                    gpu_need=gpu_need,
                    cluster_manager=search_manager,
                )
            finally:
                search_manager.clear_job_context()
            
            if best_combo is None:
                logger.error(f"未能找到任务 {job_idx} 的合适组合")
                continue
            
            try:
                best_combo = _ensure_combo_feasible(
                    best_combo,
                    total_gpu=total_gpu,
                    node_size=search_manager.node_size,
                    avail_gpu=avail_gpu,
                )
            except ValueError as exc:
                logger.error(
                    f"[搜索阶段] 任务 {job_idx} 获取的组合非法：{exc}，已跳过该任务"
                )
                continue

            # 3. 分配并获取预测带宽
            try:
                predicted_final_bw = search_manager.allocate_job(job_id=job_idx, combo=best_combo)
            except ValueError as exc:
                logger.error(
                    f"[搜索阶段] 任务 {job_idx} 分配失败（非法组合）：{exc}"
                )
                continue
            
            # 获取预测的独占带宽
            predicted_standalone_bw = None
            for job in search_manager.active_jobs:
                if job["job_id"] == job_idx:
                    predicted_standalone_bw = job["standalone_bw"]
                    break
            
            if predicted_standalone_bw is None:
                logger.warning(f"未能找到任务 {job_idx} 的预测独占带宽")
                predicted_standalone_bw = predicted_final_bw
            
            combo_str = _format_combo_as_node_counts(best_combo, total_gpu, node_size=8)
            logger.info(
                f"[搜索阶段] 任务 {job_idx} 分配成功. "
                f"GPU组合（节点分布）: {combo_str}, "
                f"预测独占带宽: {predicted_standalone_bw:.2f}, "
                f"预测最终带宽: {predicted_final_bw:.2f}"
            )
            
            # 保存搜索阶段的结果
            search_results.append({
                "job_idx": job_idx,
                "gpu_need": gpu_need,
                "combo": best_combo.copy(),
                "predicted_standalone_bw": predicted_standalone_bw,
                "predicted_final_bw": predicted_final_bw,
            })
        
        logger.info(f"[搜索阶段] 完成，共分配 {len(search_results)} 个任务")
        
        # ==================== 阶段二：评估阶段（使用真实数据） ====================
        logger.info("--- 阶段二：评估阶段（使用真实数据重新计算） ---")
        
        # 创建用于评估的 ClusterStateManager（使用真实数据）
        eval_predictor = create_bandwidth_predictor(
            if_real_data=True,  # 评估阶段始终使用真实数据
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            model=None,  # 真实数据模式不需要模型
            device=None,
            artifact_dir=None,
        )
        
        eval_manager = ClusterStateManager(
            total_gpu=total_gpu,
            bandwidth_predictor=eval_predictor,
            contention_mode=contention_mode,
            occupancy_seed=occupancy_seed,
        )
        
        # 按照相同的顺序重新分配所有任务，使用真实数据计算带宽
        for search_result in search_results:
            job_idx = search_result["job_idx"]
            gpu_need = search_result["gpu_need"]
            combo = search_result["combo"]

            eval_avail_gpu = eval_manager.get_available_gpus()
            try:
                combo = _ensure_combo_feasible(
                    combo,
                    total_gpu=total_gpu,
                    node_size=eval_manager.node_size,
                    avail_gpu=eval_avail_gpu,
                )
            except ValueError as exc:
                logger.error(
                    f"[评估阶段] 任务 {job_idx} 的组合非法：{exc}，该任务结果将被跳过"
                )
                continue
            
            combo_str = _format_combo_as_node_counts(combo, total_gpu, node_size=8)
            logger.info(f"[评估阶段] 重新分配任务 {job_idx}: GPU组合（节点分布）{combo_str}")
            
            # 使用真实数据重新分配并计算带宽
            try:
                real_final_bw = eval_manager.allocate_job(job_id=job_idx, combo=combo)
            except ValueError as exc:
                logger.error(
                    f"[评估阶段] 任务 {job_idx} 分配失败（非法组合）：{exc}"
                )
                continue
            
            # 获取真实数据的独占带宽
            real_standalone_bw = None
            for job in eval_manager.active_jobs:
                if job["job_id"] == job_idx:
                    real_standalone_bw = job["standalone_bw"]
                    break
            
            if real_standalone_bw is None:
                logger.warning(f"未能找到任务 {job_idx} 的真实独占带宽")
                real_standalone_bw = real_final_bw
            
            # 计算真实数据下的争用比例
            real_contention_ratio = real_final_bw / real_standalone_bw if real_standalone_bw > 0 else 0.0
            
            # 计算真实数据下的集群总吞吐量
            # 注意：这是所有活跃任务的 current_bw 之和，即当前时刻的全局总带宽
            # 对于最后一个 job_id，这个值就是整个 workload 完成分配后的全局总带宽大小
            real_cluster_throughput = sum(job["current_bw"] for job in eval_manager.active_jobs)
            num_active_jobs = len(eval_manager.active_jobs)
            
            logger.info(
                f"[评估阶段] 任务 {job_idx} 重新计算完成. "
                f"真实独占带宽: {real_standalone_bw:.2f}, "
                f"真实最终带宽: {real_final_bw:.2f}, "
                f"争用比例: {real_contention_ratio:.2%}"
            )
            
            # 记录完整结果（包含搜索和评估两个阶段的数据）
            # combo 格式化为每个节点的 GPU 数量列表，如 "[0,2,2,0]"
            combo_str = _format_combo_as_node_counts(combo, total_gpu, node_size=8)
            
            result = {
                "repeat_id": repeat_id if repeat_num > 1 else 0,
                "job_id": job_idx,
                "gpu_need": gpu_need,
                "combo": combo_str,
                "predicted_standalone_bw": search_result["predicted_standalone_bw"],
                "predicted_final_bw": search_result["predicted_final_bw"],
                "real_standalone_bw": real_standalone_bw,
                "real_final_bw": real_final_bw,
                "real_contention_ratio": real_contention_ratio,
                "real_cluster_throughput": real_cluster_throughput,
                "num_active_jobs": num_active_jobs,
            }
            all_results.append(result)
        
        # 打印本次仿真的最终状态
        logger.info(f"========== 第 {repeat_id + 1} 次仿真结束 ==========")
        predicted_total_throughput = sum(job["current_bw"] for job in search_manager.active_jobs)
        real_total_throughput = sum(job["current_bw"] for job in eval_manager.active_jobs)
        logger.info(f"搜索阶段最终集群总吞吐量（预测）: {predicted_total_throughput:.2f}")
        logger.info(f"评估阶段最终集群总吞吐量（真实）: {real_total_throughput:.2f}")
        
        # 检查是否有任务经历了争用（使用真实数据）
        for job in eval_manager.active_jobs:
            if job["current_bw"] < job["standalone_bw"]:
                logger.info(
                    f"任务 {job['job_id']} 在真实数据下经历了争用: "
                    f"{job['standalone_bw']:.2f} -> {job['current_bw']:.2f}"
                )
        
        if progress_callback is not None:
            progress_callback(repeat_id)
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_results)
    
    # 如果只有一次仿真，移除 repeat_id 列（保持向后兼容）
    if repeat_num == 1 and "repeat_id" in df.columns:
        df = df.drop(columns=["repeat_id"])
    elif repeat_num > 1:
        # 确保 repeat_id 列在第一位（便于查看）
        cols = ["repeat_id"] + [col for col in df.columns if col != "repeat_id"]
        df = df[cols]
    
    return df



# ==================== 离线 MINLP / Branch-and-Bound GroundTruth 求解器 ====================
# 直接对齐 ClusterStateManager 的争用/带宽规则：
# - 叶子解评估时，使用 ClusterStateManager.allocate_job 顺序 commit
# - objective = 最终所有 job 的 current_bw 之和
#
# 关键假设（与你现有模型一致）：
# - 带宽只取决于“每个节点上放了多少 GPU”，节点内 GPU 同质
#   因此离线搜索用“node-count pattern”而非任意 GPU 子集，极大缩小搜索空间
#
# 如果未来你引入节点内异质 GPU，需要把 pattern -> 实际 GPU 选择再做一层细化搜索。

from typing import Sequence, Tuple, Dict, NamedTuple, List, Callable, Optional
from functools import lru_cache


class _NodePattern(NamedTuple):
    """一个 job 的节点计数模式（counts）及其独占带宽（standalone_bw）。"""
    counts: Tuple[int, ...]
    standalone_bw: float
    involved_nodes: Tuple[int, ...]
    is_cross: bool


def _enumerate_node_patterns(gpu_need: int, num_nodes: int, node_size: int) -> List[Tuple[int, ...]]:
    """枚举 gpu_need 在 num_nodes 个节点上的所有可行计数分配（每节点 <= node_size）。"""
    patterns: List[Tuple[int, ...]] = []

    def rec(idx: int, remaining: int, cur: List[int]):
        if idx == num_nodes:
            if remaining == 0:
                patterns.append(tuple(cur))
            return
        max_take = min(node_size, remaining)
        for k in range(max_take + 1):
            cur.append(k)
            rec(idx + 1, remaining - k, cur)
            cur.pop()

    rec(0, gpu_need, [])
    return patterns


def _canonical_combo_from_counts(counts: Tuple[int, ...], total_gpu: int, node_size: int) -> np.ndarray:
    """把 node-count 模式转成一个“代表性 combo”（每节点取前 counts[n] 张 GPU）。"""
    combo = np.zeros(total_gpu, dtype=int)
    for n, c in enumerate(counts):
        if c > 0:
            start = n * node_size
            combo[start:start + c] = 1
    return combo


def _counts_solution_to_combos(
    patterns: Sequence[_NodePattern],
    total_gpu: int,
    node_size: int
) -> List[np.ndarray]:
    """
    把一组 pattern（每 job 的 counts）具体化为不冲突的 GPU combo。
    由于同质假设，具体哪几张卡无关紧要，但要保证 disjoint 才能给 manager 用。
    """
    num_nodes = total_gpu // node_size
    free_lists = [list(range(n * node_size, (n + 1) * node_size)) for n in range(num_nodes)]
    combos: List[np.ndarray] = []

    for pat in patterns:
        combo = np.zeros(total_gpu, dtype=int)
        for n, c in enumerate(pat.counts):
            if c <= 0:
                continue
            if len(free_lists[n]) < c:
                raise ValueError(f"Node {n} 资源不足，无法具体化 counts={pat.counts}")
            chosen = free_lists[n][:c]
            free_lists[n] = free_lists[n][c:]
            combo[chosen] = 1
        combos.append(combo)

    return combos


def _evaluate_patterns_with_manager(
    patterns: Sequence[_NodePattern],
    total_gpu: int,
    bandwidth_predictor: Callable[[np.ndarray], float],
    node_size: int = 8,
    contention_mode: str = "intensive",
    occupancy_seed: Optional[int] = None,
) -> Tuple[float, List[Tuple[np.ndarray, float, float]]]:
    """
    对一组 patterns 做精确叶子评估：
    - 具体化为 disjoint combos
    - 用 ClusterStateManager.allocate_job 顺序 commit
    返回：
      total_throughput,
      per_job[(combo, standalone_bw, final_bw)]

    Args:
        contention_mode: 传递给 ClusterStateManager 的争用模式
        occupancy_seed: 控制 common 模式下占用比率的基准种子，确保多次评估一致
    """
    combos = _counts_solution_to_combos(patterns, total_gpu, node_size)

    manager = ClusterStateManager(
        total_gpu=total_gpu,
        bandwidth_predictor=bandwidth_predictor,
        contention_mode=contention_mode,
        occupancy_seed=occupancy_seed,
    )

    per_job: List[Tuple[np.ndarray, float, float]] = []
    total_throughput = 0.0

    for job_id, combo in enumerate(combos):
        final_bw = manager.allocate_job(job_id=job_id, combo=combo)
        standalone_bw = manager.active_jobs[-1]["standalone_bw"]
        per_job.append((combo, standalone_bw, final_bw))
        total_throughput += final_bw

    return total_throughput, per_job


def minlp_offline_optimal_solver(
    workload_requests: Sequence[int],
    total_gpu: int,
    bandwidth_predictor: Callable[[np.ndarray], float],
    node_size: int = 8,
    verbose: bool = False,
    contention_mode: str = "intensive",
    occupancy_seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[float], List[float], float]:
    """
    离线全局最优（GroundTruth）求解：
    输入：完整 workload_requests（每个 job 的 gpu_need），以及带宽黑盒 bandwidth_predictor
    输出：
      - best_combos: 每个 job 的 combo（与现有系统对齐）
      - predicted_standalone_bws
      - predicted_final_bws（按你的争用规则）
      - best_total_throughput

    Args:
        contention_mode: 传递给 ClusterStateManager 的争用模式
    """
    num_nodes = total_gpu // node_size

    # 1) 预枚举每种 gpu_need 的可行模式，并计算独占带宽常数
    patterns_by_need: Dict[int, List[_NodePattern]] = {}
    for g in set(workload_requests):
        raw_patterns = _enumerate_node_patterns(g, num_nodes, node_size)
        pats: List[_NodePattern] = []
        for counts in raw_patterns:
            combo = _canonical_combo_from_counts(counts, total_gpu, node_size)
            s_bw = float(bandwidth_predictor(combo))
            involved = tuple(i for i, c in enumerate(counts) if c > 0)
            pats.append(
                _NodePattern(
                    counts=counts,
                    standalone_bw=s_bw,
                    involved_nodes=involved,
                    is_cross=(len(involved) > 1),
                )
            )
        # 搜索时按独占带宽从大到小排序 → 提前找到好解增强剪枝
        pats.sort(key=lambda p: p.standalone_bw, reverse=True)
        patterns_by_need[g] = pats

    # 2) 可行模式缓存（按剩余容量过滤）
    @lru_cache(maxsize=None)
    def feasible_patterns_for_need(g: int, caps: Tuple[int, ...]) -> Tuple[_NodePattern, ...]:
        feas = []
        for pat in patterns_by_need[g]:
            ok = True
            for n in range(num_nodes):
                if pat.counts[n] > caps[n]:
                    ok = False
                    break
            if ok:
                feas.append(pat)
        return tuple(feas)

    # 3) 改进的上界计算：使用更保守的策略避免过早剪枝
    def upper_bound(idx: int, caps: Tuple[int, ...], chosen: List[_NodePattern]) -> float:
        """
        计算上界（保守策略，避免过早剪枝最优解）：
        - 对于已选择的任务，使用 _evaluate_patterns_with_manager 计算实际总带宽（考虑争用）
        - 对于未选择的任务，使用保守估计：
          * 如果剩余任务数量少，尝试评估所有剩余任务的最优组合
          * 否则，使用最大standalone_bw但应用争用折扣因子
        """
        # 计算已选择任务的实际总带宽
        if len(chosen) > 0:
            try:
                actual_total, _ = _evaluate_patterns_with_manager(
                    chosen,
                    total_gpu,
                    bandwidth_predictor,
                    node_size=node_size,
                    contention_mode=contention_mode,
                    occupancy_seed=occupancy_seed,
                )
                ub = actual_total
            except Exception as e:
                # 如果评估失败（例如资源不足），回退到 standalone_bw 之和
                logger.warning(f"上界计算时评估失败，回退到 standalone_bw: {e}")
                ub = sum(p.standalone_bw for p in chosen)
        else:
            ub = 0.0
        
        # 对于未选择的任务，使用更保守的上界估计
        remaining_requests = workload_requests[idx:]
        if len(remaining_requests) <= 3 and len(remaining_requests) > 0:
            # 如果剩余任务数量少（<=3），尝试评估最优组合（保守但准确）
            try:
                # 为每个剩余任务选择最大standalone_bw的pattern
                remaining_patterns = []
                temp_caps = list(caps)
                for g in remaining_requests:
                    feas = feasible_patterns_for_need(g, tuple(temp_caps))
                    if not feas:
                        return float("-inf")
                    best_pat = feas[0]  # 已按standalone_bw降序
                    remaining_patterns.append(best_pat)
                    # 更新临时容量
                    for n in range(num_nodes):
                        temp_caps[n] -= best_pat.counts[n]
                
                # 评估这些pattern组合的实际总带宽
                if remaining_patterns:
                    remaining_total, _ = _evaluate_patterns_with_manager(
                        remaining_patterns,
                        total_gpu,
                        bandwidth_predictor,
                        node_size=node_size,
                        contention_mode=contention_mode,
                        occupancy_seed=occupancy_seed,
                    )
                    ub += remaining_total
            except Exception:
                # 如果评估失败，回退到简单相加（不使用折扣，保持乐观）
                for g in remaining_requests:
                    feas = feasible_patterns_for_need(g, tuple(caps))
                    if not feas:
                        return float("-inf")
                    ub += feas[0].standalone_bw
        else:
            # 如果剩余任务数量多，使用保守的争用折扣因子
            # 假设未选择任务之间会有争用，使用折扣因子0.7-0.9（根据任务数量调整）
            discount_factor = max(0.7, 1.0 - 0.05 * len(remaining_requests))
            for k in range(idx, len(workload_requests)):
                g = workload_requests[k]
                feas = feasible_patterns_for_need(g, caps)
                if not feas:
                    return float("-inf")
                # 应用折扣因子，考虑争用影响
                ub += feas[0].standalone_bw * discount_factor
        
        return ub

    best_obj = float("-inf")
    best_patterns: Optional[List[_NodePattern]] = None
    nodes_evaluated = 0  # 统计评估的节点数
    nodes_pruned = 0  # 统计剪枝的节点数

    # 4) DFS + B&B（改进搜索顺序，优先探索有希望的路径）
    def dfs(idx: int, caps: Tuple[int, ...], chosen: List[_NodePattern]):
        nonlocal best_obj, best_patterns, nodes_evaluated, nodes_pruned

        if idx == len(workload_requests):
            # 叶子节点：评估完整解
            nodes_evaluated += 1
            obj, _ = _evaluate_patterns_with_manager(
                chosen,
                total_gpu,
                bandwidth_predictor,
                node_size=node_size,
                contention_mode=contention_mode,
                occupancy_seed=occupancy_seed,
            )
            if obj > best_obj:
                best_obj = obj
                best_patterns = list(chosen)
                if verbose:
                    logger.info(f"[MINLP] New best obj = {best_obj:.4f}, patterns = {[p.counts for p in best_patterns]}")
            return

        ub = upper_bound(idx, caps, chosen)
        if ub <= best_obj:
            nodes_pruned += 1
            return  # 剪枝

        g = workload_requests[idx]
        feas = feasible_patterns_for_need(g, caps)
        if not feas:
            return

        # 改进：按照standalone_bw降序搜索，优先探索有希望的路径
        # 这样可以更快找到好的解，从而增强剪枝效果
        # feas已经按standalone_bw降序排列（在patterns_by_need中已排序）
        for pat in feas:
            new_caps = list(caps)
            for n in range(num_nodes):
                new_caps[n] -= pat.counts[n]
            chosen.append(pat)
            dfs(idx + 1, tuple(new_caps), chosen)
            chosen.pop()

    init_caps = tuple([node_size] * num_nodes)
    dfs(0, init_caps, [])
    
    if verbose:
        logger.info(f"[MINLP] 搜索完成: 评估了 {nodes_evaluated} 个叶子节点, 剪枝了 {nodes_pruned} 个节点")

    if best_patterns is None:
        raise RuntimeError("MINLP GroundTruth 未找到可行解（workload 总和可能超过集群容量）。")

    # 5) 从 best_patterns 得到 combos 和带宽（按 manager 精确重算）
    best_total_throughput, per_job = _evaluate_patterns_with_manager(
        best_patterns,
        total_gpu,
        bandwidth_predictor,
        node_size=node_size,
        contention_mode=contention_mode,
        occupancy_seed=occupancy_seed,
    )
    best_combos = [x[0] for x in per_job]
    predicted_standalone_bws = [x[1] for x in per_job]
    predicted_final_bws = [x[2] for x in per_job]
    
    # 验证：确保找到的解的总吞吐量等于 best_obj（应该一致，但添加检查以确保正确性）
    if abs(best_total_throughput - best_obj) > 1e-6:
        logger.warning(
            f"[MINLP] 警告：best_total_throughput ({best_total_throughput:.4f}) 与 best_obj ({best_obj:.4f}) 不一致，"
            f"差异: {abs(best_total_throughput - best_obj):.6f}"
        )
    
    if verbose:
        logger.info(
            f"[MINLP] 找到最优解: 总吞吐量 = {best_total_throughput:.4f}, "
            f"评估了 {nodes_evaluated} 个叶子节点, 剪枝了 {nodes_pruned} 个节点"
        )

    return best_combos, predicted_standalone_bws, predicted_final_bws, best_total_throughput


def brute_force_optimal_solver(
    workload_requests: Sequence[int],
    total_gpu: int,
    bandwidth_predictor: Callable[[np.ndarray], float],
    node_size: int = 8,
    verbose: bool = False,
    max_combinations: int = 1000000,  # 限制组合数量，避免超时
    brute_force_concrete=True, 
    contention_mode: str = "intensive",
    occupancy_seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[float], List[float], float]:
    """
    暴力搜索全局最优解（GroundTruth）：
    枚举所有可能的pattern组合，找到全局最优解。
    
    注意：对于大规模问题，这个函数可能非常慢。主要用于小规模问题的验证。
    
    输入：完整 workload_requests（每个 job 的 gpu_need），以及带宽黑盒 bandwidth_predictor
    输出：
      - best_combos: 每个 job 的 combo（与现有系统对齐）
      - predicted_standalone_bws
      - predicted_final_bws（按你的争用规则）
      - best_total_throughput

    Args:
        contention_mode: 传递给 ClusterStateManager 的争用模式
        occupancy_seed: 控制 common 模式下占用比率的基准种子，确保多次评估一致
    """
    num_nodes = total_gpu // node_size
    
    # 1) 预枚举每种 gpu_need 的可行模式
    patterns_by_need: Dict[int, List[_NodePattern]] = {}
    for g in set(workload_requests):
        raw_patterns = _enumerate_node_patterns(g, num_nodes, node_size)
        pats: List[_NodePattern] = []
        for counts in raw_patterns:
            combo = _canonical_combo_from_counts(counts, total_gpu, node_size)
            s_bw = float(bandwidth_predictor(combo))
            involved = tuple(i for i, c in enumerate(counts) if c > 0)
            pats.append(
                _NodePattern(
                    counts=counts,
                    standalone_bw=s_bw,
                    involved_nodes=involved,
                    is_cross=(len(involved) > 1),
                )
            )
        patterns_by_need[g] = pats
    
    # 2) 计算总组合数
    total_combinations = 1
    for g in workload_requests:
        total_combinations *= len(patterns_by_need[g])
        if total_combinations > max_combinations:
            logger.warning(
                f"[BruteForce] 组合数量过多 ({total_combinations})，超过限制 ({max_combinations})。"
                f"考虑使用 minlp_offline_optimal_solver 代替。"
            )
            raise ValueError(
                f"组合数量过多 ({total_combinations})，超过限制 ({max_combinations})。"
                f"请使用 minlp_offline_optimal_solver 代替暴力搜索。"
            )
    
    if verbose:
        logger.info(f"[BruteForce] 开始暴力搜索，总组合数: {total_combinations}")
    
    # 3) 枚举所有可能的组合
    best_obj = float("-inf")
    best_patterns: Optional[List[_NodePattern]] = None
    combinations_evaluated = 0
    
    def enumerate_combinations(idx: int, caps: Tuple[int, ...], chosen: List[_NodePattern]):
        nonlocal best_obj, best_patterns, combinations_evaluated
        
        if idx == len(workload_requests):
            # 找到一个完整组合，评估它
            combinations_evaluated += 1
            
            # 评估这个组合（容量约束已在递归过程中检查）
            try:
                obj, _ = _evaluate_patterns_with_manager(
                    chosen,
                    total_gpu,
                    bandwidth_predictor,
                    node_size=node_size,
                    contention_mode=contention_mode,
                    occupancy_seed=occupancy_seed,
                )
                if obj > best_obj:
                    best_obj = obj
                    best_patterns = list(chosen)
                    if verbose and combinations_evaluated % max(1, total_combinations // 10) == 0:
                        logger.info(
                            f"[BruteForce] 评估了 {combinations_evaluated}/{total_combinations} 个组合，"
                            f"当前最优: {best_obj:.4f}"
                        )
            except Exception as e:
                logger.warning(f"[BruteForce] 评估组合失败: {e}")
            return
        
        # 为当前任务尝试所有可行的pattern
        g = workload_requests[idx]
        for pat in patterns_by_need[g]:
            # 检查容量约束
            feasible = True
            new_caps = list(caps)
            for n in range(num_nodes):
                if pat.counts[n] > new_caps[n]:
                    feasible = False
                    break
                new_caps[n] -= pat.counts[n]
            
            if feasible:
                chosen.append(pat)
                enumerate_combinations(idx + 1, tuple(new_caps), chosen)
                chosen.pop()
    
    init_caps = tuple([node_size] * num_nodes)
    enumerate_combinations(0, init_caps, [])
    
    if verbose:
        logger.info(
            f"[BruteForce] 搜索完成: 评估了 {combinations_evaluated} 个组合，"
            f"找到最优解: {best_obj:.4f}"
        )
    
    if best_patterns is None:
        raise RuntimeError("BruteForce GroundTruth 未找到可行解（workload 总和可能超过集群容量）。")
    
    # 4) 从 best_patterns 得到 combos 和带宽（按 manager 精确重算）
    best_total_throughput, per_job = _evaluate_patterns_with_manager(
        best_patterns,
        total_gpu,
        bandwidth_predictor,
        node_size=node_size,
        contention_mode=contention_mode,
        occupancy_seed=occupancy_seed,
    )
    best_combos = [x[0] for x in per_job]
    predicted_standalone_bws = [x[1] for x in per_job]
    predicted_final_bws = [x[2] for x in per_job]
    
    # 验证：确保找到的解的总吞吐量等于 best_obj
    if abs(best_total_throughput - best_obj) > 1e-6:
        logger.warning(
            f"[BruteForce] 警告：best_total_throughput ({best_total_throughput:.4f}) 与 best_obj ({best_obj:.4f}) 不一致，"
            f"差异: {abs(best_total_throughput - best_obj):.6f}"
        )
    
    return best_combos, predicted_standalone_bws, predicted_final_bws, best_total_throughput


def run_multi_tenant_simulation_offline_minlp(
    total_gpu: int,
    gpu_bw_dict_list: List,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    data_path: str,
    artifact_dir: Path,
    device: torch.device,
    contention_mode: str = "intensive",
    workload_mode: str = "fixed_sum",
    total_gpu_sum: int = 32,
    num_jobs: int = 10,
    job_sizes: List[int] = None,
    repeat_num: int = 1,
    search_if_real_data: bool = True,
    random_seed: Optional[int] = None,
    cluster_type: Optional[str] = None,
    workload_sequences: Optional[List[List[int]]] = None,
    use_brute_force: bool = False,  # 是否使用暴力搜索（用于小规模问题验证）
    progress_callback: Optional[Callable[[int], None]] = None,
) -> pd.DataFrame:
    """
    离线 MINLP GroundTruth 的多租户仿真：
    输出 df 与 run_multi_tenant_simulation 完全一致，便于 compare.py 统一对比。
    
    Args:
        contention_mode: 争用模式（与在线仿真保持一致）
        progress_callback: 可选回调函数，在每次 repeat 完成后调用，用于更新外部进度条。
    """
    if job_sizes is None:
        job_sizes = [1, 2, 4, 8]

    # 若搜索阶段不用真实数据，需要加载模型
    model = None
    if not search_if_real_data:
        model = _load_predictor(model_path, device, model_cfg)

    all_results = []

    for repeat_id in range(repeat_num):
        # workload 获取方式与 run_multi_tenant_simulation 对齐
        current_seed = None
        if random_seed is not None:
            current_seed = random_seed + repeat_id
            np.random.seed(current_seed)
            import random as _random
            _random.seed(current_seed)

        occupancy_seed = current_seed if current_seed is not None else int(np.random.randint(0, 2**31 - 1))

        if workload_sequences is not None:
            workload_requests = workload_sequences[repeat_id]
        else:
            if workload_mode == "fixed_sum":
                workload_requests = _generate_workload_fixed_sum(
                    total_gpu_sum, job_sizes, random_seed=current_seed
                )
            elif workload_mode == "random":
                workload_requests = _generate_workload_random(
                    num_jobs, job_sizes, random_seed=current_seed
                )
            else:
                raise ValueError(f"未知 workload_mode: {workload_mode}")

        logger.info(
            f"[MINLP] Repeat {repeat_id}: workload={workload_requests}, total={sum(workload_requests)}"
        )

        # ==================== 阶段一：离线搜索（全局最优） ====================
        # 关键修复：GroundTruth 必须使用真实数据进行搜索，确保找到的解在真实数据下是最优的
        # 即使 search_if_real_data=False，GroundTruth 也应该使用真实数据
        # 因为 GroundTruth 的目标是在真实数据下找到全局最优解
        logger.info("[MINLP] GroundTruth 强制使用真实数据进行搜索，确保全局最优")
        search_predictor = create_bandwidth_predictor(
            if_real_data=True,  # 强制使用真实数据
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            model=None,  # 真实数据模式不需要模型
            device=None,
            artifact_dir=None,
        )

        # 选择使用暴力搜索或B&B算法
        if use_brute_force:
            logger.info("[BruteForce] 使用暴力搜索算法（用于验证）")
            best_combos, pred_standalone_bws, pred_final_bws, pred_total = brute_force_optimal_solver(
                workload_requests=workload_requests,
                total_gpu=total_gpu,
                bandwidth_predictor=search_predictor,
                node_size=8,
                verbose=False,
                contention_mode=contention_mode,
                occupancy_seed=occupancy_seed,
            )
        else:
            logger.info("[MINLP] 使用Branch-and-Bound算法")
            best_combos, pred_standalone_bws, pred_final_bws, pred_total = minlp_offline_optimal_solver(
                workload_requests=workload_requests,
                total_gpu=total_gpu,
                bandwidth_predictor=search_predictor,
                node_size=8,
                verbose=False,
                contention_mode=contention_mode,
                occupancy_seed=occupancy_seed,
            )

        search_results = []
        for job_idx, gpu_need in enumerate(workload_requests):
            search_results.append(
                {
                    "job_idx": job_idx,
                    "gpu_need": gpu_need,
                    "combo": best_combos[job_idx].copy(),
                    "predicted_standalone_bw": pred_standalone_bws[job_idx],
                    "predicted_final_bw": pred_final_bws[job_idx],
                }
            )

        logger.info(f"[MINLP] 搜索阶段最优预测总吞吐: {pred_total:.4f}")

        # ==================== 阶段二：评估阶段（真实数据重新计算） ====================
        eval_predictor = create_bandwidth_predictor(
            if_real_data=True,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            model=None,
            device=None,
            artifact_dir=None,
        )
        eval_manager = ClusterStateManager(
            total_gpu=total_gpu,
            bandwidth_predictor=eval_predictor,
            contention_mode=contention_mode,
            occupancy_seed=occupancy_seed,
        )

        for search_result in search_results:
            job_idx = search_result["job_idx"]
            gpu_need = search_result["gpu_need"]
            combo = search_result["combo"]

            real_final_bw = eval_manager.allocate_job(job_id=job_idx, combo=combo)

            # 找真实独占带宽
            real_standalone_bw = None
            for job in eval_manager.active_jobs:
                if job["job_id"] == job_idx:
                    real_standalone_bw = job["standalone_bw"]
                    break
            if real_standalone_bw is None:
                real_standalone_bw = real_final_bw

            real_contention_ratio = real_final_bw / real_standalone_bw if real_standalone_bw > 0 else 0.0
            # 计算真实数据下的集群总吞吐量
            # 注意：这是所有活跃任务的 current_bw 之和，即当前时刻的全局总带宽
            # 对于最后一个 job_id，这个值就是整个 workload 完成分配后的全局总带宽大小
            real_cluster_throughput = sum(job["current_bw"] for job in eval_manager.active_jobs)
            num_active_jobs = len(eval_manager.active_jobs)

            combo_str = _format_combo_as_node_counts(combo, total_gpu, node_size=8)

            result = {
                "repeat_id": repeat_id if repeat_num > 1 else 0,
                "job_id": job_idx,
                "gpu_need": gpu_need,
                "combo": combo_str,
                "predicted_standalone_bw": search_result["predicted_standalone_bw"],
                "predicted_final_bw": search_result["predicted_final_bw"],
                "real_standalone_bw": real_standalone_bw,
                "real_final_bw": real_final_bw,
                "real_contention_ratio": real_contention_ratio,
                "real_cluster_throughput": real_cluster_throughput,
                "num_active_jobs": num_active_jobs,
            }
            all_results.append(result)

        if progress_callback is not None:
            progress_callback(repeat_id)

    df = pd.DataFrame(all_results)
    if repeat_num == 1 and "repeat_id" in df.columns:
        df = df.drop(columns=["repeat_id"])
    elif repeat_num > 1:
        cols = ["repeat_id"] + [col for col in df.columns if col != "repeat_id"]
        df = df[cols]

    return df
