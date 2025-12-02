"""均衡驱动启发式算法 (EHA)。"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values, prepare_model_inputs
from training.evaluator import predict_with_model
# 只有类型检查时导入，避免循环依赖
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.cluster_state import ClusterStateManager


def _predict_config_bandwidth(
    config: np.ndarray,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
) -> float:
    """
    根据当前运行环境（cluster_manager / 真实数据 / 模型）统一计算单个配置的带宽。
    """
    if cluster_manager:
        return float(cluster_manager.predict_with_contention(config))
    if if_real_data:
        bw_value, _, _ = calculate_bandwidth_values(
            config, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        return float(bw_value)

    part_bws, node_counts, total_counts = prepare_model_inputs(
        np.array([config]), total_gpu, gpu_bw_dict_list, switch_config, data_path
    )
    prediction = predict_with_model(
        model, part_bws, node_counts, total_counts, device, artifact_dir
    )
    prediction_array = np.asarray(prediction)
    return float(prediction_array.reshape(-1)[0])


def _score_config_bandwidth(
    config: np.ndarray,
    *,
    avail_gpu: Optional[Sequence[int]],
    num_dimensions: int,
    global_mode: bool,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
) -> float:
    """
    计算候选配置在普通或全局模式下的得分。
    """
    current_bw = _predict_config_bandwidth(
        config,
        model,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        data_path,
        device,
        artifact_dir,
        if_real_data,
        cluster_manager,
    )

    if not global_mode or avail_gpu is None or len(avail_gpu) == 0:
        return current_bw

    selected_gpus = set(np.where(config == 1)[0])
    remaining_gpus = [gpu for gpu in avail_gpu if gpu not in selected_gpus]
    if not remaining_gpus:
        return current_bw

    remaining_config = np.zeros(num_dimensions, dtype=int)
    remaining_config[remaining_gpus] = 1
    remaining_bw = _predict_config_bandwidth(
        remaining_config,
        model,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        data_path,
        device,
        artifact_dir,
        if_real_data,
        cluster_manager,
    )
    return current_bw + remaining_bw


def _run_subset_tree_search(
    node_id: int,
    node_gpus: Sequence[int],
    target_gpu_count: int,
    *,
    num_dimensions: int,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool,
    avail_gpu: Optional[Sequence[int]],
    allow_global_mode: bool,
) -> Optional[np.ndarray]:
    """
    在指定节点内运行带宽感知 tree_search，返回长度为 target_gpu_count 的最优子配置。
    """
    if target_gpu_count <= 0:
        return np.zeros(num_dimensions, dtype=int)
    if len(node_gpus) < target_gpu_count:
        return None

    config = np.zeros(num_dimensions, dtype=int)
    if target_gpu_count == len(node_gpus):
        config[list(node_gpus)] = 1
        return config

    start_combo = np.zeros(num_dimensions, dtype=int)
    start_combo[list(node_gpus)] = 1

    # 避免循环依赖，仅在需要时导入
    from algorithms.search import greedy_recursive_search  # noqa: WPS433

    subset_config = greedy_recursive_search(
        current_combo=start_combo,
        gpu_need=target_gpu_count,
        model=model,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        data_path=data_path,
        device=device,
        artifact_dir=artifact_dir,
        if_real_data=if_real_data,
        cluster_manager=cluster_manager,
        global_mode=global_mode and allow_global_mode,
        avail_gpu=avail_gpu if (global_mode and allow_global_mode) else None,
    )
    return subset_config


def eha_search(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    model,
    gpu_need: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool = False,
    max_candidates: int = 200,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
) -> np.ndarray | None:
    """
    均衡驱动启发式算法 (Equilibrium-driven Heuristic Algorithm, EHA)。
    
    该算法基于从真实数据中观察到的通信特性（局部性、节点数、均衡性），
    通过确定性的分步构造来快速生成高质量的GPU组合。
    
    算法分为两个阶段：
    1. **阶段一：单节点最优解搜索**（最高优先级）
       - 如果存在单个节点能够独立满足 gpu_need，则在该节点内选择最优配置
       - 单节点配置通常具有最好的局部性和通信性能
    
    2. **阶段二：跨节点构造最优解**
       - 当没有单节点能满足需求时，从多个节点组合构造配置
       - 使用两种分配策略：
         a) **策略1：剩余资源均衡** - 贪心策略，每次分配给剩余资源最多的节点
         b) **策略2：分配数量均衡** - 均匀分配策略，尝试均匀分配及±1/±2的变体
    
    Args:
        num_dimensions: GPU总数（即 total_gpu）
        avail_gpu: 可用GPU的索引列表
        model: 用于预测带宽的PyTorch模型
        gpu_need: 需要选择的GPU数量
        total_gpu: 集群中的总GPU数量
        gpu_bw_dict_list: GPU带宽字典列表
        switch_config: 交换机配置
        data_path: 带宽数据文件路径
        device: PyTorch设备（CPU或CUDA）
        artifact_dir: 模型和scaler文件所在目录
        if_real_data: 是否使用真实带宽数据（True）或模型预测（False）
        max_candidates: 最大候选配置数量，用于控制搜索规模
        cluster_manager: 集群状态管理器（用于多租户感知），如果提供则优先使用其 predict_with_contention 方法
        global_mode: 如果为 True，使用全局带宽评估（当前配置带宽 + 剩余GPU带宽）
    
    Returns:
        最优GPU组合（0/1向量），如果无法满足需求则返回 None
    """
    # ==================== 预处理：按节点对可用GPU进行分组 ====================
    # 将可用GPU按照其所属的物理节点进行分组
    # 假设每8个GPU为一个节点（节点ID = GPU索引 // 8）
    node_map: Dict[int, List[int]] = {}
    for gpu_idx in avail_gpu:
        node_id = gpu_idx // 8
        node_map.setdefault(node_id, []).append(gpu_idx)

    # ==================== 阶段一：单节点最优解搜索（最高优先级） ====================
    # 寻找所有能够独立满足 gpu_need 的节点
    # 单节点配置通常具有最好的通信性能（无跨节点通信开销）
    candidate_nodes = [node_id for node_id, gpus in node_map.items() if len(gpus) >= gpu_need]
    candidate_configs: List[np.ndarray] = []

    if candidate_nodes:
        # 如果存在单节点能满足需求，为每个候选节点生成配置
        # 简单策略：取每个节点的前 gpu_need 个GPU
        for node_id in candidate_nodes:
            config = _run_subset_tree_search(
                node_id,
                node_map[node_id],
                gpu_need,
                num_dimensions=num_dimensions,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                data_path=data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                global_mode=global_mode,
                avail_gpu=avail_gpu,
                allow_global_mode=True,
            )
            if config is not None:
                candidate_configs.append(config)
        
        # 如果有多个候选节点，使用模型或真实数据评估并选择最优配置
        if len(candidate_configs) == 1:
            return candidate_configs[0]
        
        # 批量评估所有单节点候选配置
        best_idx = -1
        best_bw = -1.0
        for idx, config in enumerate(candidate_configs):
            bw = _score_config_bandwidth(
                config,
                avail_gpu=avail_gpu,
                num_dimensions=num_dimensions,
                global_mode=global_mode,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                data_path=data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
            )
            if bw > best_bw:
                best_bw = bw
                best_idx = idx
        return candidate_configs[best_idx] if best_idx >= 0 else None

    # ==================== 阶段二：跨节点构造最优解 ====================
    # 当没有单节点能满足需求时，需要从多个节点组合构造配置
    
    # 步骤1：按节点上的可用GPU数量降序排序
    # 优先考虑GPU数量多的节点，有助于找到更好的分配方案
    sorted_nodes = sorted(node_map.items(), key=lambda item: len(item[1]), reverse=True)
    
    # 步骤2：确定最少需要几个节点（k）来满足 gpu_need
    # 贪心策略：从GPU数量最多的节点开始累加，直到总和 >= gpu_need
    k = 0
    gpu_sum = 0
    for _, gpus in sorted_nodes:
        gpu_sum += len(gpus)
        k += 1
        if gpu_sum >= gpu_need:
            break

    # 边界检查：如果所有可用GPU的总数仍无法满足需求，返回 None
    if gpu_sum < gpu_need:
        return None

    # 步骤3：构造最优的 k-节点组合
    # 使用集合来防止重复配置
    seen_configs = set()
    
    # 遍历所有可能的 k 个节点的组合
    for node_group in itertools.combinations(sorted_nodes, k):
        # 获取当前节点组合中每个节点的可用GPU数量
        group_counts = [len(gpu_list) for _, gpu_list in node_group]
        
        # 快速检查：如果当前组合的总GPU数不足，跳过
        if sum(group_counts) < gpu_need:
            continue

        # ========== 策略1：剩余资源均衡（贪心分配策略） ==========
        # 当候选配置数量较少时（<= 5），使用此策略
        # 核心思想：每次将GPU分配给当前剩余资源最多的节点
        # 这样可以保持各节点资源使用的相对均衡
        if len(candidate_configs) <= max_candidates:
            # 初始化分配方案：每个节点分配0个GPU
            alloc_remain_balance = [0] * k
            gpus_to_distribute = gpu_need
            # 临时可用资源列表，用于跟踪分配过程中的剩余资源
            temp_avail = list(group_counts)
            
            # 贪心分配：每次分配1个GPU给剩余资源最多的节点
            for _ in range(gpus_to_distribute):
                best_node_idx = -1
                max_avail = -1
                # 找到当前剩余资源最多的节点
                for i in range(k):
                    if temp_avail[i] > 0:
                        if temp_avail[i] > max_avail:
                            max_avail = temp_avail[i]
                            best_node_idx = i
                
                # 如果找到可用节点，分配1个GPU给它
                if best_node_idx != -1:
                    alloc_remain_balance[best_node_idx] += 1
                    temp_avail[best_node_idx] -= 1
                else:
                    # 如果没有可用节点，分配失败
                    break
            
            # 根据分配方案构造GPU配置
            config_remain = np.zeros(num_dimensions, dtype=int)
            is_possible_remain = True
            for i in range(k):
                node_id, gpu_list = node_group[i]
                num_to_take = alloc_remain_balance[i]
                if num_to_take == 0:
                    continue
                subset_config = _run_subset_tree_search(
                    node_id,
                    gpu_list,
                    num_to_take,
                    num_dimensions=num_dimensions,
                    model=model,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    data_path=data_path,
                    device=device,
                    artifact_dir=artifact_dir,
                    if_real_data=if_real_data,
                    cluster_manager=cluster_manager,
                    global_mode=global_mode,
                    avail_gpu=None,
                    allow_global_mode=False,
                )
                if subset_config is None:
                    is_possible_remain = False
                    break
                config_remain = np.maximum(config_remain, subset_config)
            
            # 如果分配成功，添加到候选配置列表
            if is_possible_remain:
                config_tuple = tuple(config_remain)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    candidate_configs.append(config_remain)

        # ========== 策略2：分配数量均衡（均匀分配 + 不均匀变体） ==========
        # 核心思想：尝试均匀分配，如果失败则尝试±1/±2/±3的不均匀分配变体
        
        # 步骤2.1：计算基础均匀分配方案
        # base_alloc 表示每个节点应该分配的GPU数量
        # 例如：gpu_need=8, k=3 -> base_alloc = [3, 3, 2] (8//3=2, 余数2分配给前两个)
        base_alloc = [gpu_need // k] * k
        for i in range(gpu_need % k):
            base_alloc[i] += 1
        
        # 步骤2.2：生成所有可能的分配变体
        # 包括：基础均匀分配、±1变体、±2变体、±3变体
        allocation_variants: set[Tuple[int, ...]] = set()

        def _register_variant(variant_alloc: Sequence[int]) -> None:
            for perm in set(itertools.permutations(variant_alloc)):
                allocation_variants.add(tuple(perm))

        # 2.2.1：添加基础均匀分配的所有排列
        _register_variant(base_alloc)

        # 2.2.2/2.2.3：生成±1/±2/±3的不均匀分配变体
        for delta in (1, 2, 3,4):
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    variant = list(base_alloc)
                    variant[i] += delta
                    variant[j] -= delta
                    if sum(variant) == gpu_need and all(x >= 0 for x in variant):
                        _register_variant(variant)
        
        # 步骤2.3：遍历所有分配变体，检查可行性并生成配置
        for alloc_variant in allocation_variants:
            # 检查此分配变体是否可行（每个节点都有足够的GPU）
            is_possible = all(group_counts[i] >= alloc_variant[i] for i in range(k))
            
            if is_possible:
                # 根据分配方案构造GPU配置
                config = np.zeros(num_dimensions, dtype=int)
                for idx in range(k):
                    node_id, gpu_list = node_group[idx]
                    num_to_take = alloc_variant[idx]
                    if num_to_take == 0:
                        continue
                    subset_config = _run_subset_tree_search(
                        node_id,
                        gpu_list,
                        num_to_take,
                        num_dimensions=num_dimensions,
                        model=model,
                        total_gpu=total_gpu,
                        gpu_bw_dict_list=gpu_bw_dict_list,
                        switch_config=switch_config,
                        data_path=data_path,
                        device=device,
                        artifact_dir=artifact_dir,
                        if_real_data=if_real_data,
                        cluster_manager=cluster_manager,
                        global_mode=global_mode,
                        avail_gpu=None,
                        allow_global_mode=False,
                    )
                    if subset_config is None:
                        is_possible = False
                        break
                    config = np.maximum(config, subset_config)
                
                if not is_possible:
                    continue
                # 检查并添加配置，防止重复
                config_tuple = tuple(config)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    candidate_configs.append(config)
                    # 如果达到最大候选数，提前退出
                    if len(candidate_configs) >= max_candidates:
                        break
        
        # 如果已达到最大候选数，退出外层循环
        if len(candidate_configs) >= max_candidates:
            break

    # ==================== 最终评估和选择 ====================
    # 如果没有任何候选配置，返回 None
    if not candidate_configs:
        return None

    best_idx = -1
    best_bw = -1.0
    for idx, config in enumerate(candidate_configs):
        bw = _score_config_bandwidth(
            config,
            avail_gpu=avail_gpu,
            num_dimensions=num_dimensions,
            global_mode=global_mode,
            model=model,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            device=device,
            artifact_dir=artifact_dir,
            if_real_data=if_real_data,
            cluster_manager=cluster_manager,
        )
        if bw > best_bw:
            best_bw = bw
            best_idx = idx
    return candidate_configs[best_idx] if best_idx >= 0 else None


# eha, old版本
def eha_search_old(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    model,
    gpu_need: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool = False,
    max_candidates: int = 50,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    均衡驱动启发式算法 (Equilibrium-driven Heuristic Algorithm, EHA) - Old版本。
    
    该算法基于从真实数据中观察到的通信特性（局部性、节点数、均衡性），
    通过确定性的分步构造来快速生成高质量的GPU组合，速度远超遗传算法。
    
    注意：这是old版本，使用简单的策略（直接取前n个GPU），不使用tree_search优化。
    与eha_search的主要区别在于：
    - 单节点阶段：直接取前gpu_need个GPU，不使用tree_search
    - 跨节点阶段：直接取前n个GPU，不使用tree_search
    
    算法分为两个阶段：
    1. **阶段一：单节点最优解搜索**（最高优先级）
       - 如果存在单个节点能够独立满足 gpu_need，则在该节点内选择前gpu_need个GPU
       - 单节点配置通常具有最好的局部性和通信性能
    
    2. **阶段二：跨节点构造最优解**
       - 当没有单节点能满足需求时，从多个节点组合构造配置
       - 使用两种分配策略：
         a) **策略1：剩余资源均衡** - 贪心策略，每次分配给剩余资源最多的节点
         b) **策略2：分配数量均衡** - 均匀分配策略，尝试均匀分配的所有排列
    
    Args:
        num_dimensions: GPU总数（即 total_gpu）
        avail_gpu: 可用GPU的索引列表
        model: 用于预测带宽的PyTorch模型
        gpu_need: 需要选择的GPU数量
        total_gpu: 集群中的总GPU数量
        gpu_bw_dict_list: GPU带宽字典列表
        switch_config: 交换机配置
        data_path: 带宽数据文件路径
        device: PyTorch设备（CPU或CUDA）
        artifact_dir: 模型和scaler文件所在目录
        if_real_data: 是否使用真实带宽数据（True）或模型预测（False）
        max_candidates: 最大候选配置数量，用于控制搜索规模（默认50，与old版本保持一致）
        cluster_manager: 集群状态管理器（用于多租户感知），如果提供则优先使用其 predict_with_contention 方法
        global_mode: 如果为 True，使用全局带宽评估（当前配置带宽 + 剩余GPU带宽）
        verbose: 是否打印详细日志
    
    Returns:
        最优GPU组合（0/1向量），如果无法满足需求则返回 None
    """
    # ==================== 预处理：按节点对可用GPU进行分组 ====================
    # 将可用GPU按照其所属的物理节点进行分组
    # 假设每8个GPU为一个节点（节点ID = GPU索引 // 8）
    node_map: Dict[int, List[int]] = {}
    for gpu_idx in avail_gpu:
        node_id = gpu_idx // 8
        node_map.setdefault(node_id, []).append(gpu_idx)

    if verbose:
        print("可用GPU已按节点分组:", {k: len(v) for k, v in node_map.items()})

    # ==================== 阶段一：单节点最优解搜索（最高优先级） ====================
    # 寻找所有能够独立满足 gpu_need 的节点
    # 单节点配置通常具有最好的通信性能（无跨节点通信开销）
    candidate_nodes = [node_id for node_id, gpus in node_map.items() if len(gpus) >= gpu_need]
    candidate_configs: List[np.ndarray] = []

    if candidate_nodes:
        if verbose:
            print(f"阶段一：在 {len(candidate_nodes)} 个候选单节点中寻找最优解...")
        
        # Old版本策略：为每个候选节点创建一个测试配置（简单地取前n个GPU）
        for node_id in candidate_nodes:
            selected_gpus = node_map[node_id][:gpu_need]
            config = np.zeros(num_dimensions, dtype=int)
            config[selected_gpus] = 1
            candidate_configs.append(config)
        
        # 如果有多个候选节点，使用统一评估方法选择最优配置
        if len(candidate_configs) == 1:
            return candidate_configs[0]
        
        # 批量评估所有单节点候选配置
        best_idx = -1
        best_bw = -1.0
        for idx, config in enumerate(candidate_configs):
            bw = _score_config_bandwidth(
                config,
                avail_gpu=avail_gpu,
                num_dimensions=num_dimensions,
                global_mode=global_mode,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                data_path=data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
            )
            if bw > best_bw:
                best_bw = bw
                best_idx = idx
        
        if verbose:
            print(f"阶段一完成。最优单节点配置带宽预测为: {best_bw:.2f}")
        
        return candidate_configs[best_idx] if best_idx >= 0 else None

    # ==================== 阶段二：跨节点构造最优解 ====================
    # 当没有单节点能满足需求时，需要从多个节点组合构造配置
    if verbose:
        print("阶段二：无单节点可满足，开始构造跨节点最优解...")
    
    # 步骤1：按节点上的可用GPU数量降序排序
    # 优先考虑GPU数量多的节点，有助于找到更好的分配方案
    sorted_nodes = sorted(node_map.items(), key=lambda item: len(item[1]), reverse=True)
    
    # 步骤2：确定最少需要几个节点（k）来满足 gpu_need
    # 贪心策略：从GPU数量最多的节点开始累加，直到总和 >= gpu_need
    k = 0
    gpu_sum = 0
    for _, gpus in sorted_nodes:
        gpu_sum += len(gpus)
        k += 1
        if gpu_sum >= gpu_need:
            break

    # 边界检查：如果所有可用GPU的总数仍无法满足需求，返回 None
    if gpu_sum < gpu_need:
        if verbose:
            print("错误：所有可用GPU数量仍无法满足需求！")
        return None

    if verbose:
        print(f"最少需要 {k} 个节点来满足 {gpu_need} 个GPU的需求。")

    # 步骤3：构造最优的 k-节点组合
    # 使用集合来防止重复配置
    final_configs: List[np.ndarray] = []
    seen_configs = set()

    # 遍历所有可能的 k 个节点的组合
    for node_group_tuple in itertools.combinations(sorted_nodes, k):
        group_avail_gpus = [len(gpus) for _, gpus in node_group_tuple]
        
        # 快速检查：如果当前组合的总GPU数不足，跳过
        if sum(group_avail_gpus) < gpu_need:
            continue
        
        # ========== 策略1：剩余资源均衡（贪心分配策略） ==========
        # Old版本：当候选配置数量较少时（<= 5），使用此策略
        # 核心思想：每次将GPU分配给当前剩余资源最多的节点
        # 这样可以保持各节点资源使用的相对均衡
        if len(final_configs) <= 5:
            # 初始化分配方案：每个节点分配0个GPU
            alloc_remain_balance = [0] * k
            gpus_to_distribute = gpu_need
            # 临时可用资源列表，用于跟踪分配过程中的剩余资源
            temp_avail = list(group_avail_gpus)
            
            # 贪心分配：每次分配1个GPU给剩余资源最多的节点
            for _ in range(gpus_to_distribute):
                best_node_idx = -1
                max_avail = -1
                # 找到当前剩余资源最多的节点
                for i in range(k):
                    if temp_avail[i] > 0:
                        if temp_avail[i] > max_avail:
                            max_avail = temp_avail[i]
                            best_node_idx = i
                
                # 如果找到可用节点，分配1个GPU给它
                if best_node_idx != -1:
                    alloc_remain_balance[best_node_idx] += 1
                    temp_avail[best_node_idx] -= 1
                else:
                    # 如果没有可用节点，分配失败
                    break
            
            # 根据分配方案构造GPU配置（Old版本：直接取前n个GPU）
            config_remain = np.zeros(num_dimensions, dtype=int)
            is_possible_remain = True
            for i in range(k):
                _, gpu_list = node_group_tuple[i]
                num_to_take = alloc_remain_balance[i]
                if len(gpu_list) >= num_to_take:
                    # Old版本策略：直接取前num_to_take个GPU
                    selected_on_node = gpu_list[:num_to_take]
                    config_remain[selected_on_node] = 1
                else:
                    is_possible_remain = False
                    break
            
            # 如果分配成功，添加到候选配置列表
            if is_possible_remain:
                config_tuple = tuple(config_remain)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    final_configs.append(config_remain)

        # ========== 策略2：分配数量均衡（均匀分配 + 所有排列） ==========
        # Old版本策略：尝试均匀分配的所有排列组合
        # 步骤2.1：计算基础均匀分配方案
        # base_alloc 表示每个节点应该分配的GPU数量
        # 例如：gpu_need=8, k=3 -> base_alloc = [3, 3, 2] (8//3=2, 余数2分配给前两个)
        base_alloc = [gpu_need // k] * k
        for i in range(gpu_need % k):
            base_alloc[i] += 1
        
        # 步骤2.2：生成该方案所有不重复的排列组合
        # 使用 set 来自动处理重复的排列（例如 [3,3,2] 会产生重复）
        unique_alloc_permutations = set(itertools.permutations(base_alloc))
        
        # 步骤2.3：遍历每一种不重复的分配排列
        for alloc_permutation in unique_alloc_permutations:
            # 检查此分配排列是否可行（每个节点都有足够的GPU）
            is_possible = all(group_avail_gpus[i] >= alloc_permutation[i] for i in range(k))
            
            if is_possible:
                # 根据分配方案构造GPU配置（Old版本：直接取前n个GPU）
                config = np.zeros(num_dimensions, dtype=int)
                for i in range(k):
                    # 获取节点信息
                    _, gpu_list = node_group_tuple[i]
                    # 从当前的排列中获取要拿走的GPU数量
                    num_to_take = alloc_permutation[i]
                    # Old版本策略：从节点上直接取前num_to_take个GPU
                    selected_on_node = gpu_list[:num_to_take]
                    # 更新配置向量
                    config[selected_on_node] = 1
                
                # 检查并添加配置，防止重复
                config_tuple = tuple(config)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    final_configs.append(config)
                    # 如果达到最大候选数，提前退出
                    if len(final_configs) >= max_candidates:
                        break
        
        # 如果已达到最大候选数，退出外层循环
        if len(final_configs) >= max_candidates:
            break

    # ==================== 备用策略：如果没有任何候选配置 ====================
    if not final_configs:
        if verbose:
            print("备用策略：直接从最富余的节点中贪心拾取。")
        config = np.zeros(num_dimensions, dtype=int)
        gpus_taken = 0
        for node_id, gpu_list in sorted_nodes:
            can_take = gpu_need - gpus_taken
            to_take = min(can_take, len(gpu_list))
            config[gpu_list[:to_take]] = 1
            gpus_taken += to_take
            if gpus_taken == gpu_need:
                break
        final_configs.append(config)

    if verbose:
        print(f"共生成 {len(final_configs)} 个高质量候选配置，送入模型进行最终决策。")

    # ==================== 最终评估和选择 ====================
    # 使用统一的评估方法（支持global_mode、cluster_manager等）
    best_idx = -1
    best_bw = -1.0
    for idx, config in enumerate(final_configs):
        bw = _score_config_bandwidth(
            config,
            avail_gpu=avail_gpu,
            num_dimensions=num_dimensions,
            global_mode=global_mode,
            model=model,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            device=device,
            artifact_dir=artifact_dir,
            if_real_data=if_real_data,
            cluster_manager=cluster_manager,
        )
        if bw > best_bw:
            best_bw = bw
            best_idx = idx
    
    return final_configs[best_idx] if best_idx >= 0 else None


