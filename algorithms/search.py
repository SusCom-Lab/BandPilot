"""搜索与组合生成算法集合（支持多租户感知）。"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values, prepare_model_inputs
from core.gpu_config import generate_data_minmax_restricted
from training.evaluator import predict_with_model
# 只有类型检查时导入，避免循环依赖
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.cluster_state import ClusterStateManager
logger = logging.getLogger(__name__)


def generate_next_combos(combo: np.ndarray) -> np.ndarray:
    """生成所有将单个1变为0的组合。"""
    indices = np.where(combo == 1)[0]
    next_combos = np.tile(combo, (len(indices), 1))
    next_combos[np.arange(len(indices)), indices] = 0
    return next_combos


def generate_add_combos(combo: np.ndarray, avail_gpu: Sequence[int], num_dimensions: int) -> np.ndarray:
    """生成所有在可用GPU上增加一个1的组合。"""
    current_zeros = np.where(combo == 0)[0]
    valid_indices = [idx for idx in current_zeros if idx in avail_gpu]
    if not valid_indices:
        return np.array([])
    next_combos = np.tile(combo, (len(valid_indices), 1))
    for i, idx in enumerate(valid_indices):
        next_combos[i, idx] = 1
    return next_combos


def find_best_2gpu_combo(
    avail_gpu: Sequence[int],
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    num_dimensions: int,
    data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool = False,
    cluster_manager: Optional['ClusterStateManager'] = None,
) -> Optional[np.ndarray]:
    """遍历所有2卡组合，寻找最佳起点。"""
    combos = []
    for pair in itertools.combinations(avail_gpu, 2):
        combo = np.zeros(num_dimensions, dtype=int)
        combo[list(pair)] = 1
        combos.append(combo)
    if not combos:
        return None
    all_combos = np.array(combos)
    
    # 优先使用 cluster_manager 进行带宽评估
    if cluster_manager:
        # 多租户场景：直接依赖集群状态管理器的实时预测，避免重复构造模型输入
        best_idx = -1
        best_bw = -1.0
        for idx, combo in enumerate(all_combos):
            bw = cluster_manager.predict_with_contention(combo)
            if bw > best_bw:
                best_bw = bw
                best_idx = idx
        return all_combos[best_idx] if best_idx >= 0 else None
    elif if_real_data:
        # 真实数据模式：遍历所有组合并用测得的带宽排序
        best_idx = -1
        best_bw = -1
        for idx, combo in enumerate(all_combos):
            bw_value, _, _ = calculate_bandwidth_values(
                combo, total_gpu, gpu_bw_dict_list, switch_config, data_path
            )
            if bw_value > best_bw:
                best_bw = bw_value
                best_idx = idx
        return all_combos[best_idx] if best_idx >= 0 else None

    # 模型预测模式：一次性批量构造输入，减少推理次数
    part_bws, node_counts, total_counts = prepare_model_inputs(
        all_combos, total_gpu, gpu_bw_dict_list, switch_config, data_path
    )
    scores = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)
    return all_combos[int(np.argmax(scores))]


def greedy_recursive_search(
    current_combo: np.ndarray,
    gpu_need: int,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool = False,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
    avail_gpu: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    递归式贪心搜索：不断剔除一个GPU直至满足需求。
    
    :param global_mode: 如果为 True，使用全局带宽评估（当前配置带宽 + 剩余GPU带宽）
    :param avail_gpu: 可用GPU列表，global_mode=True 时需要提供
    """
    ones_count = int(np.sum(current_combo))
    if ones_count == gpu_need:
        return current_combo

    # 生成所有可能的单卡剔除方案
    candidate_combos = generate_next_combos(current_combo)
    
    # 修改：带宽评估逻辑，优先使用 cluster_manager
    scores = []
    if cluster_manager:
        # 多租户模式：考虑争用
        if global_mode and avail_gpu is not None:
            # 全局模式：评估全局带宽（当前配置 + 剩余GPU）
            num_dimensions = len(current_combo)
            for combo in candidate_combos:
                # 每个候选都需要结合剩余 GPU 进行全局评估
                bw = _evaluate_global_bandwidth(
                    combo, avail_gpu, num_dimensions, cluster_manager
                )
                scores.append(bw)
        else:
            # 普通模式：只评估当前配置
            for combo in candidate_combos:
                bw = cluster_manager.predict_with_contention(combo)
                scores.append(bw)
    elif if_real_data:
        for combo in candidate_combos:
            bw_value, _, _ = calculate_bandwidth_values(
                combo, total_gpu, gpu_bw_dict_list, switch_config, data_path
            )
            scores.append(bw_value)
    else:
        # 通过一次批量推理拿到所有候选配置的带宽预测
        part_bws, node_counts, total_counts = prepare_model_inputs(
            candidate_combos, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        scores = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)
        # predict_with_model 返回的是 tensor 或 array，确保转为 list
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif hasattr(scores, '__iter__') and not isinstance(scores, (list, tuple)):
            scores = list(scores)
            
    # 选出带宽得分最高的组合并继续向下递归
    # 根据得分选择下一轮起点，继续递归逼近目标 GPU 数
    best_idx = int(np.argmax(scores))
    best_next_combo = candidate_combos[best_idx]
    
    return greedy_recursive_search(
        best_next_combo,
        gpu_need,
        model,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        data_path,
        device,
        artifact_dir,
        if_real_data,
        cluster_manager=cluster_manager,  # 递归传递
        global_mode=global_mode,  # 递归传递
        avail_gpu=avail_gpu,  # 递归传递
    )


def tree_search_only(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    model,
    gpu_need: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool = False,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
) -> Optional[np.ndarray]:
    """
    改进的搜索算法：从最大集合剔除，支持节点插入优化，且可选全局带宽评估。
    
    :param num_dimensions: GPU总数
    :param avail_gpu: 可用GPU列表
    :param model: 预测模型
    :param gpu_need: 需要的GPU数量
    :param total_gpu: 总GPU数
    :param gpu_bw_dict_list: GPU带宽字典
    :param switch_config: 交换机配置
    :param data_path: 数据路径
    :param device: 设备
    :param artifact_dir: 模型目录
    :param if_real_data: 是否使用真实数据
    :param global_mode: 是否基于“已选 + 剩余”计算全局带宽（需 cluster_manager）
    :return: 最优GPU组合
    """
    if len(avail_gpu) < gpu_need:
        logger.warning("可用GPU数量小于需求数量")
        return None

    if len(avail_gpu) == gpu_need:
        return generate_data_minmax_restricted(
            1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
        )[0]

    max_gpu_combo = generate_data_minmax_restricted(
        1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
    )[0]
    complete_host_list = [[int(8 * i + e) for e in range(0, 8)] for i in range(0, int(num_dimensions / 8))]
    complete_node_list = [[int(4 * i + e) for e in range(0, 4)] for i in range(0, int(num_dimensions / 4))]
    avail_set = set(avail_gpu)

    def _score_combo(combo: np.ndarray, use_real_data: bool) -> float:
        """统一输出当前组合的得分，必要时启用全局带宽评估。"""
        if global_mode and cluster_manager:
            return _evaluate_global_bandwidth(combo, avail_gpu, num_dimensions, cluster_manager)
        return _evaluate_bandwidth(
            combo,
            model,
            total_gpu,
            gpu_bw_dict_list,
            switch_config,
            data_path,
            device,
            artifact_dir,
            use_real_data,
            cluster_manager,
        )

    def _run_tree_paths(start_combo: np.ndarray, use_real_data: bool) -> np.ndarray:
        """从指定起点运行单卡剔除路径。"""
        combo_from_subtract = greedy_recursive_search(
            current_combo=start_combo,
            gpu_need=gpu_need,
            model=model,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            device=device,
            artifact_dir=artifact_dir,
            if_real_data=use_real_data,
            cluster_manager=cluster_manager,
            global_mode=global_mode,
            avail_gpu=avail_gpu,
        )
        return combo_from_subtract

    def _select_best_complete_group(group_list: List[List[int]]) -> Optional[np.ndarray]:
        """从可用节点/主机中选出带宽最高的一个组合。"""
        candidates = [group for group in group_list if set(group).issubset(avail_set)]
        if not candidates:
            return None

        chosen_group = candidates[0]
        if len(candidates) > 1:
            best_bw = float('-inf')
            for group in candidates:
                temp_combo = np.zeros(num_dimensions, dtype=int)
                temp_combo[group] = 1
                bw = _score_combo(temp_combo, use_real_data=True)
                if bw > best_bw:
                    best_bw = bw
                    chosen_group = group

        combo = np.zeros(num_dimensions, dtype=int)
        combo[chosen_group] = 1
        return combo

    def _attempt_insert(group_list: List[List[int]], limit: int) -> Optional[np.ndarray]:
        """若所需 GPU 在 limit 内，尝试整节点/整机插入并直接运行树搜索。"""
        if gpu_need > limit:
            return None
        start_combo = _select_best_complete_group(group_list)
        if start_combo is None:
            return None
        # 历史逻辑：插入后统一按真实数据路径评估，以保证带宽排序一致
        return _run_tree_paths(start_combo, use_real_data=True)

    # 先尝试四卡节点，再尝试八卡主机，均失败时退回最大集合
    node_result = _attempt_insert(complete_node_list, limit=4)
    if node_result is not None:
        return node_result

    host_result = _attempt_insert(complete_host_list, limit=8)
    if host_result is not None:
        return host_result

    return _run_tree_paths(max_gpu_combo, use_real_data=if_real_data)


def _try_node_insert_optimization(
    gpu_need: int,
    num_dimensions: int,
    avail_gpu: Sequence[int],
    complete_node_list: List[List[int]],
    complete_host_list: List[List[int]],
    max_gpu_combo: np.ndarray,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool = False,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    尝试节点插入优化：检查是否可以使用四卡节点或八卡主机作为起点。
    
    :param global_mode: 如果为 True，使用全局带宽评估
    :return: (success_insert, combo_result) 如果成功插入则返回 (True, combo)，否则返回 (False, None)
    """
    success_insert = False
    best_gpu = np.zeros(num_dimensions, dtype=int)
    
    # 检查四卡节点插入
    if gpu_need <= 4:
        # 收集所有完全可用的 4 卡节点
        available_nodes = []
        for comp in complete_node_list:
            if set(comp).issubset(set(avail_gpu)):
                available_nodes.append(comp)
        if available_nodes:
            success_insert = True
            if len(available_nodes) > 1:
                max_bw = -1
                best_node = None
                for node in available_nodes:
                    temp_gpu = np.zeros(num_dimensions, dtype=int)
                    temp_gpu[node] = 1
                    bw_value, _, _ = calculate_bandwidth_values(
                        temp_gpu, total_gpu, gpu_bw_dict_list, switch_config, data_path
                    )
                    if bw_value > max_bw:
                        max_bw = bw_value
                        best_node = node
                best_gpu[best_node] = 1
            else:
                best_gpu[available_nodes[0]] = 1
            
            combo_result = greedy_recursive_search(
                current_combo=best_gpu,
                gpu_need=gpu_need,
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
            )
            return (True, combo_result)
    
    # 检查八卡主机插入
    if gpu_need <= 8 and not success_insert:
        # 收集所有完全可用的 8 卡整机
        available_hosts = []
        for comp in complete_host_list:
            if set(comp).issubset(set(avail_gpu)):
                available_hosts.append(comp)
        
        if available_hosts:
            success_insert = True
            if len(available_hosts) > 1:
                max_bw = -1
                best_host = None
                for host in available_hosts:
                    temp_gpu = np.zeros(num_dimensions, dtype=int)
                    temp_gpu[host] = 1
                    bw_value, _, _ = calculate_bandwidth_values(
                        temp_gpu, total_gpu, gpu_bw_dict_list, switch_config, data_path
                    )
                    if bw_value > max_bw:
                        max_bw = bw_value
                        best_host = host
                best_gpu[best_host] = 1
            else:
                best_gpu[available_hosts[0]] = 1
            
            combo_result = greedy_recursive_search(
                current_combo=best_gpu,
                gpu_need=gpu_need,
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
            )
            return (True, combo_result)
    
    return (False, None)


def _evaluate_bandwidth(
    combo: np.ndarray,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
) -> float:
    """
    统一带宽评估函数：根据 cluster_manager 和 if_real_data 选择评估方法。
    
    :return: 带宽值
    """
    if cluster_manager:
        # 首选在线预测，保证与调度器的一致性
        return cluster_manager.predict_with_contention(combo)
    elif if_real_data:
        # 真实数据路径：直接查表/插值得到带宽
        bw_value, _, _ = calculate_bandwidth_values(
            combo, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        return bw_value
    else:
        # 模型推理路径：单条配置也复用 prepare_model_inputs，避免重复代码
        part_bws_list, node_counts_list, total_counts_list = prepare_model_inputs(
            np.array([combo]), total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        bw_value = predict_with_model(
            model, part_bws_list, node_counts_list, total_counts_list, device, artifact_dir
        )[0]
        return bw_value


def _evaluate_global_bandwidth(
    config: np.ndarray,
    avail_gpu: Sequence[int],
    num_dimensions: int,
    cluster_manager: Optional['ClusterStateManager'],
) -> float:
    """
    评估全局带宽：当前配置带宽 + 剩余可用GPU带宽。
    
    当 global=True 时，不仅评估当前配置的带宽，还评估采用当前配置后
    剩余可用GPU的带宽，两者相加得到全局带宽。
    
    :param config: 当前GPU配置（0/1向量）
    :param avail_gpu: 可用GPU列表
    :param num_dimensions: GPU总数
    :param cluster_manager: 集群状态管理器
    :return: 全局带宽值（当前配置带宽 + 剩余GPU带宽）
    """
    # 评估当前配置的带宽，反映已分配训练任务的收益
    current_bw = cluster_manager.predict_with_contention(config)
    
    # 计算剩余可用GPU（avail_gpu 中不在 config 中的 GPU）
    selected_gpus = set(np.where(config == 1)[0])
    remaining_gpus = [gpu for gpu in avail_gpu if gpu not in selected_gpus]
    
    # 如果剩余GPU为空，只返回当前带宽
    if not remaining_gpus:
        return current_bw
    
    # 构造剩余GPU的配置向量（所有剩余GPU都选中）
    remaining_config = np.zeros(num_dimensions, dtype=int)
    remaining_config[remaining_gpus] = 1
    
    # 评估剩余GPU配置的带宽
    remaining_bw = cluster_manager.predict_with_contention(remaining_config)
    
    # 返回两者之和
    return current_bw + remaining_bw


def _compare_and_select_best(
    combo_from_subtract: Optional[np.ndarray],
    combo_from_eha: Optional[np.ndarray],
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    success_insert: bool = False,
    global_mode: bool = False,
    avail_gpu: Optional[Sequence[int]] = None,
    num_dimensions: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    比较两种方法的结果，选择带宽更高的组合。
    
    :param combo_from_subtract: 从剔除方法得到的结果
    :param combo_from_eha: 从EHA方法得到的结果
    :param success_insert: 是否成功使用了节点插入（用于else分支的特殊处理）
    :param global_mode: 如果为 True，使用全局带宽评估
    :param avail_gpu: 可用GPU列表，global_mode=True 时需要提供
    :param num_dimensions: GPU总数，global_mode=True 时需要提供
    :return: 最优GPU组合
    """
    # 处理 None 情况
    candidates: List[Tuple[str, np.ndarray]] = []
    if combo_from_eha is not None:
        candidates.append(("eha", combo_from_eha))
    if combo_from_subtract is not None:
        candidates.append(("subtract", combo_from_subtract))

    if not candidates:
        logger.warning("所有候选结果均为 None")
        return None

    if len(candidates) == 1:
        logger.warning("只有一个候选结果，直接返回")
        return candidates[0][1]

    if cluster_manager:
        # 多租户模式下直接重用在线带宽评估
        if global_mode and avail_gpu is not None and num_dimensions is not None:
            bw_list = [
                _evaluate_global_bandwidth(combo, avail_gpu, num_dimensions, cluster_manager)
                for _, combo in candidates
            ]
        else:
            bw_list = [cluster_manager.predict_with_contention(combo) for _, combo in candidates]
    elif if_real_data:
        bw_list = [
            calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, data_path)[0]
            for _, combo in candidates
        ]
    else:
        combo_array = np.array([combo for _, combo in candidates])
        part_bws_list, node_counts_list, total_counts_list = prepare_model_inputs(
            combo_array, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        bw_pred = predict_with_model(
            model, part_bws_list, node_counts_list, total_counts_list, device, artifact_dir
        )
        if hasattr(bw_pred, 'tolist'):
            bw_list = bw_pred.tolist()
        elif hasattr(bw_pred, '__iter__') and not isinstance(bw_pred, (list, tuple)):
            bw_list = list(bw_pred)
        else:
            bw_list = bw_pred

        if success_insert:
            # 若经历了节点插入优化，需要用真实带宽修正 subtract 路径的得分
            for idx, (label, combo) in enumerate(candidates):
                if label == "subtract":
                    bw_list[idx], _, _ = calculate_bandwidth_values(
                        combo, total_gpu, gpu_bw_dict_list, switch_config, data_path
                    )
                    break

    max_idx = int(np.argmax(bw_list))
    if not if_real_data:
        logger.info(f"预测的最优带宽为：{bw_list[max_idx]},选择的算法是：{candidates[max_idx][0]}")
    return candidates[max_idx][1]


def improved_searching_algo(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    model,
    gpu_need: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool = False,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
) -> Optional[np.ndarray]:
    """
    改进的搜索算法：支持多租户感知的带宽预测。
    
    :param num_dimensions: GPU总数
    :param avail_gpu: 可用GPU列表
    :param model: 预测模型
    :param gpu_need: 需要的GPU数量
    :param total_gpu: 总GPU数
    :param gpu_bw_dict_list: GPU带宽字典
    :param switch_config: 交换机配置
    :param data_path: 数据路径
    :param device: 设备
    :param artifact_dir: 模型目录
    :param if_real_data: 是否使用真实数据
    :param cluster_manager: 集群状态管理器（用于多租户感知）
    :param global_mode: 如果为 True，使用全局带宽评估（当前配置带宽 + 剩余GPU带宽）
    :return: 最优GPU组合
    """
    if len(avail_gpu) < gpu_need:
        logger.warning("可用GPU数量小于需求数量")
        return None

    if len(avail_gpu) == gpu_need:
        return generate_data_minmax_restricted(
            1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
        )[0]
    # 第一条路径：原有的从全部可用GPU开始剔除的方法
    max_gpu_combo = generate_data_minmax_restricted(
        1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
    )[0]

    # 准备节点和主机列表
    complete_host_list = [[int(8 * i + e) for e in range(0, 8)] for i in range(0, int(num_dimensions / 8))]
    complete_node_list = [[int(4 * i + e) for e in range(0, 4)] for i in range(0, int(num_dimensions / 4))]

    # 尝试节点插入优化：若存在整节点/整机可用，直接以其为起点显著减少搜索空间
    success_insert, combo_from_subtract = _try_node_insert_optimization(
        gpu_need=gpu_need,
        num_dimensions=num_dimensions,
        avail_gpu=avail_gpu,
        complete_node_list=complete_node_list,
        complete_host_list=complete_host_list,
        max_gpu_combo=max_gpu_combo,
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
    )

    # 如果没有成功使用节点插入，使用原始的方法
    if not success_insert:
        combo_from_subtract = greedy_recursive_search(
            current_combo=max_gpu_combo,
            gpu_need=gpu_need,
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
        )

    # 第二条路径：采用启发式算法
    from algorithms.eha import eha_search

    combo_from_eha = eha_search(
        num_dimensions,
        avail_gpu,
        model,
        gpu_need,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        data_path,
        device,
        artifact_dir,
        if_real_data=if_real_data,  # 修复：使用函数的 if_real_data 参数，而不是硬编码
        cluster_manager=cluster_manager,
        global_mode=global_mode,
    )

    # 比较两种方法的结果，选择带宽更高的
    return _compare_and_select_best(
        combo_from_subtract=combo_from_subtract,
        combo_from_eha=combo_from_eha,
        model=model,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        data_path=data_path,
        device=device,
        artifact_dir=artifact_dir,
        if_real_data=if_real_data,
        cluster_manager=cluster_manager,
        success_insert=success_insert,
        global_mode=global_mode,
        avail_gpu=avail_gpu,
        num_dimensions=num_dimensions,
    )

