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
        best_idx = -1
        best_bw = -1.0
        for idx, combo in enumerate(all_combos):
            bw = cluster_manager.predict_with_contention(combo)
            if bw > best_bw:
                best_bw = bw
                best_idx = idx
        return all_combos[best_idx] if best_idx >= 0 else None
    elif if_real_data:
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

    candidate_combos = generate_next_combos(current_combo)
    
    # 修改：带宽评估逻辑，优先使用 cluster_manager
    scores = []
    if cluster_manager:
        # 多租户模式：考虑争用
        if global_mode and avail_gpu is not None:
            # 全局模式：评估全局带宽（当前配置 + 剩余GPU）
            num_dimensions = len(current_combo)
            for combo in candidate_combos:
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
        part_bws, node_counts, total_counts = prepare_model_inputs(
            candidate_combos, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        scores = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)
        # predict_with_model 返回的是 tensor 或 array，确保转为 list
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif hasattr(scores, '__iter__') and not isinstance(scores, (list, tuple)):
            scores = list(scores)
            
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
) -> Optional[np.ndarray]:
    """
    改进的搜索算法：从最大集合剔除，支持节点插入优化。
    
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

    # 检查是否可以使用节点插入的优化方法
    success_insert = False
    best_gpu = np.zeros(num_dimensions, dtype=int)
    complete_host_list = [[int(8 * i + e) for e in range(0, 8)] for i in range(0, int(num_dimensions / 8))]
    complete_node_list = [[int(4 * i + e) for e in range(0, 4)] for i in range(0, int(num_dimensions / 4))]

    if if_real_data:
        # 检查四卡节点插入
        if gpu_need <= 4:
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

                combo_from_subtract = greedy_recursive_search(
                    current_combo=best_gpu,
                    gpu_need=gpu_need,
                    model=model,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    data_path=data_path,
                    device=device,
                    artifact_dir=artifact_dir,
                    if_real_data=True,
                    cluster_manager=cluster_manager,
                )
                return combo_from_subtract

        # 检查八卡主机插入
        if gpu_need <= 8 and not success_insert:
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

                combo_from_subtract = greedy_recursive_search(
                    current_combo=best_gpu,
                    gpu_need=gpu_need,
                    model=model,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    data_path=data_path,
                    device=device,
                    artifact_dir=artifact_dir,
                    if_real_data=True,
                    cluster_manager=cluster_manager,
                )
                return combo_from_subtract

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
                if_real_data=True,
                cluster_manager=cluster_manager,
            )
            return combo_from_subtract
    else:
        if gpu_need <= 4:
            # 检查四卡节点插入
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

                combo_from_subtract = greedy_recursive_search(
                    current_combo=best_gpu,
                    gpu_need=gpu_need,
                    model=model,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    data_path=data_path,
                    device=device,
                    artifact_dir=artifact_dir,
                    if_real_data=True,
                    cluster_manager=cluster_manager,
                )
                return combo_from_subtract

        # 检查八卡主机插入
        if gpu_need <= 8 and not success_insert:
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

                combo_from_subtract = greedy_recursive_search(
                    current_combo=best_gpu,
                    gpu_need=gpu_need,
                    model=model,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    data_path=data_path,
                    device=device,
                    artifact_dir=artifact_dir,
                    if_real_data=True,
                    cluster_manager=cluster_manager,
                )
                return combo_from_subtract

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
            )
            return combo_from_subtract

    return None


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
        return cluster_manager.predict_with_contention(combo)
    elif if_real_data:
        bw_value, _, _ = calculate_bandwidth_values(
            combo, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        return bw_value
    else:
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
    # 评估当前配置的带宽
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
    if combo_from_subtract is None:
        logger.warning("combo_from_subtract 为 None，使用 EHA 结果")
        return combo_from_eha
    
    if combo_from_eha is None:
        logger.warning("combo_from_eha 为 None，使用 subtract 结果")
        return combo_from_subtract
    
    # 评估带宽
    if cluster_manager:
        if global_mode and avail_gpu is not None and num_dimensions is not None:
            # 全局模式：评估全局带宽（当前配置 + 剩余GPU）
            bw_subtract = _evaluate_global_bandwidth(
                combo_from_subtract, avail_gpu, num_dimensions, cluster_manager
            )
            bw_eha = _evaluate_global_bandwidth(
                combo_from_eha, avail_gpu, num_dimensions, cluster_manager
            )
        else:
            # 普通模式：只评估当前配置
            bw_subtract = cluster_manager.predict_with_contention(combo_from_subtract)
            bw_eha = cluster_manager.predict_with_contention(combo_from_eha)
    elif if_real_data:
        bw_subtract, _, _ = calculate_bandwidth_values(
            combo_from_subtract, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        bw_eha, _, _ = calculate_bandwidth_values(
            combo_from_eha, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
    else:
        # 使用模型预测
        part_bws_list_subtract, node_counts_list_subtract, total_counts_list_subtract = prepare_model_inputs(
            np.array([combo_from_subtract]), total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        part_bws_list_eha, node_counts_list_eha, total_counts_list_eha = prepare_model_inputs(
            np.array([combo_from_eha]), total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        
        bw_subtract = predict_with_model(
            model, part_bws_list_subtract, node_counts_list_subtract, total_counts_list_subtract, device, artifact_dir
        )[0]
        bw_eha = predict_with_model(
            model, part_bws_list_eha, node_counts_list_eha, total_counts_list_eha, device, artifact_dir
        )[0]
        
        # else分支的特殊处理：如果使用了节点插入，subtract结果使用真实数据评估
        if success_insert:
            bw_subtract, _, _ = calculate_bandwidth_values(
                combo_from_subtract, total_gpu, gpu_bw_dict_list, switch_config, data_path
            )
    
    # 选择带宽最大的组合
    bw_list = [bw_subtract, bw_eha]
    combo_list = [combo_from_subtract, combo_from_eha]
    max_idx = int(np.argmax(bw_list))
    if not if_real_data:
        logger.info(f"预测的最优带宽为：{np.max(bw_list)},选择的算法是：{np.argmax(bw_list)}")
    return combo_list[max_idx]


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
    global_mode: bool = True,
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

    # 尝试节点插入优化
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

