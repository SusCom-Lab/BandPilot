"""评估指标与上界估计。"""
from __future__ import annotations

import ast
import logging
from typing import List, Sequence, Tuple

import numpy as np

from core.bandwidth import BandwidthLookupCache

logger = logging.getLogger(__name__)


def find_max_bw_for_k_gpus(
    k: int,
    gpu_bw_dict_list,
    total_gpu: int,
    switch_config,
    avail_gpu: Sequence[int],
    data_path: str,
) -> Tuple[float, List[int]]:
    """依据查表结果，估计可用GPU约束下的最大带宽。
    
    该函数会在查找过程中显示进度信息，包括：
    - 查找的候选配置数量
    - 已检查的配置数量
    - 当前找到的最大带宽值
    """
    # Special case: total GPU 必须满足拓扑分块（8 的倍数），否则无需继续查找
    if total_gpu % 8 != 0:
        raise ValueError("total_gpu必须是8的倍数")
    # `avail_gpu` 可能是 numpy 数组，改用显式长度判断避免 bool 语义错误
    if len(avail_gpu) == 0:
        logger.warning("find_max_bw_for_k_gpus: avail_gpu 为空，返回 0 带宽。")
        return 0.0, []
    # 如果请求的 GPU 数超出可用范围，同样直接返回
    if not (1 <= k <= len(avail_gpu)):
        logger.warning(
            "find_max_bw_for_k_gpus: k=%s 超出可用 GPU 数 %s，返回 0 带宽。",
            k,
            len(avail_gpu),
        )
        return 0.0, []

    num_machines = total_gpu // 8
    avail_node_counts = [0] * num_machines
    for gpu_idx in avail_gpu:
        node_idx = gpu_idx // 8
        if 0 <= node_idx < num_machines:
            avail_node_counts[node_idx] += 1
    avail_distribution = sorted([cnt for cnt in avail_node_counts if cnt > 0], reverse=True)

    lookup = BandwidthLookupCache.ensure_loaded(data_path)

    # 统计进度信息
    matching_keys = [key for key in lookup.keys() if key[0] == k]
    total_candidates = sum(len(lookup[key]) for key in matching_keys)
    checked_count = 0
    feasible_count = 0
    
    if total_candidates > 0:
        logger.info(f"find_max_bw_for_k_gpus: 开始查找 k={k} 的最大带宽，候选配置总数: {total_candidates}")
    else:
        logger.warning(f"find_max_bw_for_k_gpus: 未找到 k={k} 的候选配置")
        return 0.0, []

    max_bandwidth = 0.0
    best_distribution: List[int] = []

    for key, records in lookup.items():
        total_active, _, distribution = key
        # 查找表中存储的活跃 GPU 数与需求不一致时，跳过当前 key
        if total_active != k:
            continue
        distribution_list = list(distribution)
        distribution_list.sort(reverse=True)

        # 若候选模式使用的节点数多于当前可用节点，同样无法满足
        if len(distribution_list) > len(avail_distribution):
            continue
        feasible = True
        for idx in range(len(distribution_list)):
            if avail_distribution[idx] < distribution_list[idx]:
                feasible = False
                break
        # 节点资源不足，继续下一个候选
        if not feasible:
            continue

        for mapping_str, bandwidth in records:
            checked_count += 1
            # 每检查 100 个候选配置显示一次进度（避免除零错误）
            if total_candidates > 0 and checked_count % 100 == 0:
                progress_pct = checked_count * 100 // total_candidates
                logger.info(
                    f"find_max_bw_for_k_gpus: 进度 {checked_count}/{total_candidates} "
                    f"({progress_pct}%), "
                    f"当前最大带宽: {max_bandwidth:.2f}, 可行配置数: {feasible_count}"
                )
            
            bw_value = float(bandwidth)
            # 所有通过可行性检查的配置都计入可行配置数
            feasible_count += 1
            if bw_value > max_bandwidth:
                max_bandwidth = bw_value
                best_distribution = distribution_list
                # 找到更大的带宽值时也输出提示
                logger.debug(
                    f"find_max_bw_for_k_gpus: 找到更大带宽 {bw_value:.2f}, "
                    f"节点分布: {best_distribution}"
                )

    # 查找完成，输出最终结果
    if max_bandwidth > 0:
        logger.info(
            f"find_max_bw_for_k_gpus: 查找完成，共检查 {checked_count} 个候选配置，"
            f"找到 {feasible_count} 个可行配置，最大带宽: {max_bandwidth:.4f}"
        )
    else:
        logger.warning(
            f"find_max_bw_for_k_gpus: 查找完成，共检查 {checked_count} 个候选配置，"
            f"未找到可通过可用GPU实现的带宽配置"
        )

    return max_bandwidth, best_distribution

