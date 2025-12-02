"""评估指标与上界估计。"""
from __future__ import annotations

import ast
import logging
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

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
    if len(gpu_bw_dict_list) < num_machines:
        raise ValueError(
            "gpu_bw_dict_list 长度不足，"
            f"当前需要 {num_machines} 个节点字典，实际 {len(gpu_bw_dict_list)}"
        )

    # 构建可用 GPU 的节点视角
    avail_gpu_sorted = sorted(int(gpu) for gpu in avail_gpu)
    avail_local_indices: List[List[int]] = [[] for _ in range(num_machines)]
    for gpu_idx in avail_gpu_sorted:
        node_idx = gpu_idx // 8
        local_idx = gpu_idx % 8
        if 0 <= node_idx < num_machines:
            avail_local_indices[node_idx].append(local_idx)
    avail_capacity = [len(local_list) for local_list in avail_local_indices]
    if k > sum(avail_capacity):
        logger.warning(
            "find_max_bw_for_k_gpus: k=%s 仍大于可用 GPU 数 %s，返回 0 带宽。",
            k,
            sum(avail_capacity),
        )
        return 0.0, []

    lookup = BandwidthLookupCache.ensure_loaded(data_path)
    cross_lookup: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
    for key, records in lookup.items():
        max_bw = max(float(bw) for _, bw in records)
        cross_lookup[key] = max_bw

    # 预计算每个节点在不同 GPU 数下的最佳节点内组合
    best_intra_configs: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]] = {}
    for node_idx in range(num_machines):
        local_avail = avail_local_indices[node_idx]
        if not local_avail:
            continue

        # 单 GPU 情况不会形成瓶颈，带宽视为正无穷
        single_mask = [0] * 8
        single_mask[local_avail[0]] = 1
        best_intra_configs[(node_idx, 1)] = (float("inf"), tuple(single_mask))

        node_dict = gpu_bw_dict_list[node_idx]
        for gpu_cnt in range(2, len(local_avail) + 1):
            best_bw = -1.0
            best_mask: Optional[Tuple[int, ...]] = None
            for combo in itertools.combinations(local_avail, gpu_cnt):
                mask = [0] * 8
                for local_idx in combo:
                    mask[local_idx] = 1
                mask_tuple = tuple(mask)
                bw_value = float(node_dict.get(mask_tuple, 0.0))
                if bw_value > best_bw:
                    best_bw = bw_value
                    best_mask = mask_tuple
            if best_mask is not None:
                best_intra_configs[(node_idx, gpu_cnt)] = (max(best_bw, 0.0), best_mask)

    def build_config(distribution: Tuple[int, ...]) -> List[int]:
        config = [0] * total_gpu
        for node_idx, gpu_cnt in enumerate(distribution):
            if gpu_cnt == 0:
                continue
            _, mask = best_intra_configs[(node_idx, gpu_cnt)]
            base = node_idx * 8
            for local_idx, flag in enumerate(mask):
                if flag:
                    config[base + local_idx] = 1
        return config

    suffix_capacity = [0] * (num_machines + 1)
    for idx in range(num_machines - 1, -1, -1):
        suffix_capacity[idx] = suffix_capacity[idx + 1] + avail_capacity[idx]

    def generate_distributions(machine_idx: int, remaining: int, current: List[int]):
        if remaining < 0:
            return
        if machine_idx == num_machines:
            if remaining == 0:
                yield tuple(current)
            return
        if remaining > suffix_capacity[machine_idx]:
            return
        max_take = min(8, avail_capacity[machine_idx], remaining)
        for cnt in range(max_take, -1, -1):
            if cnt > 0 and (machine_idx, cnt) not in best_intra_configs:
                continue
            current.append(cnt)
            yield from generate_distributions(machine_idx + 1, remaining - cnt, current)
            current.pop()

    best_bandwidth = 0.0
    best_config: List[int] = []

    for distribution in generate_distributions(0, k, []):
        active_counts = [cnt for cnt in distribution if cnt > 0]
        if not active_counts:
            continue
        if len(active_counts) == 1:
            cross_bw = float("inf")
        else:
            key = (k, len(active_counts), tuple(sorted(active_counts)))
            cross_bw = cross_lookup.get(key, 0.0)
            if cross_bw <= 0:
                continue

        intra_bw = float("inf")
        feasible = True
        for node_idx, gpu_cnt in enumerate(distribution):
            if gpu_cnt == 0:
                continue
            intra_info = best_intra_configs.get((node_idx, gpu_cnt))
            if intra_info is None:
                feasible = False
                break
            if gpu_cnt >= 2:
                intra_bw = min(intra_bw, intra_info[0])
        if not feasible:
            continue

        candidate_bw = min(cross_bw, intra_bw)
        if candidate_bw > best_bandwidth:
            best_bandwidth = candidate_bw
            best_config = build_config(distribution)

    if best_bandwidth <= 0:
        logger.warning(
            "find_max_bw_for_k_gpus: 未在当前可用 GPU 条件下找到可行的 k=%s 配置", k
        )
        return 0.0, []

    return best_bandwidth, best_config

