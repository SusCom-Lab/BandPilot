"""模拟Slurm拓扑感知分配算法及k-clique采样。"""
from __future__ import annotations

import itertools
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values, prepare_model_inputs
from core.topology import calculate_connectivity_score
from training.evaluator import predict_with_model

logger = logging.getLogger(__name__)


def slurm_best_fit_algo(
    total_gpu: int,
    avail_gpu_indices: Sequence[int],
    gpu_need: int,
    topo_matrix: pd.DataFrame,
    gpu_to_node_map: Dict[int, int],
) -> np.ndarray:
    """模仿Slurm Best-Fit策略。"""
    final_mask = np.zeros(total_gpu, dtype=int)
    if gpu_need <= 0:
        return final_mask

    valid_avail = sorted([gpu for gpu in avail_gpu_indices if 0 <= gpu < total_gpu])
    if gpu_need > len(valid_avail):
        logger.error("需要的GPU数量超过可用数量")
        return final_mask

    gpus_on_nodes: Dict[int, List[int]] = {}
    for gpu_idx in valid_avail:
        node_idx = gpu_to_node_map.get(gpu_idx, -1)
        gpus_on_nodes.setdefault(node_idx, []).append(gpu_idx)

    candidate_nodes = [node for node, gpus in gpus_on_nodes.items() if len(gpus) >= gpu_need]
    if candidate_nodes:
        best_selection: List[int] = []
        best_score = -1.0
        for node_idx in candidate_nodes:
            available = gpus_on_nodes[node_idx]
            for combo in itertools.combinations(available, gpu_need):
                score = calculate_connectivity_score(combo, topo_matrix)
                if score > best_score:
                    best_score = score
                    best_selection = list(combo)
        final_mask[best_selection] = 1
        return final_mask

    selected: List[int] = []
    sorted_nodes = sorted(gpus_on_nodes.items(), key=lambda item: len(item[1]), reverse=True)
    for _, gpus in sorted_nodes:
        needed = gpu_need - len(selected)
        take = gpus[:needed]
        selected.extend(take)
        if len(selected) >= gpu_need:
            break
    final_mask[selected[:gpu_need]] = 1
    return final_mask


def k_clique_bandwidth_sampling_search(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    gpu_need: int,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    device: torch.device,
    artifact_dir: Path,
    sample_factor: int = 3,
) -> np.ndarray:
    """基于模型预测权重的随机采样搜索。"""
    if gpu_need <= 0 or len(avail_gpu) < gpu_need:
        return np.zeros(num_dimensions, dtype=int)

    pairwise = list(itertools.combinations(avail_gpu, 2))
    if not pairwise:
        final_config = np.zeros(num_dimensions, dtype=int)
        final_config[avail_gpu] = 1
        return final_config

    pair_configs = np.zeros((len(pairwise), num_dimensions), dtype=int)
    for idx, pair in enumerate(pairwise):
        pair_configs[idx, list(pair)] = 1

    part_bws, node_counts, total_counts = prepare_model_inputs(
        pair_configs, total_gpu, gpu_bw_dict_list, switch_config, data_path
    )
    pair_scores = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)

    weighted_degrees = {gpu: 0.0 for gpu in avail_gpu}
    for score, pair in zip(pair_scores, pairwise):
        weighted_degrees[pair[0]] += score
        weighted_degrees[pair[1]] += score

    total_weight = sum(weighted_degrees.values())
    probabilities = (
        [weighted_degrees[gpu] / total_weight for gpu in avail_gpu] if total_weight > 0 else None
    )

    pool_size = min(len(avail_gpu), max(gpu_need, gpu_need * sample_factor))
    try:
        candidate_pool = np.random.choice(avail_gpu, pool_size, replace=False, p=probabilities)
    except ValueError:
        candidate_pool = random.sample(avail_gpu, pool_size)

    combos = list(itertools.combinations(candidate_pool, gpu_need))
    if not combos:
        final_config = np.zeros(num_dimensions, dtype=int)
        final_config[candidate_pool[:gpu_need]] = 1
        return final_config

    combo_configs = np.zeros((len(combos), num_dimensions), dtype=int)
    for idx, combo in enumerate(combos):
        combo_configs[idx, list(combo)] = 1

    part_bws, node_counts, total_counts = prepare_model_inputs(
        combo_configs, total_gpu, gpu_bw_dict_list, switch_config, data_path
    )
    scores = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)

    best_idx = int(np.argmax(scores))
    final_mask = np.zeros(num_dimensions, dtype=int)
    final_mask[list(combos[best_idx])] = 1
    return final_mask

