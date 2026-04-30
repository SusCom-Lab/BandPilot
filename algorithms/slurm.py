"""Simulated Slurm topology-aware allocation and k-clique sampling."""
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
    """Slurm-like best-fit strategy (improved: topology-aware across nodes).

    Compactness is defined as "using as few nodes as possible"; under the same node count,
    choose the GPU combo with best connectivity via topo_matrix/connectivity_score.
    """
    # Initialize result mask
    final_mask = np.zeros(total_gpu, dtype=int)
    if gpu_need <= 0:
        return final_mask

    # Normalize available GPU indices
    valid_avail = sorted({gpu for gpu in avail_gpu_indices if 0 <= gpu < total_gpu})
    if gpu_need > len(valid_avail):
        logger.error("Requested GPU count (%d) exceeds available (%d)", gpu_need, len(valid_avail))
        return final_mask

    # Group available GPUs by node
    gpus_on_nodes: Dict[int, List[int]] = {}
    for gpu_idx in valid_avail:
        node_idx = gpu_to_node_map.get(gpu_idx, -1)
        gpus_on_nodes.setdefault(node_idx, []).append(gpu_idx)

    nodes = sorted(gpus_on_nodes.keys())

    # Final selected GPU combo
    best_selection: List[int] = []
    best_score: float = -1.0

    # Search node_count from small to large to enforce compactness
    for node_count in range(1, len(nodes) + 1):
        best_for_this_node_count: List[int] = []
        best_score_for_this_node_count: float = -1.0
        found_candidate = False

        # Enumerate all node_count-sized node subsets
        for node_subset in itertools.combinations(nodes, node_count):
            # Count available GPUs on this node subset; skip if insufficient
            union_gpus: List[int] = []
            for n in node_subset:
                union_gpus.extend(gpus_on_nodes[n])
            if len(union_gpus) < gpu_need:
                continue

            # Greedily select gpu_need GPUs from the union to maximize connectivity_score.
            union_gpus = sorted(union_gpus)
            selected: List[int] = []
            remaining = union_gpus.copy()

            while len(selected) < gpu_need and remaining:
                best_gpu = None
                best_partial_score = -1.0

                for g in remaining:
                    combo = selected + [g]
                    score = calculate_connectivity_score(combo, topo_matrix)
                    if score > best_partial_score:
                        best_partial_score = score
                        best_gpu = g

                if best_gpu is None:
                    # Should not happen in theory; defensive exit
                    break

                selected.append(best_gpu)
                remaining.remove(best_gpu)

            if len(selected) != gpu_need:
                # Node subset cannot yield a full combo; skip
                continue

            # Recompute final score for complete combo
            final_score = calculate_connectivity_score(selected, topo_matrix)
            found_candidate = True

            # Keep best connectivity_score under same node_count
            if final_score > best_score_for_this_node_count:
                best_score_for_this_node_count = final_score
                best_for_this_node_count = selected

        # If any feasible plan exists for this node_count, since we increase node_count,
        # this is the most compact node usage; keep best connectivity among them.
        if found_candidate and best_for_this_node_count:
            best_selection = best_for_this_node_count
            best_score = best_score_for_this_node_count
            break

    if not best_selection:
        # Should not happen in theory (gpu_need <= len(valid_avail)); defensive log
        logger.error(
            "slurm_best_fit_algo found no feasible allocation: gpu_need=%d, avail=%d",
            gpu_need,
            len(valid_avail),
        )
        return final_mask

    final_mask[best_selection] = 1
    logger.debug(
        "slurm_best_fit_algo selected GPUs: %s, node_count=%d, connectivity_score=%.4f",
        best_selection,
        len({gpu_to_node_map.get(i, -1) for i in best_selection}),
        best_score,
    )
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
    """Randomized sampling search guided by model-predicted bandwidth weights."""
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

