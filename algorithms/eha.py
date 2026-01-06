"""Equilibrium-driven Heuristic Algorithm (EHA)."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values, prepare_model_inputs
from training.evaluator import predict_with_model
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.cluster_state import ClusterStateManager


def _select_data_path_for_mode(
    use_real_data: bool,
    training_data_path: str,
    evaluation_data_path: Optional[str],
) -> str:
    """Return data path based on mode."""
    if use_real_data and evaluation_data_path:
        return evaluation_data_path
    return training_data_path


def _predict_config_bandwidth(
    config: np.ndarray,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    evaluation_data_path: Optional[str] = None,
) -> float:
    """Compute bandwidth for a single config under current environment (cluster_manager / real / model)."""
    if cluster_manager:
        return float(cluster_manager.predict_with_contention(config))
    if if_real_data:
        real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
        bw_value, _, _ = calculate_bandwidth_values(
            config, total_gpu, gpu_bw_dict_list, switch_config, real_path
        )
        return float(bw_value)

    part_bws, node_counts, total_counts = prepare_model_inputs(
        np.array([config]), total_gpu, gpu_bw_dict_list, switch_config, training_data_path
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
    global_mode_all: bool,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    evaluation_data_path: Optional[str] = None,
) -> float:
    """Score a candidate config in normal or global mode."""
    current_bw = _predict_config_bandwidth(
        config,
        model,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
        device,
        artifact_dir,
        if_real_data,
        cluster_manager,
        evaluation_data_path,
    )

    if not global_mode or avail_gpu is None or len(avail_gpu) == 0:
        return current_bw

    selected_gpus = set(np.where(config == 1)[0])
    remaining_gpus = [gpu for gpu in avail_gpu if gpu not in selected_gpus]
    if not remaining_gpus:
        remaining_bw = 0.0
    else:
        remaining_config = np.zeros(num_dimensions, dtype=int)
        remaining_config[remaining_gpus] = 1
        remaining_bw = _predict_config_bandwidth(
            remaining_config,
            model,
            total_gpu,
            gpu_bw_dict_list,
            switch_config,
            training_data_path,
            device,
            artifact_dir,
            if_real_data,
            cluster_manager,
            evaluation_data_path,
        )
    total_bw = current_bw + remaining_bw
    if global_mode_all and cluster_manager:
        total_bw += cluster_manager.get_total_active_bandwidth()
    return total_bw


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
    training_data_path: str,
    device: torch.device,
    artifact_dir: Path,
    evaluation_data_path: Optional[str],
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool,
    avail_gpu: Optional[Sequence[int]],
    allow_global_mode: bool,
    global_mode_all: bool,
) -> Optional[np.ndarray]:
    """Run bandwidth-aware tree_search on a node and return best sub-config of length target_gpu_count."""
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

    from algorithms.search import greedy_recursive_search

    subset_config = greedy_recursive_search(
        current_combo=start_combo,
        gpu_need=target_gpu_count,
        model=model,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
        device=device,
        artifact_dir=artifact_dir,
        if_real_data=if_real_data,
        cluster_manager=cluster_manager,
        global_mode=global_mode and allow_global_mode,
        avail_gpu=avail_gpu if (global_mode and allow_global_mode) else None,
        global_mode_all=global_mode_all if (global_mode and allow_global_mode) else False,
        evaluation_data_path=evaluation_data_path,
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
    training_data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool = False,
    max_candidates: int = 200,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
    global_mode_all: bool = False,
    evaluation_data_path: Optional[str] = None,
) -> np.ndarray | None:
    """
    Equilibrium-driven Heuristic Algorithm (EHA).

    Uses communication characteristics observed from real data (locality, node count, balance)
    and a deterministic step-wise construction to quickly generate high-quality GPU combinations.

    Two phases:
    1. Single-node optimum search (highest priority)
       - If any node alone can satisfy gpu_need, choose the best config within that node.
       - Single-node configs typically have best locality and communication performance.

    2. Cross-node construction
       - When no single node can satisfy the demand, build configs across multiple nodes.
       - Two allocation strategies:
         a) Strategy 1: remaining-resource balance – greedy: always give GPUs to node with most remaining GPUs.
         b) Strategy 2: balanced counts – evenly distribute with ±1/±2 variants.

    Args:
        num_dimensions: Total GPUs (i.e., total_gpu).
        avail_gpu: List of available GPU indices.
        model: PyTorch model for bandwidth prediction.
        gpu_need: Required GPU count.
        total_gpu: Total GPUs in cluster.
        gpu_bw_dict_list: Per-node bandwidth dictionaries.
        switch_config: Switch configuration.
        training_data_path: Data path for model prediction.
        device: PyTorch device (CPU/CUDA).
        artifact_dir: Directory for model and scaler artifacts.
        if_real_data: Use real bandwidth (True) vs model prediction (False).
        max_candidates: Max candidate configs to control search size.
        cluster_manager: ClusterStateManager for multi-tenant awareness, if provided.
        global_mode: If True, score as current + remaining GPUs.
        global_mode_all: If True with global_mode, add history + remaining to global score.

    Returns:
        Best GPU combo (0/1 vector), or None if infeasible.
    """
    model_data_path = training_data_path
    real_data_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
    # ==================== Preprocessing: group available GPUs by node ====================
    # Group GPUs by physical node; assume 8 GPUs per node (node_id = index // 8)
    node_map: Dict[int, List[int]] = {}
    for gpu_idx in avail_gpu:
        node_id = gpu_idx // 8
        node_map.setdefault(node_id, []).append(gpu_idx)

    # ==================== Phase 1: single-node optimum search (highest priority) ====================
    # Find nodes that can individually satisfy gpu_need (best locality, no cross-node cost).
    candidate_nodes = [node_id for node_id, gpus in node_map.items() if len(gpus) >= gpu_need]
    candidate_configs: List[np.ndarray] = []

    if candidate_nodes:
        # If any node can satisfy demand, generate config per candidate node
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
                training_data_path=model_data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                global_mode=global_mode,
                avail_gpu=avail_gpu,
                allow_global_mode=True,
                global_mode_all=global_mode_all,
                evaluation_data_path=real_data_path,
            )
            if config is not None:
                candidate_configs.append(config)
        
        # If there are multiple candidate nodes, use model or real data to evaluate and select the best configuration
        if len(candidate_configs) == 1:
            return candidate_configs[0]
        
        # Batch evaluate all single-node candidate configurations
        best_idx = -1
        best_bw = -1.0
        for idx, config in enumerate(candidate_configs):
            bw = _score_config_bandwidth(
                config,
                avail_gpu=avail_gpu,
                num_dimensions=num_dimensions,
                global_mode=global_mode,
                global_mode_all=global_mode_all,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=model_data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                evaluation_data_path=real_data_path,
            )
            if bw > best_bw:
                best_bw = bw
                best_idx = idx
        return candidate_configs[best_idx] if best_idx >= 0 else None

    # ==================== Phase 2: build cross-node solution ====================
    # When no single node can satisfy demand, combine multiple nodes.
    # Step 1: sort nodes by available GPU count descending (favor larger nodes).
    sorted_nodes = sorted(node_map.items(), key=lambda item: len(item[1]), reverse=True)
    
    # Step 2: determine minimum number of nodes (k) to satisfy gpu_need
    # Greedy: accumulate from nodes with most GPUs until sum >= gpu_need.
    k = 0
    gpu_sum = 0
    for _, gpus in sorted_nodes:
        gpu_sum += len(gpus)
        k += 1
        if gpu_sum >= gpu_need:
            break

    # Boundary check: if total GPUs still cannot meet demand, return None
    if gpu_sum < gpu_need:
        return None

    # Step 3: construct best k-node combinations; use a set to avoid duplicates.
    seen_configs = set()
    
    # Iterate through all possible combinations of k nodes
    for node_group in itertools.combinations(sorted_nodes, k):
        # Get available GPU counts per node in this group
        group_counts = [len(gpu_list) for _, gpu_list in node_group]
        
        # Quick check: skip if total GPUs in group insufficient
        if sum(group_counts) < gpu_need:
            continue

        # ========== Strategy 1: remaining-resource balance (greedy) ==========
        # For small candidate counts (<= max_candidates), greedily allocate
        # one GPU at a time to the node with most remaining capacity.
        if len(candidate_configs) <= max_candidates:
            # Initialize allocation: 0 GPUs per node
            alloc_remain_balance = [0] * k
            gpus_to_distribute = gpu_need
            # Temporary remaining capacity per node
            temp_avail = list(group_counts)
            
            # Greedy: allocate 1 GPU each time to node with most remaining GPUs
            for _ in range(gpus_to_distribute):
                best_node_idx = -1
                max_avail = -1
                # Find node with maximum remaining capacity
                for i in range(k):
                    if temp_avail[i] > 0:
                        if temp_avail[i] > max_avail:
                            max_avail = temp_avail[i]
                            best_node_idx = i
                
                # If a node is found, allocate 1 GPU to it
                if best_node_idx != -1:
                    alloc_remain_balance[best_node_idx] += 1
                    temp_avail[best_node_idx] -= 1
                else:
                    # No node can accept more; allocation fails
                    break
            
            # Build GPU config according to allocation plan
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
                    training_data_path=model_data_path,
                    device=device,
                    artifact_dir=artifact_dir,
                    if_real_data=if_real_data,
                    cluster_manager=cluster_manager,
                    global_mode=global_mode,
                    avail_gpu=None,
                    allow_global_mode=False,
                    global_mode_all=False,
                    evaluation_data_path=real_data_path,
                )
                if subset_config is None:
                    is_possible_remain = False
                    break
                config_remain = np.maximum(config_remain, subset_config)
            
            # If allocation succeeds, add to candidates
            if is_possible_remain:
                config_tuple = tuple(config_remain)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    candidate_configs.append(config_remain)

        # ========== Strategy 2: balanced counts (even + perturbation variants) ==========
        # Try even distribution and ±1/±2/±3 variants.
        # Step 2.1: compute base even allocation per node, e.g. gpu_need=8, k=3 -> base_alloc=[3,3,2]
        base_alloc = [gpu_need // k] * k
        for i in range(gpu_need % k):
            base_alloc[i] += 1
        
        # Step 2.2: generate all allocation variants: base, ±1, ±2, ±3 etc.
        allocation_variants: set[Tuple[int, ...]] = set()

        def _register_variant(variant_alloc: Sequence[int]) -> None:
            for perm in set(itertools.permutations(variant_alloc)):
                allocation_variants.add(tuple(perm))

        # 2.2.1: add all permutations of base allocation
        _register_variant(base_alloc)

        # 2.2.2/2.2.3: generate ±1/±2/±3 uneven variants
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
        
        # Step 2.3: iterate variants, check feasibility, and build configs
        for alloc_variant in allocation_variants:
            # Check feasibility: each node must have enough GPUs
            is_possible = all(group_counts[i] >= alloc_variant[i] for i in range(k))
            
            if is_possible:
                # Build GPU config for this allocation
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
                    training_data_path=model_data_path,
                        device=device,
                        artifact_dir=artifact_dir,
                        if_real_data=if_real_data,
                        cluster_manager=cluster_manager,
                        global_mode=global_mode,
                        avail_gpu=None,
                        allow_global_mode=False,
                        global_mode_all=False,
                    evaluation_data_path=real_data_path,
                    )
                    if subset_config is None:
                        is_possible = False
                        break
                    config = np.maximum(config, subset_config)
                
                if not is_possible:
                    continue
                # De-duplicate and append candidate
                config_tuple = tuple(config)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    candidate_configs.append(config)
                    # Reached max candidate count; early exit
                    if len(candidate_configs) >= max_candidates:
                        break
        
        # If max candidates reached, break outer loop
        if len(candidate_configs) >= max_candidates:
            break

    # ==================== Final evaluation and selection ====================
    # If no candidates, return None
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
            global_mode_all=global_mode_all,
            model=model,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=model_data_path,
            device=device,
            artifact_dir=artifact_dir,
            if_real_data=if_real_data,
            cluster_manager=cluster_manager,
            evaluation_data_path=real_data_path,
        )
        if bw > best_bw:
            best_bw = bw
            best_idx = idx
    return candidate_configs[best_idx] if best_idx >= 0 else None


# eha, old version
def eha_search_old(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    model,
    gpu_need: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device: torch.device,
    artifact_dir: Path,
    if_real_data: bool = False,
    max_candidates: int = 50,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
    verbose: bool = False,
    global_mode_all: bool = False,
    evaluation_data_path: Optional[str] = None,
) -> np.ndarray | None:
    """
    Equilibrium-driven Heuristic Algorithm (EHA) - old version.

    Uses deterministic construction based on locality/node-count/balance to generate
    high-quality GPU combos much faster than genetic algorithms.

    Note: old version uses simple "take first n GPUs" strategy and does not use tree_search.
    Differences from eha_search:
    - Single-node phase: directly take first gpu_need GPUs on a node.
    - Cross-node phase: directly take first n GPUs, no tree_search.

    Two phases:
    1. Single-node optimum search (highest priority)
       - If any node can independently satisfy gpu_need, pick first gpu_need GPUs there.
    2. Cross-node construction
       - When no single node can satisfy demand, combine multiple nodes using:
         a) Strategy 1: remaining-resource balance – greedy, give GPUs to node with most remaining.
         b) Strategy 2: balanced counts – even distribution with all permutations.
    """
    # ==================== Preprocessing: group available GPUs by node ====================
    # Group GPUs by physical node; assume 8 GPUs per node (node_id = index // 8)
    node_map: Dict[int, List[int]] = {}
    for gpu_idx in avail_gpu:
        node_id = gpu_idx // 8
        node_map.setdefault(node_id, []).append(gpu_idx)

    if verbose:
        print("Available GPUs grouped by node:", {k: len(v) for k, v in node_map.items()})
    candidate_nodes = [node_id for node_id, gpus in node_map.items() if len(gpus) >= gpu_need]
    candidate_configs: List[np.ndarray] = []

    if candidate_nodes:
        if verbose:
            print(f"Phase 1: searching best config among {len(candidate_nodes)} candidate single nodes...")
        
        # Old-version strategy: build a test config per candidate node (take first n GPUs)
        for node_id in candidate_nodes:
            selected_gpus = node_map[node_id][:gpu_need]
            config = np.zeros(num_dimensions, dtype=int)
            config[selected_gpus] = 1
            candidate_configs.append(config)
        
        # If multiple candidate nodes, evaluate and pick best config
        if len(candidate_configs) == 1:
            return candidate_configs[0]
        
        # Batch-evaluate all single-node candidates
        best_idx = -1
        best_bw = -1.0
        for idx, config in enumerate(candidate_configs):
            bw = _score_config_bandwidth(
                config,
                avail_gpu=avail_gpu,
                num_dimensions=num_dimensions,
                global_mode=global_mode,
                global_mode_all=global_mode_all,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=training_data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                evaluation_data_path=evaluation_data_path,
            )
            if bw > best_bw:
                best_bw = bw
                best_idx = idx
        
        if verbose:
            print(f"Phase 1 done. Best single-node predicted bandwidth: {best_bw:.2f}")
        
        return candidate_configs[best_idx] if best_idx >= 0 else None

    # ==================== Phase 2: cross-node construction ====================
    # When no single node can satisfy demand, combine multiple nodes.
    if verbose:
        print("Phase 2: no single node suffices, constructing cross-node optimum...")
    
    # Step 1: sort nodes by available GPUs descending (favor larger nodes)
    sorted_nodes = sorted(node_map.items(), key=lambda item: len(item[1]), reverse=True)
    
    # Step 2: determine minimum number of nodes (k) to satisfy gpu_need.
    # Greedy: accumulate from nodes with most GPUs until sum >= gpu_need.
    k = 0
    gpu_sum = 0
    for _, gpus in sorted_nodes:
        gpu_sum += len(gpus)
        k += 1
        if gpu_sum >= gpu_need:
            break

    # Boundary check: if total GPUs still cannot meet demand, return None.
    if gpu_sum < gpu_need:
        if verbose:
            print("Error: total available GPUs still cannot satisfy demand!")
        return None

    if verbose:
        print(f"Minimum {k} nodes required to satisfy demand of {gpu_need} GPUs.")

    # Step 3: construct the best k-node combinations
    # Use a set to prevent duplicate configurations
    final_configs: List[np.ndarray] = []
    seen_configs = set()

    # Iterate through all possible combinations of k nodes
    for node_group_tuple in itertools.combinations(sorted_nodes, k):
        group_avail_gpus = [len(gpus) for _, gpus in node_group_tuple]
        
        # Quick check: skip if total GPUs in this group are insufficient
        if sum(group_avail_gpus) < gpu_need:
            continue
        
        # ========== Strategy 1: remaining-resource balance (greedy, old version) ==========
        # For few candidates (<= 5), greedily give GPUs to the node with most remaining capacity.
        if len(final_configs) <= 5:
            # Initialize allocation: 0 GPUs per node
            alloc_remain_balance = [0] * k
            gpus_to_distribute = gpu_need
            # Temporary remaining capacity per node
            temp_avail = list(group_avail_gpus)
            
            # Greedy: allocate 1 GPU at a time to the node with most remaining GPUs
            for _ in range(gpus_to_distribute):
                best_node_idx = -1
                max_avail = -1
                # Find node with maximum remaining capacity
                for i in range(k):
                    if temp_avail[i] > 0:
                        if temp_avail[i] > max_avail:
                            max_avail = temp_avail[i]
                            best_node_idx = i
                
                # If a node is found, allocate 1 GPU to it
                if best_node_idx != -1:
                    alloc_remain_balance[best_node_idx] += 1
                    temp_avail[best_node_idx] -= 1
                else:
                    # No node can accept more; allocation fails
                    break
            
            # Build GPU config according to allocation (old version: take first n GPUs)
            config_remain = np.zeros(num_dimensions, dtype=int)
            is_possible_remain = True
            for i in range(k):
                _, gpu_list = node_group_tuple[i]
                num_to_take = alloc_remain_balance[i]
                if len(gpu_list) >= num_to_take:
                    # Old-version strategy: directly take first num_to_take GPUs
                    selected_on_node = gpu_list[:num_to_take]
                    config_remain[selected_on_node] = 1
                else:
                    is_possible_remain = False
                    break
            
            # If allocation succeeds, add to candidates
            if is_possible_remain:
                config_tuple = tuple(config_remain)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    final_configs.append(config_remain)

        # ========== Strategy 2: balanced counts (even distribution + all permutations, old version) ==========
        # Step 2.1: compute base even allocation per node.
        # base_alloc is GPUs per node, e.g. gpu_need=8, k=3 -> base_alloc=[3,3,2]
        base_alloc = [gpu_need // k] * k
        for i in range(gpu_need % k):
            base_alloc[i] += 1
        
        # Step 2.2: generate all unique permutations of base_alloc (set deduplicates permutations like [3,3,2])
        unique_alloc_permutations = set(itertools.permutations(base_alloc))
        
        # Step 2.3: iterate each unique allocation permutation
        for alloc_permutation in unique_alloc_permutations:
            # Check feasibility: each node must have enough GPUs
            is_possible = all(group_avail_gpus[i] >= alloc_permutation[i] for i in range(k))
            
            if is_possible:
                # Build GPU config according to allocation (old version: take first n GPUs)
                config = np.zeros(num_dimensions, dtype=int)
                for i in range(k):
                    # Get node information
                    _, gpu_list = node_group_tuple[i]
                    # Get the number of GPUs to take from the current permutation
                    num_to_take = alloc_permutation[i]
                    # Old-version strategy: take first num_to_take GPUs on the node
                    selected_on_node = gpu_list[:num_to_take]
                    # Update the configuration vector
                    config[selected_on_node] = 1
                
                # Check and add configuration to prevent duplicates
                config_tuple = tuple(config)
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    final_configs.append(config)
                    # If the maximum candidate count is reached, exit early
                    if len(final_configs) >= max_candidates:
                        break
        
        # If the maximum candidate count is reached, exit the outer loop
        if len(final_configs) >= max_candidates:
            break

    # ==================== Fallback strategy: if no candidate configs ====================
    if not final_configs:
        if verbose:
            print("Fallback strategy: greedy pick from nodes with most remaining GPUs.")
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
        print(f"Generated {len(final_configs)} high-quality candidate configs, running final model selection.")
    
    # ==================== Final evaluation and selection ====================
    # Use unified evaluation (supports global_mode, cluster_manager, etc.)
    best_idx = -1
    best_bw = -1.0
    for idx, config in enumerate(final_configs):
        bw = _score_config_bandwidth(
            config,
            avail_gpu=avail_gpu,
            num_dimensions=num_dimensions,
            global_mode=global_mode,
            global_mode_all=global_mode_all,
            model=model,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
            device=device,
            artifact_dir=artifact_dir,
            if_real_data=if_real_data,
            cluster_manager=cluster_manager,
            evaluation_data_path=evaluation_data_path,
        )
        if bw > best_bw:
            best_bw = bw
            best_idx = idx
    
    return final_configs[best_idx] if best_idx >= 0 else None


