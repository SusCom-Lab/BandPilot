"""Search and combination-generation algorithms (multi-tenant aware)."""
from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values, prepare_model_inputs
from core.gpu_config import generate_data_minmax_restricted
from training.evaluator import predict_with_model
# Imported only for type checking to avoid circular dependencies at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.cluster_state import ClusterStateManager
logger = logging.getLogger(__name__)


def _select_data_path_for_mode(
    use_real_data: bool,
    training_data_path: str,
    evaluation_data_path: Optional[str],
) -> str:
    """Pick data path based on mode; fallback to training path when evaluation path is missing."""
    if use_real_data and evaluation_data_path:
        return evaluation_data_path
    return training_data_path


def generate_next_combos(combo: np.ndarray) -> np.ndarray:
    """Generate all combos formed by flipping a single 1 to 0."""
    indices = np.where(combo == 1)[0]
    next_combos = np.tile(combo, (len(indices), 1))
    next_combos[np.arange(len(indices)), indices] = 0
    return next_combos


def greedy_recursive_search(
    current_combo: np.ndarray,
    gpu_need: int,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool = False,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
    avail_gpu: Optional[Sequence[int]] = None,
    global_mode_all: bool = False,
    evaluation_data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Recursive greedy search: keep removing one GPU until reaching gpu_need.

    :param global_mode: If True, score with global bandwidth (current + remaining GPUs)
    :param avail_gpu: Available GPUs; required when global_mode=True
    :param global_mode_all: When global_mode=True and this is True, global score =
        current combo + historical combos + remaining GPUs.
    """
    ones_count = int(np.sum(current_combo))
    if ones_count == gpu_need:
        return current_combo

    # Generate all single-GPU removal candidates
    candidate_combos = generate_next_combos(current_combo)
    
    # Bandwidth evaluation logic: prefer cluster_manager when available
    scores = []
    if cluster_manager:
        # Multi-tenant mode: consider contention
        if global_mode and avail_gpu is not None:
            # Global mode: evaluate combined bandwidth (current combo + remaining GPUs)
            num_dimensions = len(current_combo)
            for combo in candidate_combos:
                # Each candidate needs global evaluation with remaining GPUs
                bw = _evaluate_global_bandwidth(
                    combo,
                    avail_gpu,
                    num_dimensions,
                    cluster_manager,
                    global_mode_all=global_mode_all,
                )
                scores.append(bw)
        else:
            # Standard mode: only evaluate current combo
            for combo in candidate_combos:
                bw = cluster_manager.predict_with_contention(combo)
                scores.append(bw)
    elif if_real_data:
        real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
        for combo in candidate_combos:
            bw_value, _, _ = calculate_bandwidth_values(
                combo, total_gpu, gpu_bw_dict_list, switch_config, real_path
            )
            scores.append(bw_value)
    else:
        # Batch inference for all candidate bandwidth predictions
        part_bws, node_counts, total_counts = prepare_model_inputs(
            candidate_combos, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
        )
        scores = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)
        # predict_with_model may return tensor/array; ensure list
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif hasattr(scores, '__iter__') and not isinstance(scores, (list, tuple)):
            scores = list(scores)
            
    # Pick highest-bandwidth combo and recurse toward target GPU count
    best_idx = int(np.argmax(scores))
    best_next_combo = candidate_combos[best_idx]
    
    return greedy_recursive_search(
        best_next_combo,
        gpu_need,
        model,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
        device,
        artifact_dir,
        if_real_data,
        cluster_manager=cluster_manager,  # pass through recursively
        global_mode=global_mode,  # pass through recursively
        avail_gpu=avail_gpu,  # pass through recursively
        global_mode_all=global_mode_all,  # pass through recursively
        evaluation_data_path=evaluation_data_path,
    )


def tree_search_only(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    model,
    gpu_need: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool = False,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
    global_mode_all: bool = False,
    evaluation_data_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Improved search: remove from the largest set, support node/host insertion,
    optionally evaluate global bandwidth.

    :param num_dimensions: Total GPUs
    :param avail_gpu: Available GPU list
    :param model: Prediction model
    :param gpu_need: Required GPU count
    :param total_gpu: Total GPUs
    :param gpu_bw_dict_list: GPU bandwidth dictionaries
    :param switch_config: Switch config
    :param training_data_path: Data path for model prediction
    :param device: Device
    :param artifact_dir: Artifact directory
    :param if_real_data: Use real data if True
    :param global_mode: If True, score global bandwidth (“selected + remaining”) via cluster_manager
    :param global_mode_all: If True with global_mode, global score = current + history + remaining
    :return: Best GPU combo
    """
    if len(avail_gpu) < gpu_need:
        logger.warning("Available GPUs are fewer than required")
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
        """Score current combo, optionally using global bandwidth evaluation."""
        if global_mode and cluster_manager:
            return _evaluate_global_bandwidth(
                combo,
                avail_gpu,
                num_dimensions,
                cluster_manager,
                global_mode_all=global_mode_all,
            )
        return _evaluate_bandwidth(
            combo,
            model,
            total_gpu,
            gpu_bw_dict_list,
            switch_config,
            training_data_path,
            device,
            artifact_dir,
            use_real_data,
            cluster_manager,
            evaluation_data_path,
        )

    def _run_tree_paths(start_combo: np.ndarray, use_real_data: bool) -> np.ndarray:
        """Run single-card removal path from the given start combo."""
        combo_from_subtract = greedy_recursive_search(
            current_combo=start_combo,
            gpu_need=gpu_need,
            model=model,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
            device=device,
            artifact_dir=artifact_dir,
            if_real_data=use_real_data,
            cluster_manager=cluster_manager,
            global_mode=global_mode,
            avail_gpu=avail_gpu,
            global_mode_all=global_mode_all,
            evaluation_data_path=evaluation_data_path,
        )
        return combo_from_subtract

    def _select_best_complete_group(group_list: List[List[int]]) -> Optional[np.ndarray]:
        """Pick the best-bandwidth combo among available nodes/hosts."""
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
        """If gpu_need <= limit, try whole-node/host insertion then run tree search."""
        if gpu_need > limit:
            return None
        start_combo = _select_best_complete_group(group_list)
        if start_combo is None:
            return None
        # Historical design choice: once a node/host is inserted, always evaluate using the real-data path
        # to ensure consistent bandwidth ordering across candidates.
        return _run_tree_paths(start_combo, use_real_data=True)

    # Try 4-GPU node, then 8-GPU host; if both fail, fall back to max set
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
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool = False,
    global_mode_all: bool = False,
    evaluation_data_path: Optional[str] = None,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Try node-insertion optimization: see if a 4-GPU node or 8-GPU host can be the start.

    :param global_mode: If True, use global bandwidth evaluation
    :param global_mode_all: When global_mode=True and this is True, global score adds history + remaining
    :return: (success_insert, combo_result)
    """
    success_insert = False
    best_gpu = np.zeros(num_dimensions, dtype=int)
    
    # Check 4-GPU node insertion
    if gpu_need <= 4:
        # Collect all fully available 4-GPU nodes
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
                        temp_gpu,
                        total_gpu,
                        gpu_bw_dict_list,
                        switch_config,
                        _select_data_path_for_mode(True, training_data_path, evaluation_data_path),
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
                training_data_path=training_data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                global_mode=global_mode,
                avail_gpu=avail_gpu,
                global_mode_all=global_mode_all,
                evaluation_data_path=evaluation_data_path,
            )
            return (True, combo_result)
    

    if gpu_need <= 8 and not success_insert:
        # Collect all fully available 8-GPU hosts
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
                        temp_gpu,
                        total_gpu,
                        gpu_bw_dict_list,
                        switch_config,
                        _select_data_path_for_mode(True, training_data_path, evaluation_data_path),
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
                training_data_path=training_data_path,
                device=device,
                artifact_dir=artifact_dir,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                global_mode=global_mode,
                avail_gpu=avail_gpu,
                global_mode_all=global_mode_all,
                evaluation_data_path=evaluation_data_path,
            )
            return (True, combo_result)
    
    return (False, None)


def _evaluate_bandwidth(
    combo: np.ndarray,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    evaluation_data_path: Optional[str] = None,
) -> float:
    """
    Unified bandwidth evaluation: select method based on cluster_manager / if_real_data.

    :return: Bandwidth value
    """
    if cluster_manager:
        # Prefer online prediction to stay aligned with scheduler
        return cluster_manager.predict_with_contention(combo)
    elif if_real_data:
        # Real-data path: lookup/interpolate bandwidth
        real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
        bw_value, _, _ = calculate_bandwidth_values(
            combo, total_gpu, gpu_bw_dict_list, switch_config, real_path
        )
        return bw_value
    else:
        # Model inference path: reuse prepare_model_inputs even for single combo
        part_bws_list, node_counts_list, total_counts_list = prepare_model_inputs(
            np.array([combo]), total_gpu, gpu_bw_dict_list, switch_config, training_data_path
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
    global_mode_all: bool = False,
) -> float:
    """
    Evaluate global bandwidth.

    When global=True, evaluate current combo plus remaining GPUs.
    When global_mode_all=True, also add already-allocated combos.

    :param config: Current GPU combo (0/1)
    :param avail_gpu: Available GPUs
    :param num_dimensions: Total GPUs
    :param cluster_manager: Cluster state manager
    :return: Global bandwidth (with or without history based on global_mode_all)
    """
    # Bandwidth for current combo (already-selected training job benefit)
    current_bw = cluster_manager.predict_with_contention(config)
    
    # Remaining available GPUs (avail_gpu not in config)
    selected_gpus = set(np.where(config == 1)[0])
    remaining_gpus = [gpu for gpu in avail_gpu if gpu not in selected_gpus]
    
    remaining_bw = 0.0
    if remaining_gpus:
        # Build config for remaining GPUs (select all remaining)
        remaining_config = np.zeros(num_dimensions, dtype=int)
        remaining_config[remaining_gpus] = 1
        # Evaluate bandwidth of remaining GPUs
        remaining_bw = cluster_manager.predict_with_contention(remaining_config)

    total_bw = current_bw + remaining_bw
    if global_mode_all:
        total_bw += cluster_manager.get_total_active_bandwidth()

    return total_bw


def _compare_and_select_best(
    combo_from_subtract: Optional[np.ndarray],
    combo_from_eha: Optional[np.ndarray],
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    success_insert: bool = False,
    global_mode: bool = False,
    avail_gpu: Optional[Sequence[int]] = None,
    num_dimensions: Optional[int] = None,
    global_mode_all: bool = False,
    evaluation_data_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Compare two candidate combos and pick the higher-bandwidth one.

    :param combo_from_subtract: Result from subtract path
    :param combo_from_eha: Result from EHA path
    :param success_insert: Whether node insertion succeeded (special handling for else)
    :param global_mode: If True, use global bandwidth evaluation
    :param avail_gpu: Available GPUs (required when global_mode=True)
    :param num_dimensions: Total GPUs (required when global_mode=True)
    :param global_mode_all: When global_mode=True and True here, global score includes history + remaining
    :return: Best GPU combo
    """
    # Handle None cases
    candidates: List[Tuple[str, np.ndarray]] = []
    if combo_from_eha is not None:
        candidates.append(("eha", combo_from_eha))
    if combo_from_subtract is not None:
        candidates.append(("subtract", combo_from_subtract))

    if not candidates:
        logger.warning("All candidate results are None")
        return None

    if len(candidates) == 1:
        logger.warning("Only one candidate; returning it")
        return candidates[0][1]

    if cluster_manager:
        # In multi-tenant mode reuse online bandwidth evaluation
        if global_mode and avail_gpu is not None and num_dimensions is not None:
            bw_list = [
                _evaluate_global_bandwidth(
                    combo,
                    avail_gpu,
                    num_dimensions,
                    cluster_manager,
                    global_mode_all=global_mode_all,
                )
                for _, combo in candidates
            ]
        else:
            bw_list = [cluster_manager.predict_with_contention(combo) for _, combo in candidates]
    elif if_real_data:
        real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
        bw_list = [
            calculate_bandwidth_values(combo, total_gpu, gpu_bw_dict_list, switch_config, real_path)[0]
            for _, combo in candidates
        ]
    else:
        combo_array = np.array([combo for _, combo in candidates])
        part_bws_list, node_counts_list, total_counts_list = prepare_model_inputs(
            combo_array, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
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
            # After node insertion, adjust subtract-path score using real bandwidth
            for idx, (label, combo) in enumerate(candidates):
                if label == "subtract":
                    real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
                    bw_list[idx], _, _ = calculate_bandwidth_values(
                        combo, total_gpu, gpu_bw_dict_list, switch_config, real_path
                    )
                    break

    max_idx = int(np.argmax(bw_list))
    if not if_real_data:
        logger.info(f"Predicted best bandwidth: {bw_list[max_idx]}, algorithm: {candidates[max_idx][0]}")
    return candidates[max_idx][1]


def improved_searching_algo(
    num_dimensions: int,
    avail_gpu: Sequence[int],
    model,
    gpu_need: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool = False,
    cluster_manager: Optional['ClusterStateManager'] = None,
    global_mode: bool = False,
    global_mode_all: bool = False,
    evaluation_data_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Improved search algorithm with multi-tenant-aware bandwidth prediction.

    :param num_dimensions: Total GPUs
    :param avail_gpu: Available GPU list
    :param model: Prediction model
    :param gpu_need: Required GPU count
    :param total_gpu: Total GPUs
    :param gpu_bw_dict_list: GPU bandwidth dictionaries
    :param switch_config: Switch configuration
    :param training_data_path: Data path for model prediction
    :param device: Device
    :param artifact_dir: Artifact directory
    :param if_real_data: Use real data if True
    :param cluster_manager: Cluster state manager (for multi-tenant awareness)
    :param global_mode: If True, score with global bandwidth (current + remaining)
    :param global_mode_all: If True with global_mode, add history + remaining to global score
    :return: Best GPU combo
    """
    if len(avail_gpu) < gpu_need:
        logger.warning("Available GPUs are fewer than required")
        return None

    if len(avail_gpu) == gpu_need:
        return generate_data_minmax_restricted(
            1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
        )[0]
    # Path 1: original subtract-from-all-available approach
    max_gpu_combo = generate_data_minmax_restricted(
        1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
    )[0]

    # Prepare node and host lists
    complete_host_list = [[int(8 * i + e) for e in range(0, 8)] for i in range(0, int(num_dimensions / 8))]
    complete_node_list = [[int(4 * i + e) for e in range(0, 4)] for i in range(0, int(num_dimensions / 4))]

    # Try node insertion optimization: start from full node/host when available to shrink search space
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
        training_data_path=training_data_path,
        device=device,
        artifact_dir=artifact_dir,
        if_real_data=if_real_data,
        cluster_manager=cluster_manager,
        global_mode=global_mode,
        global_mode_all=global_mode_all,
        evaluation_data_path=evaluation_data_path,
    )


    if not success_insert:
        combo_from_subtract = greedy_recursive_search(
            current_combo=max_gpu_combo,
            gpu_need=gpu_need,
            model=model,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
            device=device,
            artifact_dir=artifact_dir,
            if_real_data=if_real_data,
            cluster_manager=cluster_manager,
            global_mode=global_mode,
            avail_gpu=avail_gpu,
            global_mode_all=global_mode_all,
            evaluation_data_path=evaluation_data_path,
        )

    # Second search path: invoke the heuristic EHA algorithm as an alternative candidate generator.
    from algorithms.eha import eha_search

    combo_from_eha = eha_search(
        num_dimensions,
        avail_gpu,
        model,
        gpu_need,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
        device,
        artifact_dir,
        if_real_data=if_real_data,  # Use function argument instead of hardcoding
        cluster_manager=cluster_manager,
        global_mode=global_mode,
        global_mode_all=global_mode_all,
        evaluation_data_path=evaluation_data_path,
    )

    # compare abd choose the best
    return _compare_and_select_best(
        combo_from_subtract=combo_from_subtract,
        combo_from_eha=combo_from_eha,
        model=model,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
        device=device,
        artifact_dir=artifact_dir,
        if_real_data=if_real_data,
        cluster_manager=cluster_manager,
        success_insert=success_insert,
        global_mode=global_mode,
        avail_gpu=avail_gpu,
        num_dimensions=num_dimensions,
        global_mode_all=global_mode_all,
        evaluation_data_path=evaluation_data_path,
    )

