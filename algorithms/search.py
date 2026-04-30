"""GPU-combination search algorithms for BandPilot.

This module implements the public search entry points, including
`improved_searching_algo(...)`, PTS sidecars, legacy exact PTS, and
threshold-legacy compatibility wrappers. Candidate scoring is centralized so
model prediction, real lookup, and `ClusterStateManager` contention evaluation
share consistent bandwidth semantics.
"""
from __future__ import annotations

import itertools
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from algorithms.hu_unit_gate import (
    build_active_unit_groups,
    build_unit_candidate_combos,
    resolve_hu_unit_sizes,
)
from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values, config_to_bandwidth, prepare_model_inputs
from core.gpu_config import generate_data_minmax_restricted
from training.evaluator import predict_with_model
# Imported only for type checking to avoid circular dependencies at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.cluster_state import ClusterStateManager
    from algorithms.cache import InferenceCache, DispatchCacheBundle
    from algorithms.runtime_adaptive import RuntimeAdaptiveKNNState
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
    inference_cache: Optional['InferenceCache'] = None,
) -> np.ndarray:
    """
    Recursive greedy search: keep removing one GPU until reaching gpu_need.

    :param global_mode: If True, score with global bandwidth (current + remaining GPUs)
    :param avail_gpu: Available GPUs; required when global_mode=True
    :param global_mode_all: When global_mode=True and this is True, global score =
        current combo + historical combos + remaining GPUs.
    :param inference_cache: Optional L1 cache for config -> bandwidth memoization
    """
    ones_count = int(np.sum(current_combo))
    if ones_count == gpu_need:
        return current_combo

    # Generate all single-GPU removal candidates
    candidate_combos = generate_next_combos(current_combo)

    # --- L1 cache-aware batch evaluation ---
    # Partition candidates into cache hits and misses, only predict misses.
    n_cands = len(candidate_combos)
    scores = np.empty(n_cands, dtype=float)
    if inference_cache is not None:
        hit_indices, miss_indices, hit_values = inference_cache.get_batch(candidate_combos)
        if hit_indices:
            scores[np.asarray(hit_indices, dtype=int)] = hit_values
    else:
        miss_indices = list(range(n_cands))

    if len(miss_indices) > 0:
        miss_combos = candidate_combos[miss_indices]

        # Bandwidth evaluation logic: prefer cluster_manager when available
        if cluster_manager:
            if global_mode and avail_gpu is not None:
                num_dimensions = len(current_combo)
                current_bws = cluster_manager.predict_with_contention_batch(miss_combos)
                remaining_configs = []
                for combo in miss_combos:
                    selected_gpus = set(np.where(combo == 1)[0])
                    remaining_gpus = [gpu for gpu in avail_gpu if gpu not in selected_gpus]
                    if remaining_gpus:
                        r_config = np.zeros(num_dimensions, dtype=int)
                        r_config[remaining_gpus] = 1
                        remaining_configs.append(r_config)
                    else:
                        remaining_configs.append(np.zeros(num_dimensions, dtype=int))
                remaining_bws = cluster_manager.predict_with_contention_batch(
                    np.array(remaining_configs)
                )
                miss_scores = current_bws + remaining_bws
                if global_mode_all:
                    miss_scores = miss_scores + cluster_manager.get_total_active_bandwidth()
            else:
                miss_scores = cluster_manager.predict_with_contention_batch(miss_combos)
        elif if_real_data:
            real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
            bw_array, _ = config_to_bandwidth(
                miss_combos, total_gpu, gpu_bw_dict_list, switch_config, real_path
            )
            miss_scores = bw_array
        else:
            part_bws, node_counts, total_counts = prepare_model_inputs(
                miss_combos, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
            )
            preds = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)
            miss_scores = np.asarray(preds, dtype=float).reshape(-1)

        miss_scores = np.asarray(miss_scores, dtype=float)
        # Store misses in L1 cache
        if inference_cache is not None:
            inference_cache.put_batch(miss_combos, miss_scores)

        scores[np.asarray(miss_indices, dtype=int)] = miss_scores

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
        cluster_manager=cluster_manager,
        global_mode=global_mode,
        avail_gpu=avail_gpu,
        global_mode_all=global_mode_all,
        evaluation_data_path=evaluation_data_path,
        inference_cache=inference_cache,
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
    :param global_mode: If True, score global bandwidth ("selected + remaining") via cluster_manager
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

    # --- Dispatch-scoped cache ---
    from algorithms.cache import DispatchCacheBundle
    cache_bundle = DispatchCacheBundle()
    _l1 = cache_bundle.l1
    if cluster_manager is not None:
        cluster_manager.set_super_combo_cache(cache_bundle.c0)

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
                inference_cache=_l1,
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
            inference_cache=_l1,
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
            inference_cache=_l1,
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

    def _cleanup_and_return(result):
        if cluster_manager is not None:
            cluster_manager.clear_super_combo_cache()
        return result

    # Try 4-GPU node, then 8-GPU host; if both fail, fall back to max set
    node_result = _attempt_insert(complete_node_list, limit=4)
    if node_result is not None:
        return _cleanup_and_return(node_result)

    host_result = _attempt_insert(complete_host_list, limit=8)
    if host_result is not None:
        return _cleanup_and_return(host_result)

    return _cleanup_and_return(_run_tree_paths(max_gpu_combo, use_real_data=if_real_data))


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
    inference_cache: Optional['InferenceCache'] = None,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Try node-insertion optimization: see if a 4-GPU node or 8-GPU host can be the start.

    :param global_mode: If True, use global bandwidth evaluation
    :param global_mode_all: When global_mode=True and this is True, global score adds history + remaining
    :return: (success_insert, combo_result)
    """
    avail_set = set(avail_gpu)

    def _run_insert_path(group_candidates: List[List[int]]) -> Tuple[bool, Optional[np.ndarray]]:
        available_groups = [group for group in group_candidates if set(group).issubset(avail_set)]
        if not available_groups:
            return False, None

        candidate_combos = np.zeros((len(available_groups), num_dimensions), dtype=int)
        for idx, group in enumerate(available_groups):
            candidate_combos[idx, group] = 1

        if global_mode and cluster_manager is not None:
            candidate_scores = _evaluate_global_bandwidth_batch(
                candidate_combos,
                avail_gpu,
                num_dimensions,
                cluster_manager,
                global_mode_all=global_mode_all,
                inference_cache=inference_cache,
            )
        else:
            candidate_scores = _evaluate_bandwidth_batch(
                candidate_combos,
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
                inference_cache=inference_cache,
            )

        start_combo = candidate_combos[int(np.argmax(candidate_scores))]
        combo_result = greedy_recursive_search(
            current_combo=start_combo,
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
            inference_cache=inference_cache,
        )
        return True, combo_result

    if gpu_need <= 4:
        success_insert, combo_result = _run_insert_path(complete_node_list)
        if success_insert:
            return True, combo_result

    if gpu_need <= 8:
        success_insert, combo_result = _run_insert_path(complete_host_list)
        if success_insert:
            return True, combo_result

    return (False, None)


def _predict_contention_batch(
    configs: Sequence[np.ndarray],
    cluster_manager: Optional['ClusterStateManager'],
    inference_cache: Optional['InferenceCache'] = None,
) -> np.ndarray:
    """Evaluate a batch through ClusterStateManager with optional L1 reuse."""
    if cluster_manager is None:
        raise ValueError("cluster_manager is required for contention-aware batch evaluation")

    configs_array = np.asarray(configs)
    if configs_array.ndim == 1:
        configs_array = configs_array.reshape(1, -1)
    if len(configs_array) == 0:
        return np.array([], dtype=float)

    if inference_cache is None:
        return np.asarray(cluster_manager.predict_with_contention_batch(configs_array), dtype=float)

    results = np.empty(len(configs_array), dtype=float)
    hit_indices, miss_indices, hit_values = inference_cache.get_batch(configs_array)
    if hit_indices:
        results[np.asarray(hit_indices, dtype=int)] = hit_values
    if not miss_indices:
        return results

    miss_configs = configs_array[miss_indices]
    miss_values = np.asarray(cluster_manager.predict_with_contention_batch(miss_configs), dtype=float)
    inference_cache.put_batch(miss_configs, miss_values)
    results[np.asarray(miss_indices, dtype=int)] = miss_values
    return results


def _evaluate_bandwidth_batch(
    configs: Sequence[np.ndarray],
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
    inference_cache: Optional['InferenceCache'] = None,
) -> np.ndarray:
    """Batch version of _evaluate_bandwidth with optional L1 reuse."""
    configs_array = np.asarray(configs)
    if configs_array.ndim == 1:
        configs_array = configs_array.reshape(1, -1)
    if len(configs_array) == 0:
        return np.array([], dtype=float)

    if cluster_manager is not None:
        return _predict_contention_batch(
            configs_array,
            cluster_manager,
            inference_cache=inference_cache,
        )

    results = np.empty(len(configs_array), dtype=float)
    if inference_cache is not None:
        hit_indices, miss_indices, hit_values = inference_cache.get_batch(configs_array)
        if hit_indices:
            results[np.asarray(hit_indices, dtype=int)] = hit_values
    else:
        miss_indices = list(range(len(configs_array)))

    if miss_indices:
        miss_configs = configs_array[miss_indices]
        if if_real_data:
            real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
            miss_values, _ = config_to_bandwidth(
                miss_configs, total_gpu, gpu_bw_dict_list, switch_config, real_path
            )
            miss_values = np.asarray(miss_values, dtype=float)
        else:
            part_bws_list, node_counts_list, total_counts_list = prepare_model_inputs(
                miss_configs, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
            )
            miss_values = np.asarray(
                predict_with_model(
                    model, part_bws_list, node_counts_list, total_counts_list, device, artifact_dir
                ),
                dtype=float,
            ).reshape(-1)
        if inference_cache is not None:
            inference_cache.put_batch(miss_configs, miss_values)
        results[np.asarray(miss_indices, dtype=int)] = miss_values

    return results


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
    inference_cache: Optional['InferenceCache'] = None,
) -> float:
    """
    Unified bandwidth evaluation: select method based on cluster_manager / if_real_data.

    :return: Bandwidth value
    """
    return float(
        _evaluate_bandwidth_batch(
            np.asarray([combo]),
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
            inference_cache=inference_cache,
        )[0]
    )


def _evaluate_global_bandwidth_batch(
    configs: Sequence[np.ndarray],
    avail_gpu: Sequence[int],
    num_dimensions: int,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode_all: bool = False,
    inference_cache: Optional['InferenceCache'] = None,
) -> np.ndarray:
    """Batch version of global bandwidth evaluation."""
    if cluster_manager is None:
        raise ValueError("cluster_manager is required for global bandwidth evaluation")

    configs_array = np.asarray(configs)
    if configs_array.ndim == 1:
        configs_array = configs_array.reshape(1, -1)
    if len(configs_array) == 0:
        return np.array([], dtype=float)

    current_bw = _predict_contention_batch(
        configs_array,
        cluster_manager,
        inference_cache=inference_cache,
    )
    remaining_configs = np.zeros((len(configs_array), num_dimensions), dtype=int)
    avail_gpu_list = list(avail_gpu)
    for idx, config in enumerate(configs_array):
        selected_gpus = set(np.where(config == 1)[0])
        remaining_gpus = [gpu for gpu in avail_gpu_list if gpu not in selected_gpus]
        if remaining_gpus:
            remaining_configs[idx, remaining_gpus] = 1

    remaining_bw = _predict_contention_batch(
        remaining_configs,
        cluster_manager,
        inference_cache=inference_cache,
    )
    total_bw = current_bw + remaining_bw
    if global_mode_all:
        total_bw = total_bw + cluster_manager.get_total_active_bandwidth()
    return total_bw


def _evaluate_global_bandwidth(
    config: np.ndarray,
    avail_gpu: Sequence[int],
    num_dimensions: int,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode_all: bool = False,
    inference_cache: Optional['InferenceCache'] = None,
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
    return float(
        _evaluate_global_bandwidth_batch(
            np.asarray([config]),
            avail_gpu,
            num_dimensions,
            cluster_manager,
            global_mode_all=global_mode_all,
            inference_cache=inference_cache,
        )[0]
    )


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
    inference_cache: Optional['InferenceCache'] = None,
) -> Optional[np.ndarray]:
    """
    Compare two candidate combos and pick the higher-bandwidth one.

    The structural rerank is already applied inside EHA. This final comparison
    therefore uses raw bandwidth scores so PTS and EHA candidates are evaluated
    on the same bandwidth basis.

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

    combo_array = np.array([combo for _, combo in candidates])
    if cluster_manager and global_mode and avail_gpu is not None and num_dimensions is not None:
        bw_array = _evaluate_global_bandwidth_batch(
            combo_array,
            avail_gpu,
            num_dimensions,
            cluster_manager,
            global_mode_all=global_mode_all,
            inference_cache=inference_cache,
        )
        bw_list = bw_array.tolist()
    else:
        bw_array = _evaluate_bandwidth_batch(
            combo_array,
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
            inference_cache=inference_cache,
        )
        bw_list = bw_array.tolist()

    if success_insert and not cluster_manager and not if_real_data:
        # Preserve the original comparison semantics for insertion-origin configs
        # in offline model mode.
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


def _run_pts_search_path(
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
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool,
    global_mode_all: bool,
    evaluation_data_path: Optional[str],
    inference_cache: Optional['InferenceCache'] = None,
) -> Tuple[Optional[np.ndarray], bool]:
    """Run the original PTS path and return (combo, used_insert_optimization)."""
    max_gpu_combo = generate_data_minmax_restricted(
        1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
    )[0]
    complete_host_list = [[int(8 * i + e) for e in range(0, 8)] for i in range(0, int(num_dimensions / 8))]
    complete_node_list = [[int(4 * i + e) for e in range(0, 4)] for i in range(0, int(num_dimensions / 4))]

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
        inference_cache=inference_cache,
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
            inference_cache=inference_cache,
        )
    return combo_from_subtract, success_insert


def _run_legacy_pts_search_path(
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
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool,
    global_mode_all: bool,
    evaluation_data_path: Optional[str],
    aggressive: bool = False,
    inference_cache: Optional['InferenceCache'] = None,
) -> Tuple[Optional[np.ndarray], bool, Dict[str, Any]]:
    """Run the legacy exact-PTS path used by `legacy_improved_searching_algo`.

    The helper normalizes legacy output to `(combo, success_insert, metadata)`
    so callers can compare it with the newer search paths without special cases.
    """

    combo_result, success_insert = _run_pts_search_path(
        num_dimensions=num_dimensions,
        avail_gpu=avail_gpu,
        model=model,
        gpu_need=gpu_need,
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
        inference_cache=inference_cache,
    )
    return combo_result, success_insert, {
        "pts_policy": "legacy_exact_pts",
        "hu_aggressive": bool(aggressive),
        "hu_host_removal_steps": 0,
        "hu_removed_host_ids": [],
        "hu_unit_removal_steps": 0,
        "hu_removed_unit_trace": [],
        "hu_unit_sizes": [],
        "hu_target_capacity": None,
        "hu_skipped_reason": "legacy_path",
    }


def _score_candidate_pool(
    candidate_combos: np.ndarray,
    *,
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool,
    avail_gpu: Optional[Sequence[int]],
    global_mode_all: bool,
    evaluation_data_path: Optional[str],
    inference_cache: Optional['InferenceCache'],
) -> np.ndarray:
    """Score a candidate pool with the same bandwidth path used by PTS.

    The coarse host/GPU pruning stage and the PTS refinement stage both call
    this helper so global-mode contention scoring, offline lookup scoring, and
    dispatch-scoped inference caches stay consistent.
    """

    if len(candidate_combos) == 0:
        return np.array([], dtype=float)

    if global_mode and cluster_manager is not None and avail_gpu is not None:
        return _evaluate_global_bandwidth_batch(
            candidate_combos,
            avail_gpu,
            int(candidate_combos.shape[1]),
            cluster_manager,
            global_mode_all=global_mode_all,
            inference_cache=inference_cache,
        )
    return _evaluate_bandwidth_batch(
        candidate_combos,
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
        inference_cache=inference_cache,
    )


def _run_hu_insertion_preserving_pts_search_path(
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
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool,
    global_mode_all: bool,
    evaluation_data_path: Optional[str],
    aggressive: bool = False,
    inference_cache: Optional['InferenceCache'] = None,
) -> Tuple[Optional[np.ndarray], bool, Dict[str, Any]]:
    """Run the insertion-preserving PTS path used by current BandPilot.

    Small requests stay in the insertion-exact zone. Larger requests first prune
    whole topology units while preserving enough slack for insertion refinement,
    then fall back to exact PTS on the reduced candidate set. `aggressive=True`
    lets scalability experiments trade more pruning for lower latency.
    """

    if gpu_need <= 8:
        max_gpu_combo = generate_data_minmax_restricted(
            1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
        )[0]
        complete_host_list = [[int(8 * i + e) for e in range(0, 8)] for i in range(0, int(num_dimensions / 8))]
        complete_node_list = [[int(4 * i + e) for e in range(0, 4)] for i in range(0, int(num_dimensions / 4))]
        success_insert, combo_result = _try_node_insert_optimization(
            num_dimensions=num_dimensions,
            avail_gpu=avail_gpu,
            model=model,
            gpu_need=gpu_need,
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
            inference_cache=inference_cache,
            complete_node_list=complete_node_list,
            complete_host_list=complete_host_list,
            max_gpu_combo=max_gpu_combo,
        )
        if success_insert:
            return combo_result, success_insert, {
                "pts_policy": "HU_ALWAYS_EXCEPT_INSERTION_REFINE_AT_KPLUS8",
                "hu_aggressive": bool(aggressive),
                "hu_host_removal_steps": 0,
                "hu_removed_host_ids": [],
                "hu_unit_removal_steps": 0,
                "hu_removed_unit_trace": [],
                "hu_unit_sizes": [],
                "hu_target_capacity": None,
                "hu_skipped_reason": "insertion_exact_zone",
            }

    max_gpu_combo = generate_data_minmax_restricted(
        1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
    )[0]
    target_capacity = int(gpu_need + 8)
    current_combo = np.asarray(max_gpu_combo, dtype=int).copy()
    removed_host_ids: List[int] = []
    removed_unit_trace: List[str] = []
    hu_unit_removal_steps = 0
    hu_skipped_reason = "coarse_stage_not_needed"
    hu_unit_sizes = resolve_hu_unit_sizes(
        num_dimensions=num_dimensions,
        gate_config={"aggressive": bool(aggressive)},
    )

    for unit_size in hu_unit_sizes:
        while int(current_combo.sum()) > target_capacity:
            active_units, active_unit_ids = build_active_unit_groups(
                current_combo,
                num_dimensions=num_dimensions,
                unit_size=int(unit_size),
            )
            removable_units: List[List[int]] = []
            removable_unit_ids: List[int] = []
            for unit_id, unit_members in zip(active_unit_ids, active_units):
                if int(current_combo.sum()) - len(unit_members) >= target_capacity:
                    removable_units.append(unit_members)
                    removable_unit_ids.append(int(unit_id))
            if not removable_units:
                if hu_unit_removal_steps == 0:
                    hu_skipped_reason = "no_removable_unit_before_target"
                break

            candidate_combos = build_unit_candidate_combos(
                current_combo,
                removable_units=removable_units,
            )
            candidate_scores = _score_candidate_pool(
                candidate_combos,
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
                inference_cache=inference_cache,
            )
            best_idx = int(np.argmax(candidate_scores))
            current_combo = np.asarray(candidate_combos[best_idx], dtype=int)
            hu_unit_removal_steps += 1
            removed_unit_id = int(removable_unit_ids[best_idx])
            removed_unit_trace.append(f"{int(unit_size)}gpu_unit:{removed_unit_id}")
            if int(unit_size) == 8:
                removed_host_ids.append(removed_unit_id)
            hu_skipped_reason = "hu_unit_removed"

    if current_combo.sum() > gpu_need:
        combo_result = greedy_recursive_search(
            current_combo=current_combo,
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
            inference_cache=inference_cache,
        )
    else:
        combo_result = np.asarray(current_combo, dtype=int)

    return combo_result, False, {
        "pts_policy": "HU_ALWAYS_EXCEPT_INSERTION_REFINE_AT_KPLUS8",
        "hu_aggressive": bool(aggressive),
        "hu_host_removal_steps": int(len(removed_host_ids)),
        "hu_removed_host_ids": removed_host_ids,
        "hu_unit_removal_steps": int(hu_unit_removal_steps),
        "hu_removed_unit_trace": removed_unit_trace,
        "hu_unit_sizes": hu_unit_sizes,
        "hu_target_capacity": int(target_capacity),
        "hu_skipped_reason": hu_skipped_reason,
    }


def _score_single_combo_for_runtime_label(
    *,
    combo: Optional[np.ndarray],
    model,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    device,
    artifact_dir: Path,
    if_real_data: bool,
    cluster_manager: Optional['ClusterStateManager'],
    global_mode: bool,
    avail_gpu: Sequence[int],
    num_dimensions: int,
    global_mode_all: bool,
    evaluation_data_path: Optional[str],
    inference_cache: Optional['InferenceCache'],
) -> float:
    """Score one combo for runtime-adaptive labels using the active search domain."""

    if combo is None:
        return 0.0
    combo_arr = np.asarray(combo, dtype=int)
    if cluster_manager and global_mode and avail_gpu is not None:
        return float(
            _evaluate_global_bandwidth_batch(
                np.asarray([combo_arr]),
                avail_gpu,
                num_dimensions,
                cluster_manager,
                global_mode_all=global_mode_all,
                inference_cache=inference_cache,
            )[0]
        )
    return float(
        _evaluate_bandwidth_batch(
            np.asarray([combo_arr]),
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
            inference_cache=inference_cache,
        )[0]
    )


def _build_runtime_adaptive_context(
    *,
    switch_config: SwitchBandwidthConfig | float | None,
    cluster_manager: Optional['ClusterStateManager'],
    total_gpu: int,
    gpu_need: int,
    avail_gpu: Sequence[int],
    if_real_data: bool,
) -> Dict[str, Any]:
    """Build the per-case context passed to the runtime-adaptive state."""

    cluster_type = str(getattr(switch_config, "cluster_type", "")) if switch_config is not None else ""
    contention_mode = str(getattr(cluster_manager, "contention_mode", "idle")).strip().lower()
    return {
        "cluster_type": cluster_type,
        "contention_mode": contention_mode or "idle",
        "total_gpu": int(total_gpu),
        "test_num": int(gpu_need),
        "if_dynamic": bool(len(avail_gpu) < int(total_gpu)),
        "search_if_real_data": bool(if_real_data),
    }


def threshold_legacy_decide_pts_trigger(
    eha_result: Optional[np.ndarray],
    eha_meta: Dict[str, Any],
    gpu_need: int,
    avail_gpu: Sequence[int],
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
) -> Tuple[bool, str]:
    """Decide whether the legacy threshold policy should trigger PTS."""
    if eha_result is None:
        return True, "eha_infeasible"

    node_count = int(eha_meta.get("node_count", 0))
    min_node_density = int(eha_meta.get("min_node_density", 0))
    num_candidates = int(eha_meta.get("num_candidates", 0))
    bw_cv = float(eha_meta.get("bw_cv", 0.0))
    top5_gap = float(eha_meta.get("top5_gap", 0.0))

    if node_count <= 1:
        return False, "fast_path_single_node"
    # Historical compact-placement and small-search-space fast paths remain
    # disabled for conservative validation.
    if num_candidates < min_candidates_for_cv:
        return True, "insufficient_candidates"
    if bw_cv < cv_threshold and top5_gap < gap_threshold:
        return False, "low_cv"
    return True, "high_cv"


def decide_pts_trigger(
    eha_result: Optional[np.ndarray],
    eha_meta: Dict[str, Any],
    gpu_need: int,
    avail_gpu: Sequence[int],
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
) -> Tuple[bool, str]:
    """Compatibility wrapper for `threshold_legacy_decide_pts_trigger`."""
    return threshold_legacy_decide_pts_trigger(
        eha_result=eha_result,
        eha_meta=eha_meta,
        gpu_need=gpu_need,
        avail_gpu=avail_gpu,
        cv_threshold=cv_threshold,
        gap_threshold=gap_threshold,
        min_candidates_for_cv=min_candidates_for_cv,
    )


def threshold_legacy_should_trigger_pts(
    eha_result: Optional[np.ndarray],
    eha_meta: Dict[str, Any],
    gpu_need: int,
    avail_gpu: Sequence[int],
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
) -> bool:
    """Return only the boolean decision from the legacy threshold policy."""
    triggered, _ = threshold_legacy_decide_pts_trigger(
        eha_result=eha_result,
        eha_meta=eha_meta,
        gpu_need=gpu_need,
        avail_gpu=avail_gpu,
        cv_threshold=cv_threshold,
        gap_threshold=gap_threshold,
        min_candidates_for_cv=min_candidates_for_cv,
    )
    return triggered


def should_trigger_pts(
    eha_result: Optional[np.ndarray],
    eha_meta: Dict[str, Any],
    gpu_need: int,
    avail_gpu: Sequence[int],
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
) -> bool:
    """Compatibility wrapper for `threshold_legacy_should_trigger_pts`."""
    return threshold_legacy_should_trigger_pts(
        eha_result=eha_result,
        eha_meta=eha_meta,
        gpu_need=gpu_need,
        avail_gpu=avail_gpu,
        cv_threshold=cv_threshold,
        gap_threshold=gap_threshold,
        min_candidates_for_cv=min_candidates_for_cv,
    )


def _improved_searching_algo_impl(
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
    adaptive_pts: bool = False,
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
    return_metadata: bool = False,
    pts_path_runner=_run_legacy_pts_search_path,
    pts_policy_name: str = "legacy_exact_pts",
    adaptive_runtime_state: Optional['RuntimeAdaptiveKNNState'] = None,
    enable_threshold_legacy_adaptive: bool = False,
    aggressive: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
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
    adaptive_policy_name = (
        "threshold_legacy"
        if adaptive_pts and enable_threshold_legacy_adaptive
        else ("adaptive_knn" if adaptive_pts else "disabled")
    )

    if len(avail_gpu) < gpu_need:
        logger.warning("Available GPUs are fewer than required")
        result: Optional[np.ndarray] = None
        if return_metadata:
            return result, {
                "adaptive_pts": adaptive_pts,
                "adaptive_policy_name": adaptive_policy_name,
                "pts_triggered": False,
                "trigger_reason": "insufficient_avail_gpu",
                "eha_meta": {},
                "eha_time": 0.0,
                "pts_time": 0.0,
                "success_insert": False,
            }
        return result

    if len(avail_gpu) == gpu_need:
        result = generate_data_minmax_restricted(
            1, num_dimensions, min_ones=len(avail_gpu), max_ones=len(avail_gpu), avail_gpu=avail_gpu
        )[0]
        if return_metadata:
            return result, {
                "adaptive_pts": adaptive_pts,
                "adaptive_policy_name": adaptive_policy_name,
                "pts_triggered": False,
                "trigger_reason": "exact_fit",
                "eha_meta": {},
                "eha_time": 0.0,
                "pts_time": 0.0,
                "success_insert": False,
            }
        return result

    from algorithms.eha import eha_search
    from algorithms.cache import DispatchCacheBundle

    # --- Dispatch-scoped cache shared by PTS, EHA and final comparison ---
    _shared_cache = DispatchCacheBundle()
    _shared_l1 = _shared_cache.l1
    if cluster_manager is not None:
        cluster_manager.set_super_combo_cache(_shared_cache.c0)

    def _cleanup_cache():
        if cluster_manager is not None:
            cluster_manager.clear_super_combo_cache()

    search_meta: Dict[str, Any] = {
        "adaptive_pts": adaptive_pts,
        "adaptive_policy_name": adaptive_policy_name,
        "pts_triggered": True,
        "trigger_reason": "always_run_pts",
        "eha_meta": {},
        "eha_time": 0.0,
        "pts_time": 0.0,
        "success_insert": False,
        "pts_policy": pts_policy_name,
        "hu_aggressive": bool(aggressive),
        "hu_host_removal_steps": 0,
        "hu_removed_host_ids": [],
        "hu_unit_removal_steps": 0,
        "hu_removed_unit_trace": [],
        "hu_unit_sizes": [],
        "hu_target_capacity": None,
        "hu_skipped_reason": "not_applicable",
        "adaptive_bank_id": "",
        "adaptive_bank_version": -1,
        "adaptive_bank_phase": "",
        "adaptive_bank_active_before": False,
        "adaptive_train_size_before": 0,
        "adaptive_case_index": -1,
        "adaptive_shadow_trigger_reason": "",
        "adaptive_shadow_trigger_pts": False,
        "adaptive_online_risk": 0.0,
        "adaptive_online_support": 0,
        "adaptive_online_low_trust": False,
        "adaptive_support_insufficient": False,
        "adaptive_decision_overhead_ms": 0.0,
    }

    def _run_eha_with_confidence() -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Run EHA and merge timing plus confidence metadata into `search_meta`."""
        eha_start = time.perf_counter()
        combo_from_eha_local, eha_meta_local = eha_search(
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
            if_real_data=if_real_data,
            cluster_manager=cluster_manager,
            global_mode=global_mode,
            global_mode_all=global_mode_all,
            evaluation_data_path=evaluation_data_path,
            return_confidence=True,
            cache_bundle=_shared_cache,
        )
        search_meta["eha_time"] = time.perf_counter() - eha_start
        search_meta["eha_meta"] = eha_meta_local
        return combo_from_eha_local, eha_meta_local

    if adaptive_pts:
        combo_from_eha, eha_meta = _run_eha_with_confidence()

        if enable_threshold_legacy_adaptive:
            pts_triggered, trigger_reason = threshold_legacy_decide_pts_trigger(
                eha_result=combo_from_eha,
                eha_meta=eha_meta,
                gpu_need=gpu_need,
                avail_gpu=avail_gpu,
                cv_threshold=cv_threshold,
                gap_threshold=gap_threshold,
                min_candidates_for_cv=min_candidates_for_cv,
            )
            search_meta["pts_triggered"] = pts_triggered
            search_meta["trigger_reason"] = trigger_reason

            combo_from_subtract = None
            success_insert = False
            if pts_triggered:
                pts_start = time.perf_counter()
                combo_from_subtract, success_insert, pts_meta = pts_path_runner(
                    num_dimensions=num_dimensions,
                    avail_gpu=avail_gpu,
                    model=model,
                    gpu_need=gpu_need,
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
                    aggressive=aggressive,
                    inference_cache=_shared_l1,
                )
                search_meta["pts_time"] = time.perf_counter() - pts_start
                search_meta["success_insert"] = success_insert
                search_meta.update(pts_meta)

                result = _compare_and_select_best(
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
                    inference_cache=_shared_l1,
                )
            else:
                result = combo_from_eha

            _cleanup_cache()
            if return_metadata:
                return result, search_meta
            return result

        if adaptive_runtime_state is None:
            _cleanup_cache()
            raise ValueError(
                "improved_searching_algo(adaptive_pts=True) requires a "
                "runtime-adaptive state; use threshold_legacy_* for threshold policies."
            )

        if (
            abs(float(cv_threshold) - 0.05) > 1e-12
            or abs(float(gap_threshold) - 0.03) > 1e-12
            or int(min_candidates_for_cv) != 5
        ):
            logger.warning(
                "Runtime adaptive ignores cv/gap/min-candidate thresholds; "
                "use threshold_legacy_* for threshold policies."
            )

        eha_final_bw = _score_single_combo_for_runtime_label(
            combo=combo_from_eha,
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
            num_dimensions=num_dimensions,
            global_mode_all=global_mode_all,
            evaluation_data_path=evaluation_data_path,
            inference_cache=_shared_l1,
        )
        runtime_context = _build_runtime_adaptive_context(
            switch_config=switch_config,
            cluster_manager=cluster_manager,
            total_gpu=total_gpu,
            gpu_need=gpu_need,
            avail_gpu=avail_gpu,
            if_real_data=if_real_data,
        )
        runtime_sample, runtime_feature_row = adaptive_runtime_state.build_eha_decision_input(
            total_gpu=total_gpu,
            gpu_need=gpu_need,
            avail_gpu=avail_gpu,
            if_real_data=if_real_data,
            cluster_type=str(runtime_context["cluster_type"]),
            contention_mode=str(runtime_context["contention_mode"]),
            eha_combo=combo_from_eha,
            eha_meta=eha_meta,
            eha_search_latency_s=float(search_meta["eha_time"]),
            eha_final_bw=float(eha_final_bw),
            context=runtime_context,
        )
        adaptive_decision = adaptive_runtime_state.decide_case(
            sample=runtime_sample,
            feature_row=runtime_feature_row,
        )
        search_meta.update(
            {
                "pts_triggered": bool(adaptive_decision.trigger_pts),
                "trigger_reason": str(adaptive_decision.trigger_reason),
                "adaptive_bank_id": str(adaptive_runtime_state.bank_id),
                "adaptive_bank_version": int(adaptive_decision.bank_version),
                "adaptive_bank_phase": str(adaptive_decision.bank_phase),
                "adaptive_bank_active_before": bool(adaptive_decision.bank_active_before),
                "adaptive_train_size_before": int(adaptive_decision.train_size_before),
                "adaptive_case_index": int(adaptive_decision.case_index),
                "adaptive_shadow_trigger_reason": str(adaptive_decision.shadow_trigger_reason),
                "adaptive_shadow_trigger_pts": bool(adaptive_decision.shadow_trigger_pts),
                "adaptive_online_risk": float(adaptive_decision.online_risk),
                "adaptive_online_support": int(adaptive_decision.support_count),
                "adaptive_online_low_trust": bool(adaptive_decision.online_low_trust),
                "adaptive_support_insufficient": bool(adaptive_decision.support_insufficient),
                "adaptive_decision_overhead_ms": float(adaptive_decision.decision_overhead_ms),
            }
        )

        combo_from_subtract = None
        success_insert = False
        if adaptive_decision.trigger_pts:
            pts_start = time.perf_counter()
            combo_from_subtract, success_insert, pts_meta = pts_path_runner(
                num_dimensions=num_dimensions,
                avail_gpu=avail_gpu,
                model=model,
                gpu_need=gpu_need,
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
                aggressive=aggressive,
                inference_cache=_shared_l1,
            )
            search_meta["pts_time"] = time.perf_counter() - pts_start
            search_meta["success_insert"] = success_insert
            search_meta.update(pts_meta)

            result = _compare_and_select_best(
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
                inference_cache=_shared_l1,
            )
            bandpilot_final_bw = _score_single_combo_for_runtime_label(
                combo=result,
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
                num_dimensions=num_dimensions,
                global_mode_all=global_mode_all,
                evaluation_data_path=evaluation_data_path,
                inference_cache=_shared_l1,
            )
            adaptive_runtime_state.observe_labeled_case(
                sample=runtime_sample,
                feature_row=runtime_feature_row,
                decision=adaptive_decision,
                bandpilot_combo=result,
                bandpilot_final_bw=float(bandpilot_final_bw),
                bandpilot_search_latency_s=float(search_meta["eha_time"] + search_meta["pts_time"]),
            )
        else:
            result = combo_from_eha
            adaptive_runtime_state.record_unlabeled_skip_case()

        _cleanup_cache()
        if return_metadata:
            return result, search_meta
        return result

    # Non-adaptive: PTS runs first, then EHA
    pts_start = time.perf_counter()
    combo_from_subtract, success_insert, pts_meta = pts_path_runner(
        num_dimensions=num_dimensions,
        avail_gpu=avail_gpu,
        model=model,
        gpu_need=gpu_need,
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
        aggressive=aggressive,
        inference_cache=_shared_l1,
    )
    search_meta["pts_time"] = time.perf_counter() - pts_start
    search_meta["success_insert"] = success_insert
    search_meta.update(pts_meta)

    combo_from_eha, eha_meta = _run_eha_with_confidence()

    result = _compare_and_select_best(
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
        inference_cache=_shared_l1,
    )
    _cleanup_cache()
    if return_metadata:
        return result, search_meta
    return result


def legacy_improved_searching_algo(
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
    adaptive_pts: bool = False,
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
    return_metadata: bool = False,
    aggressive: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run the legacy exact-PTS BandPilot implementation.

    This path is retained as the legacy reference. If `adaptive_pts=True` is
    passed, callers are redirected to the threshold-adaptive legacy wrapper
    because runtime-adaptive PTS belongs to the current BandPilot path.
    """

    if adaptive_pts:
        logger.warning(
            "legacy_improved_searching_algo(adaptive_pts=True) is deprecated; "
            "use threshold_legacy_exact_improved_searching_algo(...) instead."
        )
        return threshold_legacy_exact_improved_searching_algo(
            num_dimensions=num_dimensions,
            avail_gpu=avail_gpu,
            model=model,
            gpu_need=gpu_need,
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
            cv_threshold=cv_threshold,
            gap_threshold=gap_threshold,
            min_candidates_for_cv=min_candidates_for_cv,
            return_metadata=return_metadata,
            aggressive=aggressive,
        )

    return _improved_searching_algo_impl(
        num_dimensions=num_dimensions,
        avail_gpu=avail_gpu,
        model=model,
        gpu_need=gpu_need,
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
        adaptive_pts=False,
        cv_threshold=cv_threshold,
        gap_threshold=gap_threshold,
        min_candidates_for_cv=min_candidates_for_cv,
        return_metadata=return_metadata,
        pts_path_runner=_run_legacy_pts_search_path,
        pts_policy_name="legacy_exact_pts",
        aggressive=aggressive,
    )


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
    adaptive_pts: bool = False,
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
    return_metadata: bool = False,
    adaptive_runtime_state: Optional['RuntimeAdaptiveKNNState'] = None,
    aggressive: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run the public BandPilot search path.

    BandPilot combines EHA with the current PTS path:
    - `gpu_need <= 8` uses the insertion exact zone.
    - `gpu_need > 8` uses topology-aware unit grouping and exact refinement
      over a `target_capacity = k + 8` candidate set.
    - `aggressive=False` uses the default `8 -> 1` unit refinement schedule.
    - `aggressive=True` allows a topology-aligned unit schedule.

    When `adaptive_pts=True`, a `RuntimeAdaptiveKNNState` may skip or trigger
    PTS online. The caller owns bank boundaries through
    `adaptive_runtime_state.finish_bank()`.
    """

    return _improved_searching_algo_impl(
        num_dimensions=num_dimensions,
        avail_gpu=avail_gpu,
        model=model,
        gpu_need=gpu_need,
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
        adaptive_pts=adaptive_pts,
        cv_threshold=cv_threshold,
        gap_threshold=gap_threshold,
        min_candidates_for_cv=min_candidates_for_cv,
        return_metadata=return_metadata,
        pts_path_runner=_run_hu_insertion_preserving_pts_search_path,
        pts_policy_name="HU_ALWAYS_EXCEPT_INSERTION_REFINE_AT_KPLUS8",
        adaptive_runtime_state=adaptive_runtime_state,
        aggressive=aggressive,
    )


def threshold_legacy_exact_improved_searching_algo(
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
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
    return_metadata: bool = False,
    aggressive: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run legacy-BandPilot threshold adaptation with legacy exact PTS."""

    return _improved_searching_algo_impl(
        num_dimensions=num_dimensions,
        avail_gpu=avail_gpu,
        model=model,
        gpu_need=gpu_need,
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
        adaptive_pts=True,
        cv_threshold=cv_threshold,
        gap_threshold=gap_threshold,
        min_candidates_for_cv=min_candidates_for_cv,
        return_metadata=return_metadata,
        pts_path_runner=_run_legacy_pts_search_path,
        pts_policy_name="legacy_exact_pts",
        enable_threshold_legacy_adaptive=True,
        aggressive=aggressive,
    )


def threshold_legacy_improved_searching_algo(
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
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_candidates_for_cv: int = 5,
    return_metadata: bool = False,
    aggressive: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run legacy threshold adaptation with the current PTS path."""

    return _improved_searching_algo_impl(
        num_dimensions=num_dimensions,
        avail_gpu=avail_gpu,
        model=model,
        gpu_need=gpu_need,
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
        adaptive_pts=True,
        cv_threshold=cv_threshold,
        gap_threshold=gap_threshold,
        min_candidates_for_cv=min_candidates_for_cv,
        return_metadata=return_metadata,
        pts_path_runner=_run_hu_insertion_preserving_pts_search_path,
        pts_policy_name="HU_ALWAYS_EXCEPT_INSERTION_REFINE_AT_KPLUS8",
        enable_threshold_legacy_adaptive=True,
        aggressive=aggressive,
    )


def hu_pts_only_search(
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
    return_metadata: bool = False,
    aggressive: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run the current PTS path.

    This wrapper exposes the PTS primitive without the final EHA
    compare-and-select step. It is used by scalability sidecars to compare the
    current PTS path against legacy exact PTS and by BandPilot when runtime
    adaptation triggers the PTS backend.
    """

    if len(avail_gpu) < gpu_need:
        logger.warning("Available GPUs are fewer than required")
        result: Optional[np.ndarray] = None
        if return_metadata:
            return result, {
                "adaptive_pts": False,
                "pts_triggered": True,
                "trigger_reason": "insufficient_avail_gpu",
                "eha_meta": {},
                "eha_time": 0.0,
                "pts_time": 0.0,
                "success_insert": False,
                "pts_policy": "HU_ALWAYS_EXCEPT_INSERTION_REFINE_AT_KPLUS8",
                "hu_aggressive": bool(aggressive),
            }
        return result

    if len(avail_gpu) == gpu_need:
        result = generate_data_minmax_restricted(
            1,
            num_dimensions,
            min_ones=len(avail_gpu),
            max_ones=len(avail_gpu),
            avail_gpu=avail_gpu,
        )[0]
        if return_metadata:
            return result, {
                "adaptive_pts": False,
                "pts_triggered": True,
                "trigger_reason": "exact_fit",
                "eha_meta": {},
                "eha_time": 0.0,
                "pts_time": 0.0,
                "success_insert": False,
                "pts_policy": "HU_ALWAYS_EXCEPT_INSERTION_REFINE_AT_KPLUS8",
                "hu_aggressive": bool(aggressive),
            }
        return result

    from algorithms.cache import DispatchCacheBundle

    shared_cache = DispatchCacheBundle()
    if cluster_manager is not None:
        cluster_manager.set_super_combo_cache(shared_cache.c0)

    try:
        pts_start = time.perf_counter()
        combo_result, success_insert, pts_meta = _run_hu_insertion_preserving_pts_search_path(
            num_dimensions=num_dimensions,
            avail_gpu=avail_gpu,
            model=model,
            gpu_need=gpu_need,
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
            aggressive=aggressive,
            inference_cache=shared_cache.l1,
        )
        pts_time = time.perf_counter() - pts_start
    finally:
        if cluster_manager is not None:
            cluster_manager.clear_super_combo_cache()

    if not return_metadata:
        return combo_result

    search_meta: Dict[str, Any] = {
        "adaptive_pts": False,
        "pts_triggered": True,
        "trigger_reason": "hu_pts_only",
        "eha_meta": {},
        "eha_time": 0.0,
        "pts_time": float(pts_time),
        "success_insert": bool(success_insert),
        "pts_policy": "HU_ALWAYS_EXCEPT_INSERTION_REFINE_AT_KPLUS8",
        "hu_aggressive": bool(aggressive),
    }
    search_meta.update(pts_meta)
    return combo_result, search_meta


def legacy_pts_only_search(
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
    return_metadata: bool = False,
    aggressive: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run the legacy exact PTS path.

    This reviewer-traceable sidecar calls `_run_legacy_pts_search_path(...)`
    directly and emits metadata compatible with `hu_pts_only_search(...)`.
    Public outputs should label this algorithm as `legacy-PTS`.
    """

    if len(avail_gpu) < gpu_need:
        logger.warning("Available GPUs are fewer than required")
        result: Optional[np.ndarray] = None
        if return_metadata:
            return result, {
                "adaptive_pts": False,
                "pts_triggered": True,
                "trigger_reason": "insufficient_avail_gpu",
                "eha_meta": {},
                "eha_time": 0.0,
                "pts_time": 0.0,
                "success_insert": False,
                "pts_policy": "legacy_exact_pts",
                "hu_aggressive": bool(aggressive),
            }
        return result

    if len(avail_gpu) == gpu_need:
        result = generate_data_minmax_restricted(
            1,
            num_dimensions,
            min_ones=len(avail_gpu),
            max_ones=len(avail_gpu),
            avail_gpu=avail_gpu,
        )[0]
        if return_metadata:
            return result, {
                "adaptive_pts": False,
                "pts_triggered": True,
                "trigger_reason": "exact_fit",
                "eha_meta": {},
                "eha_time": 0.0,
                "pts_time": 0.0,
                "success_insert": False,
                "pts_policy": "legacy_exact_pts",
                "hu_aggressive": bool(aggressive),
            }
        return result

    from algorithms.cache import DispatchCacheBundle

    shared_cache = DispatchCacheBundle()
    if cluster_manager is not None:
        cluster_manager.set_super_combo_cache(shared_cache.c0)

    try:
        pts_start = time.perf_counter()
        combo_result, success_insert, pts_meta = _run_legacy_pts_search_path(
            num_dimensions=num_dimensions,
            avail_gpu=avail_gpu,
            model=model,
            gpu_need=gpu_need,
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
            aggressive=aggressive,
            inference_cache=shared_cache.l1,
        )
        pts_time = time.perf_counter() - pts_start
    finally:
        if cluster_manager is not None:
            cluster_manager.clear_super_combo_cache()

    if not return_metadata:
        return combo_result

    search_meta: Dict[str, Any] = {
        "adaptive_pts": False,
        "pts_triggered": True,
        "trigger_reason": "legacy_pts_only",
        "eha_meta": {},
        "eha_time": 0.0,
        "pts_time": float(pts_time),
        "success_insert": bool(success_insert),
        "pts_policy": "legacy_exact_pts",
        "hu_aggressive": bool(aggressive),
    }
    search_meta.update(pts_meta)
    return combo_result, search_meta
