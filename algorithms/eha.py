"""Equilibrium-driven Heuristic Algorithm (EHA)."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from algorithms.contention_score import (
    apply_contention_structural_rerank,
    is_contention_sensitive_mode,
)
from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values, config_to_bandwidth, prepare_model_inputs
from training.evaluator import predict_with_model
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.cluster_state import ClusterStateManager
    from algorithms.cache import InferenceCache, SubproblemCache, DispatchCacheBundle

logger = logging.getLogger(__name__)


_EHA_EPS = 1e-9
_EHA_GPUS_PER_NODE = 8
_EHA_FLAT_BUDGET = 5
_EHA_FLAT_EXTRA_PROBE_BUDGET = 2
_EHA_FLAT_CONTENTION_BUDGET = 8
_EHA_FLAT_CONTENTION_EXTRA_PROBE_BUDGET = 2
_EHA_FLAT_CONTENTION_MAX_PROTOTYPES = 4
_EHA_FLAT_CONTENTION_BEAM_WIDTH = 4
_EHA_POD_BUDGET = 8
_EHA_NODE_REFINE_BUDGET = 4
_EHA_DEFAULT_POD_SIZE_NODES = 4


@dataclass(frozen=True)
class _SearchUnit:
    unit_id: int
    capacity: int
    node_ids: Tuple[int, ...]


@dataclass(frozen=True)
class _AllocationPlan:
    allocation: Tuple[int, ...]
    priority: float
    source: str


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
    inference_cache: Optional['InferenceCache'] = None,
) -> float:
    """Compute bandwidth for a single config under current environment (cluster_manager / real / model)."""
    if inference_cache is not None:
        cached = inference_cache.get(config)
        if cached is not None:
            return cached

    if cluster_manager:
        bw = float(cluster_manager.predict_with_contention(config))
    elif if_real_data:
        real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
        bw_value, _, _ = calculate_bandwidth_values(
            config, total_gpu, gpu_bw_dict_list, switch_config, real_path
        )
        bw = float(bw_value)
    else:
        part_bws, node_counts, total_counts = prepare_model_inputs(
            np.array([config]), total_gpu, gpu_bw_dict_list, switch_config, training_data_path
        )
        prediction = predict_with_model(
            model, part_bws, node_counts, total_counts, device, artifact_dir
        )
        prediction_array = np.asarray(prediction)
        bw = float(prediction_array.reshape(-1)[0])

    if inference_cache is not None:
        inference_cache.put(config, bw)
    return bw


def _predict_config_bandwidth_batch(
    configs: Sequence[np.ndarray],
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
    inference_cache: Optional['InferenceCache'] = None,
) -> np.ndarray:
    """Batch version of _predict_config_bandwidth.

    For the model path, performs a single forward pass over all configs instead of
    N individual calls - the biggest latency win for large candidate sets.
    For cluster_manager, delegates to predict_with_contention_batch (loop wrapper).
    For real_data, delegates to config_to_bandwidth (batch table lookup).

    When inference_cache is provided, serves cache hits and only predicts misses.

    Args:
        configs: Sequence of 0/1 GPU combo vectors.
        (remaining args identical to _predict_config_bandwidth)

    Returns:
        1D numpy array of bandwidth values, one per config.
    """
    configs_array = np.asarray(configs)
    if configs_array.ndim == 1:
        configs_array = configs_array.reshape(1, -1)
    if len(configs_array) == 0:
        return np.array([], dtype=float)

    n = len(configs_array)

    # --- L1 cache: partition hits/misses ---
    if inference_cache is not None:
        hit_idx, miss_idx, hit_vals = inference_cache.get_batch(configs_array)
        if not miss_idx:
            # All hits
            results = np.empty(n, dtype=float)
            for i, idx in enumerate(hit_idx):
                results[idx] = hit_vals[i]
            return results
        miss_configs = configs_array[miss_idx]
    else:
        miss_idx = list(range(n))
        miss_configs = configs_array

    # --- Predict only the misses ---
    if cluster_manager:
        miss_bws = cluster_manager.predict_with_contention_batch(miss_configs)
    elif if_real_data:
        real_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)
        bw_array, _ = config_to_bandwidth(
            miss_configs, total_gpu, gpu_bw_dict_list, switch_config, real_path
        )
        miss_bws = bw_array.astype(float)
    else:
        part_bws, node_counts, total_counts = prepare_model_inputs(
            miss_configs, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
        )
        predictions = predict_with_model(
            model, part_bws, node_counts, total_counts, device, artifact_dir
        )
        miss_bws = np.asarray(predictions, dtype=float).reshape(-1)

    # --- Store misses in cache and assemble results ---
    if inference_cache is not None:
        inference_cache.put_batch(miss_configs, miss_bws)
        results = np.empty(n, dtype=float)
        for i, idx in enumerate(hit_idx):
            results[idx] = hit_vals[i]
        for i, idx in enumerate(miss_idx):
            results[idx] = miss_bws[i]
        return results

    return miss_bws


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
    inference_cache: Optional['InferenceCache'] = None,
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
        inference_cache=inference_cache,
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
            inference_cache=inference_cache,
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
    subproblem_cache: Optional['SubproblemCache'] = None,
    inference_cache: Optional['InferenceCache'] = None,
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

    # L2 cache lookup: same (node_gpus, target_count) -> same result
    if subproblem_cache is not None:
        cached = subproblem_cache.get(node_gpus, target_gpu_count)
        if cached is not None:
            return cached

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
        inference_cache=inference_cache,
    )

    # L2 cache store
    if subproblem_cache is not None and subset_config is not None:
        subproblem_cache.put(node_gpus, target_gpu_count, subset_config)

    return subset_config


def _run_parallel_subset_tree_searches(
    node_tasks: Sequence[Tuple[int, Sequence[int]]],
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
    global_mode_all: bool,
    inference_cache: Optional['InferenceCache'] = None,
) -> List[Optional[np.ndarray]]:
    """Run tree searches for multiple nodes in lockstep with merged batch inference.

    Each node task follows the same subset tree-search semantics as
    ``_run_subset_tree_search`` while sharing batched model inference across
    active frontiers. If there are N active node tasks and task i has D_i
    frontier candidates, each iteration evaluates ``sum(D_i)`` candidates in
    one batch while requiring only ``max(D_i)`` search-depth rounds.

    Args:
        node_tasks: Pairs of ``(node_id, node_gpus)``.
        target_gpu_count: Number of GPUs to retain per node task.
        Remaining parameters mirror ``_run_subset_tree_search``.

    Returns:
        Results aligned with *node_tasks*. Invalid tasks return ``None``.
    """
    from algorithms.search import generate_next_combos

    n_tasks = len(node_tasks)
    results: List[Optional[np.ndarray]] = [None] * n_tasks

    # Active tasks keep their task index and current candidate combo.
    active: List[Tuple[int, np.ndarray]] = []
    for i, (_node_id, node_gpus) in enumerate(node_tasks):
        if target_gpu_count <= 0:
            results[i] = np.zeros(num_dimensions, dtype=int)
        elif len(node_gpus) < target_gpu_count:
            results[i] = None
        elif target_gpu_count == len(node_gpus):
            cfg = np.zeros(num_dimensions, dtype=int)
            cfg[list(node_gpus)] = 1
            results[i] = cfg
        else:
            start = np.zeros(num_dimensions, dtype=int)
            start[list(node_gpus)] = 1
            active.append((i, start))

    if not active:
        return results

    # Iterate all active node tasks in lockstep so each depth shares one batch.
    while active:
        # Step 1: collect one-removal candidates for each active task.
        candidate_blocks: List[np.ndarray] = []
        # boundaries[j] = (orig_idx, slice_start, slice_end)
        boundaries: List[Tuple[int, int, int]] = []
        offset = 0

        for orig_idx, combo in active:
            cands = generate_next_combos(combo)  # (C_i, num_dimensions)
            candidate_blocks.append(cands)
            boundaries.append((orig_idx, offset, offset + len(cands)))
            offset += len(cands)

        if offset == 0:
            break

        # Step 2: evaluate all frontier candidates in one batch.
        candidate_array = np.concatenate(candidate_blocks, axis=0)
        scores = _predict_config_bandwidth_batch(
            candidate_array, model, total_gpu, gpu_bw_dict_list, switch_config,
            training_data_path, device, artifact_dir, if_real_data, cluster_manager,
            evaluation_data_path,
            inference_cache=inference_cache,
        )

        # In global mode, include the bandwidth of GPUs that remain available
        # after each candidate selection.
        if global_mode and cluster_manager and avail_gpu is not None and len(avail_gpu) > 0:
            avail_mask = np.zeros(num_dimensions, dtype=int)
            avail_mask[list(avail_gpu)] = 1
            # remaining = available AND NOT selected by the candidate.
            remaining_configs = avail_mask[np.newaxis, :] * (1 - candidate_array)
            remaining_bws = _predict_config_bandwidth_batch(
                remaining_configs, model, total_gpu, gpu_bw_dict_list, switch_config,
                training_data_path, device, artifact_dir, if_real_data, cluster_manager,
                evaluation_data_path,
                inference_cache=inference_cache,
            )
            scores = scores + remaining_bws
            if global_mode_all:
                scores = scores + cluster_manager.get_total_active_bandwidth()

        # 3. Keep the best local candidate from each active segment.
        next_active: List[Tuple[int, np.ndarray]] = []
        for orig_idx, s, e in boundaries:
            seg = scores[s:e]
            best_local = int(np.argmax(seg))
            best_combo = candidate_array[s + best_local].copy()

            if int(np.sum(best_combo)) == target_gpu_count:
                results[orig_idx] = best_combo
            else:
                next_active.append((orig_idx, best_combo))

        active = next_active

    return results


def _build_node_map(avail_gpu: Sequence[int]) -> Dict[int, List[int]]:
    """Group available GPUs by physical node (8 GPUs per node)."""
    node_map: Dict[int, List[int]] = {}
    for gpu_idx in avail_gpu:
        node_id = gpu_idx // _EHA_GPUS_PER_NODE
        node_map.setdefault(node_id, []).append(gpu_idx)
    return node_map


def _build_node_units(node_map: Dict[int, Sequence[int]]) -> List[_SearchUnit]:
    """Create node-level search units ordered by capacity then node id."""
    return sorted(
        [
            _SearchUnit(unit_id=node_id, capacity=len(gpus), node_ids=(node_id,))
            for node_id, gpus in node_map.items()
        ],
        key=lambda unit: (-unit.capacity, unit.unit_id),
    )


def _build_pod_units(
    node_units: Sequence[_SearchUnit],
    *,
    pod_size_nodes: int = _EHA_DEFAULT_POD_SIZE_NODES,
) -> List[_SearchUnit]:
    """Create pod-like units by grouping contiguous nodes."""
    pod_to_nodes: Dict[int, List[int]] = {}
    pod_to_capacity: Dict[int, int] = {}
    for unit in node_units:
        pod_id = unit.unit_id // pod_size_nodes
        pod_to_nodes.setdefault(pod_id, []).extend(unit.node_ids)
        pod_to_capacity[pod_id] = pod_to_capacity.get(pod_id, 0) + unit.capacity
    return sorted(
        [
            _SearchUnit(
                unit_id=pod_id,
                capacity=pod_to_capacity[pod_id],
                node_ids=tuple(sorted(pod_to_nodes[pod_id])),
            )
            for pod_id in pod_to_nodes
        ],
        key=lambda unit: (-unit.capacity, unit.unit_id),
    )


def _compute_k_min(capacities: Sequence[int], gpu_need: int) -> Optional[int]:
    """Return the smallest prefix length whose cumulative capacity meets gpu_need."""
    running = 0
    for idx, capacity in enumerate(capacities, start=1):
        running += int(capacity)
        if running >= gpu_need:
            return idx
    return None


def _is_flat_fast_path(total_gpu: int, num_nodes: int) -> bool:
    """Dual-mode gate: small scale stays flat, larger scale uses hierarchy."""
    return total_gpu <= 32 or num_nodes <= 4


def _balanced_prefix_allocation(
    capacities: Sequence[int],
    gpu_need: int,
    active_count: int,
) -> Optional[Tuple[int, ...]]:
    """Construct a balanced feasible allocation over the first active_count units."""
    if active_count <= 0 or active_count > len(capacities):
        return None
    if sum(capacities[:active_count]) < gpu_need:
        return None

    allocation = [0] * len(capacities)
    for _ in range(gpu_need):
        candidates = [
            idx
            for idx in range(active_count)
            if allocation[idx] < int(capacities[idx])
        ]
        if not candidates:
            return None
        best_idx = min(candidates, key=lambda idx: (allocation[idx], idx))
        allocation[best_idx] += 1
    return tuple(allocation)


def _residual_greedy_seed(capacities: Sequence[int], gpu_need: int) -> Optional[Tuple[int, ...]]:
    """Generate a locality-constrained residual greedy seed."""
    if sum(capacities) < gpu_need:
        return None

    allocation = [0] * len(capacities)
    remaining = [int(capacity) for capacity in capacities]
    for _ in range(gpu_need):
        candidates = [idx for idx, remain in enumerate(remaining) if remain > 0]
        if not candidates:
            return None
        best_idx = min(
            candidates,
            key=lambda idx: (
                allocation[idx] == 0,
                -remaining[idx],
                idx,
            ),
        )
        allocation[best_idx] += 1
        remaining[best_idx] -= 1
    return tuple(allocation)


def _allocation_imbalance(allocation: Sequence[int]) -> float:
    """Return L1 imbalance across active units."""
    active = [int(value) for value in allocation if int(value) > 0]
    if len(active) <= 1:
        return 0.0
    mean = sum(active) / len(active)
    return float(sum(abs(value - mean) for value in active))


def _allocation_span(allocation: Sequence[int]) -> int:
    """Return the positional span of active units."""
    active_indices = [idx for idx, value in enumerate(allocation) if int(value) > 0]
    if not active_indices:
        return 0
    return int(active_indices[-1] - active_indices[0])


def _allocation_priority(allocation: Sequence[int], capacities: Sequence[int]) -> float:
    """Rank canonical shapes by occupancy, active-unit count, imbalance and span."""
    active_indices = [idx for idx, value in enumerate(allocation) if int(value) > 0]
    if not active_indices:
        return float("-inf")

    densities = [
        float(allocation[idx]) / max(1, int(capacities[idx]))
        for idx in active_indices
    ]
    optimistic_bound = 100.0 * min(densities) + 10.0 * (sum(densities) / len(densities))
    active_unit_penalty = float(len(active_indices))
    imbalance_penalty = _allocation_imbalance(allocation)
    span_penalty = float(_allocation_span(allocation))
    return optimistic_bound - 4.0 * active_unit_penalty - 1.0 * imbalance_penalty - 0.25 * span_penalty


def _same_class_replacements(
    allocation: Sequence[int],
    capacities: Sequence[int],
    active_prefix: int,
) -> List[Tuple[int, ...]]:
    """Swap an active unit with an inactive unit of the same capacity."""
    variants: List[Tuple[int, ...]] = []
    for active_idx in range(active_prefix):
        if int(allocation[active_idx]) <= 0:
            continue
        for inactive_idx in range(active_prefix, len(capacities)):
            if capacities[inactive_idx] != capacities[active_idx]:
                continue
            variant = list(allocation)
            variant[inactive_idx] = variant[active_idx]
            variant[active_idx] = 0
            variants.append(tuple(variant))
    return variants


def _build_canonical_frontier(
    capacities: Sequence[int],
    gpu_need: int,
    active_count: int,
    *,
    budget: int,
    source_prefix: str,
    include_replacements: bool = True,
) -> List[_AllocationPlan]:
    """Build a canonical shape frontier around the balanced base allocation."""
    base = _balanced_prefix_allocation(capacities, gpu_need, active_count)
    if base is None:
        return []

    candidate_sources: Dict[Tuple[int, ...], str] = {base: f"{source_prefix}_base"}
    active_indices = [idx for idx, value in enumerate(base) if int(value) > 0]

    for donor_idx in active_indices:
        if int(base[donor_idx]) <= 0:
            continue
        for receiver_idx in active_indices:
            if donor_idx == receiver_idx or int(base[receiver_idx]) >= int(capacities[receiver_idx]):
                continue
            variant = list(base)
            variant[donor_idx] -= 1
            variant[receiver_idx] += 1
            if variant[donor_idx] < 0:
                continue
            candidate_sources.setdefault(tuple(variant), f"{source_prefix}_move1")

    if include_replacements:
        for variant in _same_class_replacements(base, capacities, active_count):
            candidate_sources.setdefault(variant, f"{source_prefix}_replace")

    ordered_allocations = sorted(
        candidate_sources,
        key=lambda allocation: (-_allocation_priority(allocation, capacities), allocation),
    )
    return [
        _AllocationPlan(
            allocation=allocation,
            priority=_allocation_priority(allocation, capacities),
            source=candidate_sources[allocation],
        )
        for allocation in ordered_allocations[:budget]
    ]


def _should_probe_extra_k_small(
    gpu_need: int,
    k_min_allocation: Sequence[int],
    units: Sequence[_SearchUnit],
    k_min: int,
) -> bool:
    """Gate the low-budget k_min+1 probe on the 32-GPU flat path."""
    if gpu_need % _EHA_GPUS_PER_NODE != 0 or gpu_need < 16 or k_min >= len(units):
        return False
    if not any(
        int(k_min_allocation[idx]) == int(units[idx].capacity)
        for idx in range(k_min)
        if int(k_min_allocation[idx]) > 0
    ):
        return False
    candidate_nodes: List[int] = []
    for idx in range(k_min + 1):
        candidate_nodes.extend(units[idx].node_ids)
    return bool(candidate_nodes) and (max(candidate_nodes) - min(candidate_nodes) < _EHA_DEFAULT_POD_SIZE_NODES)


def _unique_plans_by_allocation(plans: Sequence[_AllocationPlan]) -> List[_AllocationPlan]:
    """Keep the first plan for each unique allocation."""
    seen: set[Tuple[int, ...]] = set()
    deduped: List[_AllocationPlan] = []
    for plan in plans:
        if plan.allocation in seen:
            continue
        seen.add(plan.allocation)
        deduped.append(plan)
    return deduped


def _build_flat_phase2_plans(
    units: Sequence[_SearchUnit],
    gpu_need: int,
    *,
    max_candidates: int,
    contention_aware: bool = False,
) -> Tuple[List[_AllocationPlan], Dict[str, Any]]:
    """Build the flat fast-path plans for small-scale EHA Phase 2."""
    capacities = [unit.capacity for unit in units]
    k_min = _compute_k_min(capacities, gpu_need)
    flat_budget = _EHA_FLAT_CONTENTION_BUDGET if contention_aware else _EHA_FLAT_BUDGET
    extra_probe_budget = (
        _EHA_FLAT_CONTENTION_EXTRA_PROBE_BUDGET
        if contention_aware
        else _EHA_FLAT_EXTRA_PROBE_BUDGET
    )
    if k_min is None:
        return [], {
            "phase2_mode": "flat",
            "hierarchical_path": False,
            "contention_aware_flat": contention_aware,
            "k_values": [],
            "seed_plan_count": 0,
            "candidate_plan_count": 0,
            "estimated_subset_calls": 0,
            "kplus1_probe_count": 0,
        }

    base_allocation = _balanced_prefix_allocation(capacities, gpu_need, k_min)
    if base_allocation is None:
        return [], {
            "phase2_mode": "flat",
            "hierarchical_path": False,
            "contention_aware_flat": contention_aware,
            "k_values": [],
            "seed_plan_count": 0,
            "candidate_plan_count": 0,
            "estimated_subset_calls": 0,
            "kplus1_probe_count": 0,
        }

    plans: List[_AllocationPlan] = []
    seed_allocation = _residual_greedy_seed(capacities, gpu_need)
    if seed_allocation is not None:
        plans.append(
            _AllocationPlan(
                allocation=seed_allocation,
                priority=_allocation_priority(seed_allocation, capacities),
                source="residual_greedy_seed",
            )
        )

    plans.extend(
        _build_canonical_frontier(
            capacities,
            gpu_need,
            k_min,
            budget=flat_budget,
            source_prefix="flat_frontier",
        )
    )

    k_values = [k_min]
    kplus1_probe_count = 0
    should_probe_kplus1 = (
        k_min < len(units)
        and (
            contention_aware
            or _should_probe_extra_k_small(gpu_need, base_allocation, units, k_min)
        )
    )
    if should_probe_kplus1:
        extra_plans = _build_canonical_frontier(
            capacities,
            gpu_need,
            k_min + 1,
            budget=extra_probe_budget,
            source_prefix="flat_kplus1_probe",
            include_replacements=False,
        )
        if extra_plans:
            plans.extend(extra_plans)
            k_values.append(k_min + 1)
            kplus1_probe_count = len(extra_plans)

    plans = _unique_plans_by_allocation(plans)[:max_candidates]
    estimated_subset_calls = int(
        sum(sum(1 for value in plan.allocation if int(value) > 0) for plan in plans)
    )
    meta = {
        "phase2_mode": "flat",
        "hierarchical_path": False,
        "contention_aware_flat": contention_aware,
        "k_values": k_values,
        "seed_plan_count": int(sum(plan.source == "residual_greedy_seed" for plan in plans)),
        "candidate_plan_count": int(len(plans)),
        "estimated_subset_calls": estimated_subset_calls,
        "kplus1_probe_count": int(kplus1_probe_count),
    }
    return plans, meta


def _build_hierarchical_phase2_plans(
    node_units: Sequence[_SearchUnit],
    gpu_need: int,
    *,
    max_candidates: int,
) -> Tuple[List[_AllocationPlan], Dict[str, Any], List[_SearchUnit]]:
    """Build pod-level plans for large-scale EHA Phase 2."""
    pod_units = _build_pod_units(node_units)
    capacities = [unit.capacity for unit in pod_units]
    k_min = _compute_k_min(capacities, gpu_need)
    if k_min is None:
        empty_meta = {
            "phase2_mode": "hierarchical",
            "hierarchical_path": True,
            "k_values": [],
            "pod_candidate_plan_count": 0,
            "seed_plan_count": 0,
            "estimated_subset_calls": 0,
            "kplus1_probe_count": 0,
        }
        return [], empty_meta, list(pod_units)

    plans: List[_AllocationPlan] = []
    seed_allocation = _residual_greedy_seed(capacities, gpu_need)
    if seed_allocation is not None:
        plans.append(
            _AllocationPlan(
                allocation=seed_allocation,
                priority=_allocation_priority(seed_allocation, capacities),
                source="residual_greedy_seed",
            )
        )

    plans.extend(
        _build_canonical_frontier(
            capacities,
            gpu_need,
            k_min,
            budget=_EHA_POD_BUDGET,
            source_prefix="pod_frontier",
        )
    )
    k_values = [k_min]
    kplus1_probe_count = 0
    if k_min < len(pod_units):
        extra_plans = _build_canonical_frontier(
            capacities,
            gpu_need,
            k_min + 1,
            budget=min(2, _EHA_POD_BUDGET),
            source_prefix="pod_frontier_kplus1",
            include_replacements=False,
        )
        if extra_plans:
            plans.extend(extra_plans)
            k_values.append(k_min + 1)
            kplus1_probe_count = len(extra_plans)

    plans = _unique_plans_by_allocation(plans)[:max_candidates]
    estimated_subset_calls = int(
        sum(sum(1 for value in plan.allocation if int(value) > 0) for plan in plans)
    )
    meta = {
        "phase2_mode": "hierarchical",
        "hierarchical_path": True,
        "k_values": k_values,
        "pod_candidate_plan_count": int(len(plans)),
        "seed_plan_count": int(sum(plan.source == "residual_greedy_seed" for plan in plans)),
        "estimated_subset_calls": estimated_subset_calls,
        "kplus1_probe_count": int(kplus1_probe_count),
    }
    return plans, meta, list(pod_units)


def _get_local_bw_dict(
    gpu_bw_dict_list,
    node_id: int,
) -> Dict[Tuple[int, ...], float]:
    """Return the per-node bandwidth dictionary if available."""
    if not gpu_bw_dict_list:
        return {}
    if 0 <= node_id < len(gpu_bw_dict_list):
        bw_dict = gpu_bw_dict_list[node_id]
    else:
        bw_dict = gpu_bw_dict_list[node_id % len(gpu_bw_dict_list)]
    return bw_dict if isinstance(bw_dict, dict) else {}


def _prototype_budget_from_rel_range(rel_range: float, max_prototypes: int) -> int:
    """Map subset sensitivity to the retained prototype count."""
    if max_prototypes <= 1:
        return 1
    if rel_range <= 0.05:
        return int(np.ceil(max_prototypes/4))
    if rel_range <= 0.30:
        return int(np.ceil(max_prototypes/2))
    return max_prototypes


def _enumerate_local_prototypes(
    node_id: int,
    node_gpus: Sequence[int],
    target_gpu_count: int,
    *,
    num_dimensions: int,
    gpu_bw_dict_list,
    max_prototypes: int,
) -> List[Tuple[np.ndarray, float]]:
    """Enumerate a small set of representative local subsets from real-data dictionaries."""
    if target_gpu_count <= 0:
        return [(np.zeros(num_dimensions, dtype=int), 0.0)]
    if len(node_gpus) < target_gpu_count:
        return []

    if target_gpu_count == len(node_gpus):
        config = np.zeros(num_dimensions, dtype=int)
        config[list(node_gpus)] = 1
        return [(config, float(target_gpu_count))]

    local_bw_dict = _get_local_bw_dict(gpu_bw_dict_list, node_id)
    if not local_bw_dict:
        return []

    available_slots = {gpu_idx % _EHA_GPUS_PER_NODE for gpu_idx in node_gpus}
    scored_patterns: List[Tuple[float, Tuple[int, ...]]] = []
    for mask, bandwidth in local_bw_dict.items():
        if sum(mask) != target_gpu_count:
            continue
        active_slots = {idx for idx, value in enumerate(mask) if int(value) == 1}
        if active_slots.issubset(available_slots):
            scored_patterns.append((float(bandwidth), tuple(int(value) for value in mask)))

    if not scored_patterns:
        return []

    bandwidths = [score for score, _ in scored_patterns]
    max_bw = max(bandwidths)
    min_bw = min(bandwidths)
    rel_range = (max_bw - min_bw) / max(max_bw, _EHA_EPS)
    keep = _prototype_budget_from_rel_range(rel_range, max_prototypes)
    scored_patterns.sort(key=lambda item: (-item[0], item[1]))

    prototypes: List[Tuple[np.ndarray, float]] = []
    seen_configs: set[Tuple[int, ...]] = set()
    for score, mask in scored_patterns:
        config = np.zeros(num_dimensions, dtype=int)
        for local_slot, bit in enumerate(mask):
            if bit:
                config[node_id * _EHA_GPUS_PER_NODE + local_slot] = 1
        config_key = tuple(int(value) for value in config.tolist())
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        prototypes.append((config, score))
        if len(prototypes) >= keep:
            break
    return prototypes


def _realize_node_plan(
    plan: _AllocationPlan,
    units: Sequence[_SearchUnit],
    node_map: Dict[int, Sequence[int]],
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
    global_mode_all: bool,
    subproblem_cache: Optional['SubproblemCache'],
    inference_cache: Optional['InferenceCache'],
    max_prototypes: int,
    beam_width: int,
    include_tree_search_winner: bool = False,
    runtime_rescore_local_candidates: bool = False,
) -> List[np.ndarray]:
    """Instantiate a node-level allocation plan into a small config beam."""
    partials: List[Tuple[np.ndarray, float]] = [(np.zeros(num_dimensions, dtype=int), plan.priority)]
    for unit_idx, alloc_count in enumerate(plan.allocation):
        if int(alloc_count) <= 0:
            continue
        node_id = units[unit_idx].unit_id
        node_gpus = node_map[node_id]
        local_candidates = _enumerate_local_prototypes(
            node_id,
            node_gpus,
            int(alloc_count),
            num_dimensions=num_dimensions,
            gpu_bw_dict_list=gpu_bw_dict_list,
            max_prototypes=max_prototypes,
        )
        subset_config: Optional[np.ndarray] = None
        if not local_candidates or include_tree_search_winner:
            subset_config = _run_subset_tree_search(
                node_id,
                node_gpus,
                int(alloc_count),
                num_dimensions=num_dimensions,
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
                avail_gpu=None,
                allow_global_mode=False,
                global_mode_all=global_mode_all,
                evaluation_data_path=evaluation_data_path,
                subproblem_cache=subproblem_cache,
                inference_cache=inference_cache,
            )
        if not local_candidates:
            if subset_config is None:
                return []
            local_candidates = [(subset_config, 0.0)]
        elif include_tree_search_winner and subset_config is not None:
            subset_key = tuple(int(value) for value in subset_config.tolist())
            local_keys = {
                tuple(int(value) for value in config.tolist())
                for config, _ in local_candidates
            }
            if subset_key not in local_keys:
                local_candidates.append((subset_config, 0.0))

        if runtime_rescore_local_candidates:
            rescored_candidates: List[Tuple[np.ndarray, float]] = []
            for local_config, _ in local_candidates:
                runtime_score = _predict_config_bandwidth(
                    local_config,
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
                    inference_cache=inference_cache,
                )
                rescored_candidates.append((local_config, float(runtime_score)))
            local_candidates = rescored_candidates

        next_partials: List[Tuple[np.ndarray, float]] = []
        for partial_config, partial_score in partials:
            for local_config, local_score in local_candidates:
                next_partials.append(
                    (np.maximum(partial_config, local_config), partial_score + float(local_score))
                )

        deduped: Dict[Tuple[int, ...], Tuple[np.ndarray, float]] = {}
        for config, score in next_partials:
            key = tuple(int(value) for value in config.tolist())
            existing = deduped.get(key)
            if existing is None or score > existing[1]:
                deduped[key] = (config, score)
        partials = sorted(
            deduped.values(),
            key=lambda item: -item[1],
        )[:beam_width]

    return [config for config, _ in partials]


def _build_local_node_plans(
    pod_node_units: Sequence[_SearchUnit],
    gpu_need: int,
) -> List[_AllocationPlan]:
    """Build node-level canonical plans within a pod-like unit."""
    capacities = [unit.capacity for unit in pod_node_units]
    k_min = _compute_k_min(capacities, gpu_need)
    if k_min is None:
        return []

    plans = _build_canonical_frontier(
        capacities,
        gpu_need,
        k_min,
        budget=_EHA_NODE_REFINE_BUDGET,
        source_prefix="pod_node_frontier",
    )
    if k_min < len(pod_node_units):
        plans.extend(
            _build_canonical_frontier(
                capacities,
                gpu_need,
                k_min + 1,
                budget=1,
                source_prefix="pod_node_kplus1",
                include_replacements=False,
            )
        )
    return _unique_plans_by_allocation(plans)[:_EHA_NODE_REFINE_BUDGET]


def _realize_hierarchical_plan(
    plan: _AllocationPlan,
    pod_units: Sequence[_SearchUnit],
    node_map: Dict[int, Sequence[int]],
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
    global_mode_all: bool,
    subproblem_cache: Optional['SubproblemCache'],
    inference_cache: Optional['InferenceCache'],
) -> List[np.ndarray]:
    """Instantiate a pod-level plan via node-level canonical refinement."""
    partials: List[Tuple[np.ndarray, float]] = [(np.zeros(num_dimensions, dtype=int), plan.priority)]
    for pod_idx, pod_gpu_need in enumerate(plan.allocation):
        if int(pod_gpu_need) <= 0:
            continue

        pod_node_units = sorted(
            [
                _SearchUnit(unit_id=node_id, capacity=len(node_map[node_id]), node_ids=(node_id,))
                for node_id in pod_units[pod_idx].node_ids
            ],
            key=lambda unit: (-unit.capacity, unit.unit_id),
        )
        local_plans = _build_local_node_plans(pod_node_units, int(pod_gpu_need))
        if not local_plans:
            return []

        local_configs: List[Tuple[np.ndarray, float]] = []
        for local_plan in local_plans:
            realized = _realize_node_plan(
                local_plan,
                pod_node_units,
                node_map,
                num_dimensions=num_dimensions,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=training_data_path,
                device=device,
                artifact_dir=artifact_dir,
                evaluation_data_path=evaluation_data_path,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                global_mode=global_mode,
                global_mode_all=global_mode_all,
                subproblem_cache=subproblem_cache,
                inference_cache=inference_cache,
                max_prototypes=_EHA_NODE_REFINE_BUDGET,
                beam_width=_EHA_NODE_REFINE_BUDGET,
            )
            for config in realized:
                local_configs.append((config, local_plan.priority))

        deduped_local: Dict[Tuple[int, ...], Tuple[np.ndarray, float]] = {}
        for config, score in local_configs:
            key = tuple(int(value) for value in config.tolist())
            existing = deduped_local.get(key)
            if existing is None or score > existing[1]:
                deduped_local[key] = (config, score)
        ranked_local = sorted(
            deduped_local.values(),
            key=lambda item: -item[1],
        )[:_EHA_NODE_REFINE_BUDGET]
        if not ranked_local:
            return []

        next_partials: List[Tuple[np.ndarray, float]] = []
        for partial_config, partial_score in partials:
            for local_config, local_score in ranked_local:
                next_partials.append(
                    (np.maximum(partial_config, local_config), partial_score + float(local_score))
                )

        deduped_partials: Dict[Tuple[int, ...], Tuple[np.ndarray, float]] = {}
        for config, score in next_partials:
            key = tuple(int(value) for value in config.tolist())
            existing = deduped_partials.get(key)
            if existing is None or score > existing[1]:
                deduped_partials[key] = (config, score)
        partials = sorted(
            deduped_partials.values(),
            key=lambda item: -item[1],
        )[:_EHA_NODE_REFINE_BUDGET]

    return [config for config, _ in partials]


def analyze_legacy_phase2_budget(
    node_capacities: Sequence[int],
    gpu_need: int,
    *,
    max_candidates: int = 200,
) -> Dict[str, Any]:
    """Estimate the old Phase 2 budget from combinations plus permutation variants."""
    capacities = sorted([int(capacity) for capacity in node_capacities], reverse=True)
    k_min = _compute_k_min(capacities, gpu_need)
    if k_min is None:
        return {"k_min": None, "candidate_count": 0, "estimated_subset_calls": 0}

    candidate_count = 0
    subset_calls = 0
    for indices in itertools.combinations(range(len(capacities)), k_min):
        group_counts = [capacities[idx] for idx in indices]
        if sum(group_counts) < gpu_need:
            continue

        if candidate_count <= max_candidates:
            allocation = [0] * k_min
            remaining = list(group_counts)
            feasible = True
            for _ in range(gpu_need):
                best_idx = max(range(k_min), key=lambda idx: (remaining[idx], -idx))
                if remaining[best_idx] <= 0:
                    feasible = False
                    break
                allocation[best_idx] += 1
                remaining[best_idx] -= 1
            if feasible:
                candidate_count += 1
                subset_calls += sum(1 for value in allocation if value > 0)

        base_alloc = [gpu_need // k_min] * k_min
        for idx in range(gpu_need % k_min):
            base_alloc[idx] += 1

        allocation_variants: set[Tuple[int, ...]] = set()
        for perm in set(itertools.permutations(base_alloc)):
            allocation_variants.add(tuple(int(value) for value in perm))
        for delta in (1, 2, 3, 4):
            for i in range(k_min):
                for j in range(k_min):
                    if i == j:
                        continue
                    variant = list(base_alloc)
                    variant[i] += delta
                    variant[j] -= delta
                    if sum(variant) == gpu_need and all(value >= 0 for value in variant):
                        for perm in set(itertools.permutations(variant)):
                            allocation_variants.add(tuple(int(value) for value in perm))

        for variant in allocation_variants:
            if all(group_counts[idx] >= variant[idx] for idx in range(k_min)):
                candidate_count += 1
                subset_calls += sum(1 for value in variant if value > 0)
                if candidate_count >= max_candidates:
                    break
        if candidate_count >= max_candidates:
            break

    return {
        "k_min": int(k_min),
        "candidate_count": int(min(candidate_count, max_candidates)),
        "estimated_subset_calls": int(subset_calls),
    }


def analyze_eha_search_budget(
    node_capacities: Sequence[int],
    gpu_need: int,
    total_gpu: int,
    *,
    max_candidates: int = 200,
) -> Dict[str, Any]:
    """Estimate the new Phase 2 planning budget without running the predictor."""
    node_map = {
        node_id: list(range(node_id * _EHA_GPUS_PER_NODE, node_id * _EHA_GPUS_PER_NODE + int(capacity)))
        for node_id, capacity in enumerate(node_capacities)
        if int(capacity) > 0
    }
    node_units = _build_node_units(node_map)
    num_nodes = len(node_units)
    if _is_flat_fast_path(total_gpu, num_nodes):
        plans, meta = _build_flat_phase2_plans(
            node_units,
            gpu_need,
            max_candidates=max_candidates,
        )
    else:
        plans, meta, _ = _build_hierarchical_phase2_plans(
            node_units,
            gpu_need,
            max_candidates=max_candidates,
        )
    meta = dict(meta)
    meta["candidate_plan_count"] = int(len(plans))
    meta["estimated_subset_calls"] = int(
        sum(sum(1 for value in plan.allocation if int(value) > 0) for plan in plans)
    )
    return meta


def _empty_confidence_meta() -> Dict[str, Any]:
    """Return a default EHA confidence payload for infeasible searches."""
    return {
        "num_candidates": 0,
        "bw_list": [],
        "best_bw": 0.0,
        "bw_cv": 0.0,
        "top5_gap": 0.0,
        "node_count": 0,
        "min_node_density": 0,
        "phase2_mode": "",
        "hierarchical_path": False,
        "k_values": [],
        "candidate_plan_count": 0,
        "estimated_subset_calls": 0,
        "kplus1_probe_count": 0,
    }


def _summarize_config_density(config: np.ndarray) -> Tuple[int, int]:
    """Return active node count and minimum active-node density for a config."""
    reshaped = np.asarray(config, dtype=int).reshape(-1, 8)
    node_counts = reshaped.sum(axis=1)
    active_counts = node_counts[node_counts > 0]
    if active_counts.size == 0:
        return 0, 0
    return int(active_counts.size), int(active_counts.min())


def _build_confidence_meta(
    candidate_configs: Sequence[np.ndarray],
    bw_list: Sequence[float],
    best_idx: int,
) -> Dict[str, Any]:
    """Construct EHA confidence statistics from evaluated candidates."""
    if best_idx < 0 or not candidate_configs:
        return _empty_confidence_meta()

    bw_array = np.asarray(bw_list, dtype=float)
    best_bw = float(bw_array[best_idx]) if bw_array.size else 0.0
    if bw_array.size == 0:
        bw_cv = 0.0
        top5_gap = 0.0
    else:
        bw_cv = float(np.std(bw_array) / (best_bw + _EHA_EPS))
        top_k = min(5, bw_array.size)
        top_values = np.sort(bw_array)[-top_k:]
        top5_gap = (
            float((top_values[-1] - top_values[0]) / (top_values[-1] + _EHA_EPS))
            if top_values.size > 1
            else 0.0
        )

    node_count, min_node_density = _summarize_config_density(candidate_configs[best_idx])
    return {
        "num_candidates": int(len(candidate_configs)),
        "bw_list": [float(v) for v in bw_array.tolist()],
        "best_bw": best_bw,
        "bw_cv": bw_cv,
        "top5_gap": top5_gap,
        "node_count": node_count,
        "min_node_density": min_node_density,
    }


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
    return_confidence: bool = False,
    cache_bundle: Optional['DispatchCacheBundle'] = None,
) -> np.ndarray | Tuple[np.ndarray | None, Dict[str, Any]] | None:
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
         a) Strategy 1: remaining-resource balance - greedy: always give GPUs to node with most remaining GPUs.
         b) Strategy 2: balanced counts - evenly distribute with ±1/±2 variants.

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
        When ``return_confidence=True``, returns ``(best_combo, eha_meta)``.
    """
    model_data_path = training_data_path
    real_data_path = _select_data_path_for_mode(True, training_data_path, evaluation_data_path)

    # --- Dispatch-scoped cache bundle ---
    from algorithms.cache import DispatchCacheBundle
    owns_cache_bundle = cache_bundle is None
    if cache_bundle is None:
        cache_bundle = DispatchCacheBundle()
    _l1 = cache_bundle.l1
    _l2 = cache_bundle.l2

    # Attach C0 cache to cluster_manager if available
    if cluster_manager is not None:
        cluster_manager.set_super_combo_cache(cache_bundle.c0)

    def _cleanup_cache() -> None:
        if owns_cache_bundle and cluster_manager is not None:
            cluster_manager.clear_super_combo_cache()

    def _finalize(
        result: np.ndarray | None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray | Tuple[np.ndarray | None, Dict[str, Any]] | None:
        _cleanup_cache()
        if return_confidence:
            return result, meta if meta is not None else _empty_confidence_meta()
        return result

    def _select_best_candidate(
        candidates: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray | None, Dict[str, Any]]:
        if not candidates:
            return None, _empty_confidence_meta()

        # Batch predict bandwidth for all candidates in a single call
        current_bws = _predict_config_bandwidth_batch(
            candidates, model, total_gpu, gpu_bw_dict_list, switch_config,
            model_data_path, device, artifact_dir, if_real_data, cluster_manager,
            real_data_path,
            inference_cache=_l1,
        )

        if global_mode and avail_gpu is not None and len(avail_gpu) > 0:
            # Global mode: also batch predict remaining bandwidth
            remaining_configs = []
            for config in candidates:
                selected_gpus = set(np.where(config == 1)[0])
                remaining_gpus = [gpu for gpu in avail_gpu if gpu not in selected_gpus]
                if remaining_gpus:
                    r_config = np.zeros(num_dimensions, dtype=int)
                    r_config[remaining_gpus] = 1
                    remaining_configs.append(r_config)
                else:
                    remaining_configs.append(np.zeros(num_dimensions, dtype=int))

            remaining_bws = _predict_config_bandwidth_batch(
                remaining_configs, model, total_gpu, gpu_bw_dict_list, switch_config,
                model_data_path, device, artifact_dir, if_real_data, cluster_manager,
                real_data_path,
                inference_cache=_l1,
            )
            bw_array = current_bws + remaining_bws
            if global_mode_all and cluster_manager:
                bw_array = bw_array + cluster_manager.get_total_active_bandwidth()
        else:
            bw_array = current_bws

        bw_list = [float(v) for v in bw_array.tolist()]
        # `UpperBandPilot` and real-data evaluation use raw-score selection.
        # Model-domain EHA applies the structural rerank before selecting.
        if if_real_data:
            ranking_array = bw_array
        else:
            ranking_array = apply_contention_structural_rerank(
                candidates,
                bw_array,
                cluster_manager=cluster_manager,
                switch_config=switch_config,
            )
        best_idx = int(np.argmax(ranking_array))
        meta = _build_confidence_meta(candidates, bw_list, best_idx)
        best_config = candidates[best_idx] if best_idx >= 0 else None
        return best_config, meta

    # ==================== Preprocessing: group available GPUs by node ====================
    node_map = _build_node_map(avail_gpu)
    node_units = _build_node_units(node_map)

    # ==================== Phase 1: single-node optimum search (highest priority) ====================
    # Find nodes that can individually satisfy gpu_need (best locality, no cross-node cost).
    candidate_nodes = [node_id for node_id, gpus in node_map.items() if len(gpus) >= gpu_need]
    candidate_configs: List[np.ndarray] = []

    if candidate_nodes:
        # Phase 1 batch path: keep node-local searches aligned with tree_search
        # while evaluating all candidate nodes in one batch.
        node_tasks = [(nid, node_map[nid]) for nid in candidate_nodes]
        search_results = _run_parallel_subset_tree_searches(
            node_tasks,
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
            global_mode_all=global_mode_all,
            evaluation_data_path=real_data_path,
            inference_cache=_l1,
        )
        candidate_configs = [cfg for cfg in search_results if cfg is not None]

        best_config, meta = _select_best_candidate(candidate_configs)
        return _finalize(best_config, meta)

    # ==================== Phase 2: dual-mode cross-node search ====================
    if sum(unit.capacity for unit in node_units) < gpu_need:
        return _finalize(None)

    if _is_flat_fast_path(total_gpu, len(node_units)):
        flat_contention_aware = is_contention_sensitive_mode(cluster_manager)
        phase2_plans, phase2_meta = _build_flat_phase2_plans(
            node_units,
            gpu_need,
            max_candidates=max_candidates,
            contention_aware=flat_contention_aware,
        )
        flat_max_prototypes = (
            _EHA_FLAT_CONTENTION_MAX_PROTOTYPES #if flat_contention_aware else 4
        )
        flat_beam_width = (
            _EHA_FLAT_CONTENTION_BEAM_WIDTH #if flat_contention_aware else 1
        )
        for plan in phase2_plans:
            realized = _realize_node_plan(
                plan,
                node_units,
                node_map,
                num_dimensions=num_dimensions,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=model_data_path,
                device=device,
                artifact_dir=artifact_dir,
                evaluation_data_path=real_data_path,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                global_mode=global_mode,
                global_mode_all=global_mode_all,
                subproblem_cache=_l2,
                inference_cache=_l1,
                max_prototypes=flat_max_prototypes,
                beam_width=flat_beam_width,
                include_tree_search_winner=flat_contention_aware,
                runtime_rescore_local_candidates=flat_contention_aware,
            )
            candidate_configs.extend(realized)
            if len(candidate_configs) >= max_candidates:
                candidate_configs = candidate_configs[:max_candidates]
                break
    else:
        phase2_plans, phase2_meta, pod_units = _build_hierarchical_phase2_plans(
            node_units,
            gpu_need,
            max_candidates=max_candidates,
        )
        for plan in phase2_plans:
            realized = _realize_hierarchical_plan(
                plan,
                pod_units,
                node_map,
                num_dimensions=num_dimensions,
                model=model,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=model_data_path,
                device=device,
                artifact_dir=artifact_dir,
                evaluation_data_path=real_data_path,
                if_real_data=if_real_data,
                cluster_manager=cluster_manager,
                global_mode=global_mode,
                global_mode_all=global_mode_all,
                subproblem_cache=_l2,
                inference_cache=_l1,
            )
            candidate_configs.extend(realized)
            if len(candidate_configs) >= max_candidates:
                candidate_configs = candidate_configs[:max_candidates]
                break

    if candidate_configs:
        deduped_candidates: Dict[Tuple[int, ...], np.ndarray] = {}
        for config in candidate_configs:
            key = tuple(int(value) for value in config.tolist())
            deduped_candidates.setdefault(key, config)
        candidate_configs = list(deduped_candidates.values())[:max_candidates]

    # ==================== Final evaluation and selection ====================
    # If no candidates, return None
    if not candidate_configs:
        return _finalize(None)

    best_config, meta = _select_best_candidate(candidate_configs)
    meta.update(phase2_meta)

    # Log cache stats and clean up
    cache_stats = cache_bundle.stats()
    logger.debug("EHA cache stats: %s", cache_stats)
    return _finalize(best_config, meta)


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
         a) Strategy 1: remaining-resource balance - greedy, give GPUs to node with most remaining.
         b) Strategy 2: balanced counts - even distribution with all permutations.
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
