"""Evaluation metrics and upper bound estimation."""
from __future__ import annotations

import ast
import logging
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.bandwidth import BandwidthLookupCache
from core.cluster_state import ClusterStateManager

logger = logging.getLogger(__name__)


def _prepare_node_resources(
    total_gpu: int,
    avail_gpu: Sequence[int],
    gpu_bw_dict_list,
    node_size: int = 8,
    top_k_per_node: int = 1,
) -> Tuple[
    int,
    List[int],
    List[int],
    Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]],
    Dict[Tuple[int, int], List[Tuple[float, Tuple[int, ...]]]],
]:
    """Precompute intra-node combo resources from available GPUs."""
    if total_gpu % node_size != 0:
        raise ValueError(f"total_gpu must be a multiple of {node_size}")
    if len(avail_gpu) == 0:
        raise ValueError("avail_gpu is empty; cannot build node resources")

    num_machines = total_gpu // node_size
    if len(gpu_bw_dict_list) < num_machines:
        raise ValueError(
            "gpu_bw_dict_list too short: "
            f"requires {num_machines} node dicts, got {len(gpu_bw_dict_list)}"
        )

    avail_gpu_sorted = sorted(int(gpu) for gpu in avail_gpu)
    avail_local_indices: List[List[int]] = [[] for _ in range(num_machines)]
    for gpu_idx in avail_gpu_sorted:
        node_idx = gpu_idx // node_size
        local_idx = gpu_idx % node_size
        if 0 <= node_idx < num_machines:
            avail_local_indices[node_idx].append(local_idx)
    best_intra_configs: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]] = {}
    candidate_map: Dict[Tuple[int, int], List[Tuple[float, Tuple[int, ...]]]] = {}
    for node_idx in range(num_machines):
        local_avail = avail_local_indices[node_idx]
        if not local_avail:
            continue

        # Single-GPU candidates: treat each available physical GPU as a standalone candidate,
        # which facilitates constructing more diverse cross-node combinations.
        single_candidates: List[Tuple[float, Tuple[int, ...]]] = []
        for local_idx in local_avail:
            single_mask = [0] * node_size
            single_mask[local_idx] = 1
            single_candidates.append((float("inf"), tuple(single_mask)))
        if single_candidates:
            candidate_map[(node_idx, 1)] = single_candidates[:top_k_per_node]
            best_intra_configs[(node_idx, 1)] = candidate_map[(node_idx, 1)][0]

        node_dict = gpu_bw_dict_list[node_idx]
        for gpu_cnt in range(2, len(local_avail) + 1):
            candidates: List[Tuple[float, Tuple[int, ...]]] = []
            for combo in itertools.combinations(local_avail, gpu_cnt):
                mask = [0] * node_size
                for local_idx in combo:
                    mask[local_idx] = 1
                mask_tuple = tuple(mask)
                bw_value = float(node_dict.get(mask_tuple, 0.0))
                candidates.append((max(bw_value, 0.0), mask_tuple))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                candidate_map[(node_idx, gpu_cnt)] = candidates[:top_k_per_node]
                best_intra_configs[(node_idx, gpu_cnt)] = candidate_map[(node_idx, gpu_cnt)][0]

    avail_capacity = [len(local_list) for local_list in avail_local_indices]
    suffix_capacity = [0] * (num_machines + 1)
    for idx in range(num_machines - 1, -1, -1):
        suffix_capacity[idx] = suffix_capacity[idx + 1] + avail_capacity[idx]

    return num_machines, avail_capacity, suffix_capacity, best_intra_configs, candidate_map


def _build_combo_from_distribution(
    distribution: Tuple[int, ...],
    best_intra_configs: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]],
    total_gpu: int,
    node_size: int = 8,
) -> List[int]:
    """Convert a node-level distribution into a concrete GPU combo."""
    combo = [0] * total_gpu
    for node_idx, gpu_cnt in enumerate(distribution):
        if gpu_cnt == 0:
            continue
        intra = best_intra_configs.get((node_idx, gpu_cnt))
        if intra is None:
            return []
        _, mask = intra
        base = node_idx * node_size
        for local_idx, flag in enumerate(mask):
            if flag:
                combo[base + local_idx] = 1
    return combo


def _generate_distributions(
    num_machines: int,
    target_gpu: int,
    avail_capacity: List[int],
    suffix_capacity: List[int],
    best_intra_configs: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]],
    node_size: int = 8,
):
    """Generate all feasible node distributions."""

    def _backtrack(machine_idx: int, remaining: int, current: List[int]):
        if remaining < 0:
            return
        if machine_idx == num_machines:
            if remaining == 0:
                yield tuple(current)
            return
        if remaining > suffix_capacity[machine_idx]:
            return
        max_take = min(node_size, avail_capacity[machine_idx], remaining)
        for cnt in range(max_take, -1, -1):
            if cnt > 0 and (machine_idx, cnt) not in best_intra_configs:
                continue
            current.append(cnt)
            yield from _backtrack(machine_idx + 1, remaining - cnt, current)
            current.pop()

    yield from _backtrack(0, target_gpu, [])


def _enumerate_candidate_combos(
    distribution: Tuple[int, ...],
    candidate_map: Dict[Tuple[int, int], List[Tuple[float, Tuple[int, ...]]]],
    total_gpu: int,
    node_size: int,
    local_top_k: int,
    max_candidates: Optional[int],
) -> List[np.ndarray]:
    """Given a distribution, combine per-node top-k candidates into global combinations."""
    nodes: List[Tuple[int, List[Tuple[float, Tuple[int, ...]]]]] = []
    for node_idx, gpu_cnt in enumerate(distribution):
        if gpu_cnt == 0:
            continue
        candidates = candidate_map.get((node_idx, gpu_cnt), [])
        if not candidates:
            return []
        nodes.append((node_idx, candidates[: max(local_top_k, 1)]))

    combos: List[np.ndarray] = []
    buffer = np.zeros(total_gpu, dtype=int)

    def _dfs(node_pos: int) -> None:
        if max_candidates is not None and len(combos) >= max_candidates:
            return
        if node_pos == len(nodes):
            combos.append(buffer.copy())
            return
        node_idx, candidates = nodes[node_pos]
        base = node_idx * node_size
        for _, mask in candidates:
            for local_idx, flag in enumerate(mask):
                if flag:
                    buffer[base + local_idx] = 1
            _dfs(node_pos + 1)
            for local_idx, flag in enumerate(mask):
                if flag:
                    buffer[base + local_idx] = 0
            if max_candidates is not None and len(combos) >= max_candidates:
                return

    _dfs(0)
    return combos


def _estimate_distribution_upper_bound(
    distribution: Tuple[int, ...],
    k: int,
    best_intra_configs: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]],
    cross_lookup: Optional[Dict[Tuple[int, int, Tuple[int, ...]], float]] = None,
) -> float:
    """Estimate an upper bound based on intra-node best and cross-node lookup (for sorting/pruning)."""
    active_counts = [cnt for cnt in distribution if cnt > 0]
    if not active_counts:
        return 0.0

    cross_bw = float("inf")
    if len(active_counts) == 1:
        cross_bw = float("inf")
    elif cross_lookup is not None:
        key = (k, len(active_counts), tuple(sorted(active_counts)))
        cross_bw = cross_lookup.get(key, 0.0)
        if cross_bw <= 0:
            return 0.0

    intra_bw = float("inf")
    for node_idx, gpu_cnt in enumerate(distribution):
        if gpu_cnt == 0:
            continue
        intra_entry = best_intra_configs.get((node_idx, gpu_cnt))
        if intra_entry is None:
            return 0.0
        if gpu_cnt >= 2 and intra_entry[0] < intra_bw:
            intra_bw = intra_entry[0]

    if intra_bw == float("inf"):
        return cross_bw
    return min(cross_bw, intra_bw)


def _build_cross_lookup(data_path: Optional[str]) -> Optional[Dict[Tuple[int, int, Tuple[int, ...]], float]]:
    """Build cross-node upper-bound lookup for distribution upper-bound estimation."""
    if not data_path:
        return None
    lookup = BandwidthLookupCache.ensure_loaded(data_path)
    cross_lookup: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
    for key, records in lookup.items():
        max_bw = max(float(bw) for _, bw in records)
        cross_lookup[key] = max_bw
    return cross_lookup


def find_max_bw_for_k_gpus(
    k: int,
    gpu_bw_dict_list,
    total_gpu: int,
    switch_config,
    avail_gpu: Sequence[int],
    data_path: str,
) -> Tuple[float, List[int]]:
    """Estimate max bandwidth under avail_gpu constraint via lookup tables."""
    if len(avail_gpu) == 0:
        logger.warning("find_max_bw_for_k_gpus: avail_gpu empty, return 0 bandwidth.")
        return 0.0, []

    if not (1 <= k <= len(avail_gpu)):
        logger.warning(
            "find_max_bw_for_k_gpus: k=%s exceeds available GPUs %s, return 0 bandwidth.",
            k,
            len(avail_gpu),
        )
        return 0.0, []

    try:
        (
            num_machines,
            avail_capacity,
            suffix_capacity,
            best_intra_configs,
            _candidate_map,
        ) = _prepare_node_resources(
            total_gpu=total_gpu,
            avail_gpu=avail_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            node_size=8,
            top_k_per_node=1,
        )
    except ValueError as exc:
        logger.warning("find_max_bw_for_k_gpus: %s", exc)
        return 0.0, []

    if k > sum(avail_capacity):
        logger.warning(
            "find_max_bw_for_k_gpus: k=%s still greater than available GPUs %s, return 0 bandwidth.",
            k,
            sum(avail_capacity),
        )
        return 0.0, []

    lookup = BandwidthLookupCache.ensure_loaded(data_path)
    cross_lookup: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
    for key, records in lookup.items():
        max_bw = max(float(bw) for _, bw in records)
        cross_lookup[key] = max_bw

    best_bandwidth = 0.0
    best_config: List[int] = []

    distributions = _generate_distributions(
        num_machines=num_machines,
        target_gpu=k,
        avail_capacity=avail_capacity,
        suffix_capacity=suffix_capacity,
        best_intra_configs=best_intra_configs,
        node_size=8,
    )

    for distribution in distributions:
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
            best_config = _build_combo_from_distribution(
                distribution,
                best_intra_configs,
                total_gpu,
                node_size=8,
            )

    if best_bandwidth <= 0:
        logger.warning(
            "find_max_bw_for_k_gpus: no feasible config for k=%s under current avail_gpu", k
        )
        return 0.0, []

    return best_bandwidth, best_config


def find_max_bw_for_k_gpus_with_contention(
    k: int,
    total_gpu: int,
    gpu_bw_dict_list,
    avail_gpu: Sequence[int],
    cluster_manager: ClusterStateManager,
    job_id: Optional[int] = None,
    *,
    data_path: Optional[str] = None,
    local_top_k: int = 3,
    max_combos_per_distribution: int = 2048,
    max_total_combos: int = 100000,
) -> Tuple[float, List[int]]:
    """Estimate max available bandwidth for k GPUs under a ClusterStateManager (with background jobs).

    Heuristic: node-distribution enumeration + per-node top-k + combo pruning:
    1) Enumerate node-level distributions and sort by theoretical upper bound.
    2) Keep only per-node top-k combos to generate limited global candidates.
    3) Evaluate each candidate combo with cluster_manager.predict_with_contention.
    4) If reaching max_total_combos, return current best and log.

    Args:
        local_top_k: Per-node candidate cap; larger is closer to optimal but more expensive.
        max_combos_per_distribution: Global candidate cap per distribution.
        max_total_combos: Global evaluation cap to bound runtime.
    """
    if cluster_manager is None:
        raise ValueError("cluster_manager cannot be None")

    if len(avail_gpu) == 0:
        logger.warning("find_max_bw_for_k_gpus_with_contention: avail_gpu empty, return 0 bandwidth.")
        return 0.0, []

    if not (1 <= k <= len(avail_gpu)):
        logger.warning(
            "find_max_bw_for_k_gpus_with_contention: k=%s exceeds available GPUs %s, return 0 bandwidth.",
            k,
            len(avail_gpu),
        )
        return 0.0, []

    node_size = getattr(cluster_manager, "node_size", 8)
    local_top_k = max(local_top_k, 1)
    try:
        (
            num_machines,
            avail_capacity,
            suffix_capacity,
            best_intra_configs,
            candidate_map,
        ) = _prepare_node_resources(
            total_gpu=total_gpu,
            avail_gpu=avail_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            node_size=node_size,
            top_k_per_node=local_top_k,
        )
    except ValueError as exc:
        logger.warning("find_max_bw_for_k_gpus_with_contention: %s", exc)
        return 0.0, []

    if k > sum(avail_capacity):
        logger.warning(
            "find_max_bw_for_k_gpus_with_contention: k=%s still greater than available GPUs %s, return 0 bandwidth.",
            k,
            sum(avail_capacity),
        )
        return 0.0, []

    cross_lookup = _build_cross_lookup(data_path)

    distributions = list(
        _generate_distributions(
            num_machines=num_machines,
            target_gpu=k,
            avail_capacity=avail_capacity,
            suffix_capacity=suffix_capacity,
            best_intra_configs=best_intra_configs,
            node_size=node_size,
        )
    )

    distribution_entries: List[Tuple[float, Tuple[int, ...]]] = []
    for distribution in distributions:
        upper = _estimate_distribution_upper_bound(
            distribution,
            k,
            best_intra_configs,
            cross_lookup=cross_lookup,
        )
        if upper <= 0:
            continue
        distribution_entries.append((upper, distribution))

    distribution_entries.sort(key=lambda item: item[0], reverse=True)

    best_bandwidth = 0.0
    best_config: List[int] = []
    eval_cache: Dict[Tuple[int, ...], float] = {}
    total_evaluated = 0
    max_combos_per_distribution = max_combos_per_distribution or None
    max_total_combos = max_total_combos or None

    for theoretical_upper, distribution in distribution_entries:
        if theoretical_upper <= best_bandwidth:
            continue

        candidate_combos = _enumerate_candidate_combos(
            distribution=distribution,
            candidate_map=candidate_map,
            total_gpu=total_gpu,
            node_size=node_size,
            local_top_k=local_top_k,
            max_candidates=max_combos_per_distribution,
        )
        if not candidate_combos:
            continue

        for combo_arr in candidate_combos:
            combo_key = tuple(int(x) for x in combo_arr.tolist())
            if combo_key in eval_cache:
                candidate_bw = eval_cache[combo_key]
            else:
                try:
                    if job_id is not None:
                        cluster_manager.set_job_context(job_id)
                    candidate_bw = float(cluster_manager.predict_with_contention(combo_arr))
                finally:
                    if job_id is not None:
                        cluster_manager.clear_job_context()
                eval_cache[combo_key] = candidate_bw

            total_evaluated += 1
            if candidate_bw > best_bandwidth:
                best_bandwidth = candidate_bw
                best_config = list(combo_arr.tolist())

            if max_total_combos is not None and total_evaluated >= max_total_combos:
                logger.warning(
                    "find_max_bw_for_k_gpus_with_contention: hit evaluation cap %s, returning current best.",
                    max_total_combos,
                )
                return best_bandwidth, best_config

        if theoretical_upper <= best_bandwidth:
            continue

    if best_bandwidth <= 0:
        logger.warning(
            "find_max_bw_for_k_gpus_with_contention: no feasible config for k=%s under current avail_gpu",
            k,
        )
        return 0.0, []

    return best_bandwidth, best_config

