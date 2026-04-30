"""Network-aware baseline implementations.

`CasCore` builds locality-aware candidate placements and reranks them with a
shared-resource compatibility score derived from `ClusterStateManager`.
`NetworkLocality` is kept as a backward-compatible alias for `CasCore`.
`BWGreedy` is a non-ML pairwise-bandwidth-aware baseline.
"""
from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values
from core.cluster_state import SharedResourceCompatibilityScore, SharedResourceCompatibilityScorer
from core.topology import get_link_weight

logger = logging.getLogger(__name__)

CASCORE_NAME = "CasCore"
LEGACY_NETWORK_LOCALITY_NAME = "NetworkLocality"
CASCORE_ALIASES = {
    LEGACY_NETWORK_LOCALITY_NAME: CASCORE_NAME,
    CASCORE_NAME: CASCORE_NAME,
}


def normalize_network_baseline_name(name: str) -> str:
    """Normalize legacy `NetworkLocality` labels to `CasCore`."""

    normalized = CASCORE_ALIASES.get(str(name), str(name))
    if str(name) == LEGACY_NETWORK_LOCALITY_NAME:
        warnings.warn(
            "`NetworkLocality` is deprecated; use `CasCore` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return normalized


def normalize_network_baseline_series(values: Iterable[str]) -> List[str]:
    """Normalize a sequence of baseline labels for reports and artifacts."""

    return [normalize_network_baseline_name(str(value)) for value in values]


def _normalize_available(total_gpu: int, avail_gpu_indices: Sequence[int], gpu_need: int) -> List[int]:
    """Validate and sort the available GPU indices for a request."""

    normalized = sorted({int(gpu) for gpu in avail_gpu_indices if 0 <= int(gpu) < total_gpu})
    if gpu_need > len(normalized):
        raise ValueError("Requested GPU count exceeds available GPUs")
    return normalized


def _finalize_mask(total_gpu: int, selected: Sequence[int]) -> np.ndarray:
    """Convert selected GPU indices into a full-length 0/1 allocation mask."""

    mask = np.zeros(total_gpu, dtype=int)
    for gpu_idx in selected:
        mask[int(gpu_idx)] = 1
    return mask


def _build_node_to_available(
    available: Sequence[int],
    gpu_to_node_map: Dict[int, int],
) -> Dict[int, List[int]]:
    """Group available GPUs by topology node."""

    grouped: Dict[int, List[int]] = {}
    for gpu_idx in available:
        node_id = int(gpu_to_node_map.get(int(gpu_idx), -1))
        grouped.setdefault(node_id, []).append(int(gpu_idx))
    for node_id in grouped:
        grouped[node_id] = sorted(grouped[node_id])
    return grouped


def _infer_node_capacity(gpu_to_node_map: Dict[int, int]) -> int:
    """Infer the largest per-node GPU capacity from the topology map."""

    counts: Dict[int, int] = {}
    for gpu_idx, node_id in gpu_to_node_map.items():
        counts[int(node_id)] = counts.get(int(node_id), 0) + 1
    return max(counts.values()) if counts else 1


def _background_nodes(
    background_combo: Optional[np.ndarray],
    gpu_to_node_map: Dict[int, int],
) -> Set[int]:
    """Return nodes touched by the background occupancy mask."""

    if background_combo is None:
        return set()
    nodes: Set[int] = set()
    for gpu_idx, flag in enumerate(np.asarray(background_combo, dtype=int).tolist()):
        if int(flag) == 1:
            nodes.add(int(gpu_to_node_map.get(gpu_idx, -1)))
    return nodes


def _topology_pair_weight(topo_matrix, gpu_a: int, gpu_b: int) -> float:
    """Return the normalized topology weight between two GPUs."""

    return float(get_link_weight(str(topo_matrix.iloc[gpu_a, gpu_b])))


def _subset_topology_potential(
    subset_nodes: Sequence[int],
    node_to_available: Dict[int, List[int]],
    topo_matrix,
) -> float:
    """Estimate topology potential for all GPUs in a candidate node subset."""

    subset_gpus: List[int] = []
    for node_id in subset_nodes:
        subset_gpus.extend(node_to_available.get(int(node_id), []))
    score = 0.0
    sorted_gpus = sorted(subset_gpus)
    for idx, gpu_a in enumerate(sorted_gpus):
        for gpu_b in sorted_gpus[idx + 1 :]:
            score += _topology_pair_weight(topo_matrix, gpu_a, gpu_b)
    return float(score)


def _generate_node_shortlist(
    *,
    gpu_need: int,
    node_to_available: Dict[int, List[int]],
    topo_matrix,
    background_nodes: Set[int],
    shortlist_limit: int,
    extra_node_slack: int,
    node_capacity: int,
) -> List[Tuple[int, ...]]:
    """Generate a locality-aware shortlist of feasible node subsets."""

    if not node_to_available:
        return []
    nodes = sorted(node_to_available.keys())
    min_nodes = max(1, ceil(int(gpu_need) / max(1, int(node_capacity))))

    candidate_sizes = [size for size in range(min_nodes, min_nodes + int(extra_node_slack) + 1) if size <= len(nodes)]
    feasible_subsets: List[Tuple[int, ...]] = []

    def _is_feasible(node_subset: Tuple[int, ...]) -> bool:
        return sum(len(node_to_available[node_id]) for node_id in node_subset) >= int(gpu_need)

    for subset_size in candidate_sizes:
        for node_subset in itertools.combinations(nodes, subset_size):
            if _is_feasible(node_subset):
                feasible_subsets.append(tuple(int(node_id) for node_id in node_subset))

    if not feasible_subsets:
        for subset_size in range((candidate_sizes[-1] + 1) if candidate_sizes else min_nodes, len(nodes) + 1):
            for node_subset in itertools.combinations(nodes, subset_size):
                if _is_feasible(node_subset):
                    feasible_subsets.append(tuple(int(node_id) for node_id in node_subset))
            if feasible_subsets:
                break

    def _subset_sort_key(node_subset: Tuple[int, ...]) -> Tuple[float, ...]:
        shared_node_count = len(set(node_subset) & background_nodes)
        available_count = sum(len(node_to_available[node_id]) for node_id in node_subset)
        topology_potential = _subset_topology_potential(node_subset, node_to_available, topo_matrix)
        return (
            float(shared_node_count),
            float(-available_count),
            float(-topology_potential),
            *tuple(float(node_id) for node_id in node_subset),
        )

    feasible_subsets = sorted(feasible_subsets, key=_subset_sort_key)
    return feasible_subsets[: max(1, int(shortlist_limit))]


def _greedy_select_within_subset(
    *,
    total_gpu: int,
    gpu_need: int,
    subset_nodes: Sequence[int],
    node_to_available: Dict[int, List[int]],
    topo_matrix,
) -> np.ndarray:
    """Select GPUs inside one node subset using topology-aware greedy expansion."""

    subset_available: List[int] = []
    for node_id in subset_nodes:
        subset_available.extend(node_to_available.get(int(node_id), []))
    subset_available = sorted(subset_available)
    if gpu_need > len(subset_available):
        raise ValueError("GPU need exceeds available GPUs inside the chosen subset")

    best_seed = subset_available[0]
    best_seed_score = float("-inf")
    for candidate in subset_available:
        future_locality = sum(
            _topology_pair_weight(topo_matrix, candidate, other)
            for other in subset_available
            if other != candidate
        )
        if future_locality > best_seed_score or (
            future_locality == best_seed_score and int(candidate) < int(best_seed)
        ):
            best_seed = int(candidate)
            best_seed_score = float(future_locality)

    selected = [int(best_seed)]
    remaining = [gpu for gpu in subset_available if int(gpu) != int(best_seed)]
    while len(selected) < int(gpu_need) and remaining:
        best_gpu = int(remaining[0])
        best_gain = float("-inf")
        for candidate in remaining:
            gain = sum(_topology_pair_weight(topo_matrix, int(candidate), chosen) for chosen in selected)
            if gain > best_gain or (gain == best_gain and int(candidate) < int(best_gpu)):
                best_gpu = int(candidate)
                best_gain = float(gain)
        selected.append(best_gpu)
        remaining.remove(best_gpu)
    return _finalize_mask(total_gpu, selected)


@dataclass(frozen=True)
class _CasCoreCandidate:
    """CasCore candidate with topology and compatibility scores."""

    combo: np.ndarray
    subset_nodes: Tuple[int, ...]
    topology_score: float
    compatibility: SharedResourceCompatibilityScore


def cascore_algo(
    total_gpu: int,
    avail_gpu_indices: Sequence[int],
    gpu_need: int,
    topo_matrix,
    gpu_to_node_map: Dict[int, int],
    background_combo: Optional[np.ndarray] = None,
    compatibility_scorer: Optional[SharedResourceCompatibilityScorer] = None,
    shortlist_limit: int = 12,
    extra_node_slack: int = 1,
    penalty_weight: Optional[float] = None,
) -> np.ndarray:
    """`CasCore`: shared-resource compatibility-aware reranking baseline.

    Workflow:
    1. enumerate locality-aware candidate placements;
    2. rerank the shortlist by shared-resource compatibility.

    Notes:
    - `penalty_weight` is accepted for legacy CLI compatibility and ignored;
    - when no `compatibility_scorer` is supplied, the baseline falls back to the
      locality shortlist winner.
    """

    if penalty_weight is not None:
        logger.debug("CasCore ignores legacy penalty_weight=%.4f", float(penalty_weight))

    available = _normalize_available(total_gpu, avail_gpu_indices, gpu_need)
    node_to_available = _build_node_to_available(available, gpu_to_node_map)
    node_capacity = _infer_node_capacity(gpu_to_node_map)
    background_node_set = _background_nodes(background_combo, gpu_to_node_map)
    shortlist = _generate_node_shortlist(
        gpu_need=gpu_need,
        node_to_available=node_to_available,
        topo_matrix=topo_matrix,
        background_nodes=background_node_set,
        shortlist_limit=shortlist_limit,
        extra_node_slack=extra_node_slack,
        node_capacity=node_capacity,
    )

    if not shortlist:
        return _finalize_mask(total_gpu, available[:gpu_need])

    candidates: List[_CasCoreCandidate] = []
    for subset_nodes in shortlist:
        combo = _greedy_select_within_subset(
            total_gpu=total_gpu,
            gpu_need=gpu_need,
            subset_nodes=subset_nodes,
            node_to_available=node_to_available,
            topo_matrix=topo_matrix,
        )
        selected_gpus = np.where(combo == 1)[0].tolist()
        topology_score = 0.0
        for idx, gpu_a in enumerate(selected_gpus):
            for gpu_b in selected_gpus[idx + 1 :]:
                topology_score += _topology_pair_weight(topo_matrix, int(gpu_a), int(gpu_b))
        if compatibility_scorer is None:
            compatibility = SharedResourceCompatibilityScore(
                shared_node_count=len(set(subset_nodes) & background_node_set),
                candidate_standalone_bw=0.0,
                candidate_demand_bw=0.0,
                background_demand_bw=0.0,
                dual_capacity_bw=float("inf"),
                admitted_candidate_bw=0.0,
                compatibility_margin=float("inf"),
                feasible=True,
            )
        else:
            compatibility = compatibility_scorer.score_candidate(combo)
        candidates.append(
            _CasCoreCandidate(
                combo=combo,
                subset_nodes=tuple(int(node_id) for node_id in subset_nodes),
                topology_score=float(topology_score),
                compatibility=compatibility,
            )
        )

    def _candidate_sort_key(candidate: _CasCoreCandidate) -> Tuple[float, ...]:
        admitted_candidate_bw = float(candidate.compatibility.admitted_candidate_bw)
        if admitted_candidate_bw <= 0.0 and np.any(candidate.combo == 1):
            admitted_candidate_bw = float(candidate.topology_score)
        combo_signature = ",".join(str(int(idx)) for idx in np.where(candidate.combo == 1)[0].tolist())
        combo_key = tuple(float(idx) for idx in np.where(candidate.combo == 1)[0].tolist())
        feasibility_rank = 0.0 if bool(candidate.compatibility.feasible) else 1.0
        return (
            feasibility_rank,
            float(candidate.compatibility.shared_node_count),
            float(-admitted_candidate_bw),
            float(-candidate.compatibility.compatibility_margin),
            float(-candidate.topology_score),
            *combo_key,
            float(len(combo_signature)),
        )

    best_candidate = min(candidates, key=_candidate_sort_key)
    return np.asarray(best_candidate.combo, dtype=int)


def _pairwise_bandwidth(
    total_gpu: int,
    gpu_a: int,
    gpu_b: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    pair_bw_cache: MutableMapping[Tuple[int, int], float],
) -> float:
    """Compute or reuse the pairwise bandwidth score used by `BWGreedy`."""

    key = (min(int(gpu_a), int(gpu_b)), max(int(gpu_a), int(gpu_b)))
    if key in pair_bw_cache:
        return float(pair_bw_cache[key])

    combo = np.zeros(total_gpu, dtype=int)
    combo[key[0]] = 1
    combo[key[1]] = 1
    final_bw, _, _ = calculate_bandwidth_values(
        combo,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        data_path,
    )
    pair_bw_cache[key] = float(final_bw)
    return float(final_bw)


def bw_greedy_algo(
    total_gpu: int,
    avail_gpu_indices: Sequence[int],
    gpu_need: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    gpu_to_node_map: Dict[int, int],
    background_combo: Optional[np.ndarray] = None,
    pair_bw_cache: Optional[MutableMapping[Tuple[int, int], float]] = None,
    penalty_weight: float = 0.15,
) -> np.ndarray:
    """Pairwise-bandwidth-aware greedy baseline.

    The baseline is a non-ML, network-aware challenger. It seeds the allocation
    from the best pairwise bandwidth, expands by aggregate pairwise compatibility,
    and applies a coarse penalty for nodes already touched by background jobs.
    """

    available = _normalize_available(total_gpu, avail_gpu_indices, gpu_need)
    background_node_set = _background_nodes(background_combo, gpu_to_node_map)
    cache = pair_bw_cache if pair_bw_cache is not None else {}

    def _pair_penalty(gpu_a: int, gpu_b: int) -> float:
        node_a = int(gpu_to_node_map.get(gpu_a, -1))
        node_b = int(gpu_to_node_map.get(gpu_b, -1))
        if node_a == node_b:
            return 0.0
        return float((1 if node_a in background_node_set else 0) + (1 if node_b in background_node_set else 0))

    def pair_score(gpu_a: int, gpu_b: int) -> float:
        bw_value = _pairwise_bandwidth(
            total_gpu=total_gpu,
            gpu_a=gpu_a,
            gpu_b=gpu_b,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            pair_bw_cache=cache,
        )
        return bw_value - float(penalty_weight) * _pair_penalty(gpu_a, gpu_b)

    best_pair: Optional[Tuple[int, int]] = None
    best_pair_score = float("-inf")
    for idx, gpu_a in enumerate(available):
        for gpu_b in available[idx + 1 :]:
            score = pair_score(gpu_a, gpu_b)
            if best_pair is None or score > best_pair_score or (
                score == best_pair_score and (gpu_a, gpu_b) < best_pair
            ):
                best_pair = (gpu_a, gpu_b)
                best_pair_score = score

    if best_pair is None:
        return _finalize_mask(total_gpu, available[:gpu_need])

    selected = [best_pair[0], best_pair[1]]
    if gpu_need == 1:
        selected = [best_pair[0]]
    remaining = [gpu for gpu in available if gpu not in selected]

    while len(selected) < gpu_need and remaining:
        best_gpu = remaining[0]
        best_gain = float("-inf")
        for candidate in remaining:
            gain = sum(pair_score(candidate, chosen) for chosen in selected)
            if gain > best_gain or (gain == best_gain and candidate < best_gpu):
                best_gpu = candidate
                best_gain = gain
        selected.append(best_gpu)
        remaining.remove(best_gpu)

    return _finalize_mask(total_gpu, selected[:gpu_need])


def network_locality_algo(*args, **kwargs) -> np.ndarray:
    """Backward-compatible `NetworkLocality` wrapper around `CasCore`."""

    return cascore_algo(*args, **kwargs)


__all__ = [
    "CASCORE_NAME",
    "LEGACY_NETWORK_LOCALITY_NAME",
    "normalize_network_baseline_name",
    "normalize_network_baseline_series",
    "cascore_algo",
    "network_locality_algo",
    "bw_greedy_algo",
]
