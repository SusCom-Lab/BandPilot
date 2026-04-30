"""Feature vectors for online mismatch-kNN risk estimation.

The runtime bank predicts whether PTS is likely to improve on EHA by comparing
node-level placement mismatch features. The vector shape is fixed so banks
remain compatible across clusters and contention modes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from algorithms.adaptive_gate_support.utils import _as_int


_GPU_PER_NODE = 8
_MAX_NODE_BUCKETS = 8
_MISMATCH_NODE_GAP_KEYS = tuple(
    f"selected_minus_avail_share_node_{node_idx}"
    for node_idx in range(_MAX_NODE_BUCKETS)
)
_MISMATCH_KNN_FEATURE_KEYS = (
    "test_num",
    "eha_node_count",
    "coverage_score",
    "selected_active_node_ratio",
    "selected_node_entropy",
    "selected_node_max_share",
    "selected_vs_avail_l1",
    "selected_vs_uniform_l1",
    *_MISMATCH_NODE_GAP_KEYS,
)


def _parse_combo_signature(signature: object) -> List[int]:
    """Parse a comma-separated GPU-index signature into sorted unique indices."""

    if signature in ("", None):
        return []
    indices: List[int] = []
    for token in str(signature).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value >= 0:
            indices.append(value)
    return sorted(set(indices))


def _infer_total_gpu(sample: Mapping[str, object]) -> int:
    """Infer the cluster size from explicit fields and GPU signatures."""

    total_gpu = _as_int(sample.get("total_gpu", 0))
    if total_gpu > 0:
        return total_gpu

    avail_count = _as_int(sample.get("avail_gpu_count", 0))
    background_count = _as_int(sample.get("background_gpu_count", 0))
    if avail_count > 0 or background_count > 0:
        total_gpu = max(total_gpu, avail_count + background_count)

    for field_name in (
        "avail_signature",
        "background_signature",
        "eha_combo_signature",
        "bandpilot_combo_signature",
    ):
        indices = _parse_combo_signature(sample.get(field_name, ""))
        if indices:
            total_gpu = max(total_gpu, max(indices) + 1)

    return max(_GPU_PER_NODE, total_gpu)


def _signature_to_node_shares(signature: object, *, total_gpu: int) -> List[float]:
    """Convert selected GPU indices into fixed-width per-node occupancy shares."""

    node_bucket_count = min(
        _MAX_NODE_BUCKETS,
        max(1, math.ceil(max(int(total_gpu), 1) / float(_GPU_PER_NODE))),
    )
    node_counts = [0.0 for _ in range(_MAX_NODE_BUCKETS)]
    indices = _parse_combo_signature(signature)
    if not indices:
        return node_counts

    for gpu_idx in indices:
        node_idx = min(node_bucket_count - 1, max(0, gpu_idx // _GPU_PER_NODE))
        node_counts[node_idx] += 1.0

    total_count = sum(node_counts)
    if total_count <= 0:
        return node_counts
    return [count / total_count for count in node_counts]


def _normalized_entropy(shares: Sequence[float]) -> float:
    """Return normalized entropy over positive node shares."""

    positive_shares = [float(value) for value in shares if float(value) > 0.0]
    if len(positive_shares) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in positive_shares)
    return entropy / math.log(len(positive_shares))


def _l1_distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    """Return half-L1 distance between two share vectors in the range [0, 1]."""

    return 0.5 * sum(abs(float(left) - float(right)) for left, right in zip(lhs, rhs))


def _cluster_group_key(sample: Mapping[str, object]) -> str:
    """Return the cluster grouping key used by online-kNN support banks."""

    return str(sample.get("cluster_type", ""))


def _robust_scale_columns(matrix: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Return per-column robust medians and MAD-based scales for kNN distance."""

    if not matrix:
        return [], []
    dim = len(matrix[0])
    medians: List[float] = []
    scales: List[float] = []
    for col_idx in range(dim):
        column = sorted(float(row[col_idx]) for row in matrix)
        mid = len(column) // 2
        median = (
            column[mid]
            if len(column) % 2 == 1
            else 0.5 * (column[mid - 1] + column[mid])
        )
        deviations = sorted(abs(value - median) for value in column)
        mad = (
            deviations[mid]
            if len(deviations) % 2 == 1
            else 0.5 * (deviations[mid - 1] + deviations[mid])
        )
        scale = max(1e-9, 1.4826 * mad)
        medians.append(float(median))
        scales.append(float(scale))
    return medians, scales


def _compute_combo_mismatch_feature_map(sample: Mapping[str, object]) -> Dict[str, float]:
    """Build EHA-selection versus available-GPU topology mismatch features."""

    total_gpu = _infer_total_gpu(sample)
    selected_shares = _signature_to_node_shares(
        sample.get("eha_combo_signature", ""),
        total_gpu=total_gpu,
    )
    available_shares = _signature_to_node_shares(
        sample.get("avail_signature", ""),
        total_gpu=total_gpu,
    )

    active_selected_nodes = [idx for idx, share in enumerate(selected_shares) if share > 0.0]
    active_available_nodes = [idx for idx, share in enumerate(available_shares) if share > 0.0]

    uniform_selected_support = [0.0 for _ in range(_MAX_NODE_BUCKETS)]
    if active_selected_nodes:
        uniform_mass = 1.0 / float(len(active_selected_nodes))
        for node_idx in active_selected_nodes:
            uniform_selected_support[node_idx] = uniform_mass

    feature_map: Dict[str, float] = {
        "selected_active_node_ratio": (
            float(len(active_selected_nodes)) / float(max(1, len(active_available_nodes)))
        ),
        "selected_node_entropy": _normalized_entropy(selected_shares),
        "selected_node_max_share": max(selected_shares) if selected_shares else 0.0,
        "selected_vs_avail_l1": _l1_distance(selected_shares, available_shares),
        "selected_vs_uniform_l1": _l1_distance(selected_shares, uniform_selected_support),
    }
    for node_idx, gap in enumerate(
        float(selected_share) - float(available_share)
        for selected_share, available_share in zip(selected_shares, available_shares)
    ):
        feature_map[_MISMATCH_NODE_GAP_KEYS[node_idx]] = float(gap)
    return feature_map


def _sample_to_mismatch_knn_vector(
    sample: Mapping[str, object],
    feature_row: Mapping[str, object],
) -> List[float]:
    """Convert one replay sample into the fixed online mismatch-kNN vector."""

    mismatch_map = _compute_combo_mismatch_feature_map(sample)
    raw_values = {
        "test_num": float(_as_int(sample.get("test_num", 0))),
        "eha_node_count": float(_as_int(sample.get("eha_node_count", 0))),
        "coverage_score": float(feature_row.get("coverage_score", 0.0)),
        **mismatch_map,
    }
    return [float(raw_values[key]) for key in _MISMATCH_KNN_FEATURE_KEYS]
