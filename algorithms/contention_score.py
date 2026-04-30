"""Contention-aware structural rerank helpers.

The structural prior is intentionally narrow. It only downweights extreme
long-tail cross-node shapes for homogeneous clusters under `common` or
`intensive` contention. Heterogeneous clusters and real-data upper-bound paths
fall back to raw scores.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

import numpy as np

from core.bandwidth import CUSTOM_CLUSTER_NODE_TYPES, SwitchBandwidthConfig, get_gpu_dict_files


def is_contention_sensitive_mode(cluster_manager: Optional[Any]) -> bool:
    """Return whether the current cluster manager is in a contention-heavy mode."""
    return bool(
        cluster_manager is not None
        and getattr(cluster_manager, "contention_mode", None) in {"common", "intensive"}
    )


def _normalize_node_family_label(node_label: str) -> str:
    """Map node-type labels to a coarse family used by the homogeneous/heterogeneous gate."""
    normalized = str(node_label).strip()
    if normalized.endswith("_gpu_bw_dict.pkl"):
        normalized = normalized[: -len("_gpu_bw_dict.pkl")]
    # H100_26/H100_27/... should all be treated as the same homogeneous family "H100".
    normalized = re.sub(r"_[0-9]+$", "", normalized)
    return normalized


def _resolve_cluster_type(
    switch_config: Optional[SwitchBandwidthConfig],
    cluster_manager: Optional[Any],
) -> Optional[str]:
    """Resolve cluster_type from explicit switch_config first, then fall back to cluster_manager."""
    if switch_config is not None:
        cluster_type = getattr(switch_config, "cluster_type", None)
        if cluster_type:
            return str(cluster_type)

    if cluster_manager is not None:
        nested_switch_config = getattr(cluster_manager, "switch_config", None)
        if nested_switch_config is not None:
            cluster_type = getattr(nested_switch_config, "cluster_type", None)
            if cluster_type:
                return str(cluster_type)

        cluster_type = getattr(cluster_manager, "cluster_type", None)
        if cluster_type:
            return str(cluster_type)

    return None


def _resolve_num_machines(
    switch_config: Optional[SwitchBandwidthConfig],
    cluster_manager: Optional[Any],
) -> int:
    """Resolve the node count used by cluster-type expansion; fall back to 1 when unavailable."""
    if switch_config is not None and getattr(switch_config, "num_machines", None):
        return max(1, int(switch_config.num_machines))

    if cluster_manager is not None:
        nested_switch_config = getattr(cluster_manager, "switch_config", None)
        if nested_switch_config is not None and getattr(nested_switch_config, "num_machines", None):
            return max(1, int(nested_switch_config.num_machines))
        total_gpu = getattr(cluster_manager, "total_gpu", None)
        if total_gpu:
            return max(1, int(total_gpu) // 8)

    return 1


def is_homogeneous_cluster(
    *,
    switch_config: Optional[SwitchBandwidthConfig],
    cluster_manager: Optional[Any],
) -> bool:
    """Return whether the current cluster can be safely treated as homogeneous.

    The gate is intentionally conservative:
    - known custom heterogeneous cluster types such as ``Het-4Mix`` return False;
    - H100_26/H100_27/... collapse to the same family ``H100`` and therefore return True;
    - if cluster metadata is missing or unparseable, return False to avoid accidental enablement.
    """
    cluster_type = _resolve_cluster_type(switch_config, cluster_manager)
    if not cluster_type:
        return False

    if cluster_type in CUSTOM_CLUSTER_NODE_TYPES:
        node_labels = CUSTOM_CLUSTER_NODE_TYPES[cluster_type]
    else:
        node_labels = get_gpu_dict_files(cluster_type, repeat=_resolve_num_machines(switch_config, cluster_manager))

    if not node_labels:
        return False

    node_families = {_normalize_node_family_label(label) for label in node_labels}
    return len(node_families) <= 1


def structural_balance_multiplier(config: np.ndarray) -> float:
    """Only downweight extreme long-tail cross-node shapes under contention.

    The prior is intentionally narrow. It targets shapes whose active nodes are
    highly imbalanced, such as `(8, 1)` or `(8, 8, 1)`, and leaves balanced or
    compact shapes at multiplier `1.0`.
    """
    reshaped = np.asarray(config, dtype=int).reshape(-1, 8)
    node_counts = reshaped.sum(axis=1)
    active_counts = node_counts[node_counts > 0]
    if active_counts.size <= 1:
        return 1.0

    active_counts = active_counts.astype(float)
    min_count = float(active_counts.min())
    max_count = float(active_counts.max())
    if (max_count - min_count) <= 5.0:
        return 1.0

    total_count = float(active_counts.sum())
    mean_count = float(active_counts.mean())

    density_ratio = min_count / max(max_count, 1.0)
    saturation_ratio = float(np.sum(active_counts >= 8.0)) / float(len(active_counts))
    imbalance_ratio = float(np.sum(np.abs(active_counts - mean_count))) / max(total_count, 1.0)

    multiplier = (0.6 + 0.4 * density_ratio)
    multiplier *= (1.0 - 0.35 * saturation_ratio)
    multiplier *= max(0.45, 1.0 - 0.5 * imbalance_ratio)

    # Single-GPU tail nodes are the strongest long-tail signal.
    if min_count <= 1.0:
        multiplier *= 0.35

    return float(max(0.5, multiplier))


def apply_contention_structural_rerank(
    configs: Sequence[np.ndarray],
    scores: Sequence[float],
    *,
    cluster_manager: Optional[Any],
    switch_config: Optional[SwitchBandwidthConfig] = None,
) -> np.ndarray:
    """Apply structural multipliers only for homogeneous clusters under contention."""
    score_array = np.asarray(scores, dtype=float)
    if score_array.size == 0 or not is_contention_sensitive_mode(cluster_manager):
        return score_array

    # Structural rerank is a homogeneous-cluster prior. Disable it for heterogeneous
    # or metadata-unknown clusters so the selector keeps the raw predicted ordering.
    if not is_homogeneous_cluster(
        switch_config=switch_config,
        cluster_manager=cluster_manager,
    ):
        return score_array

    multipliers = np.asarray(
        [structural_balance_multiplier(np.asarray(config, dtype=int)) for config in configs],
        dtype=float,
    )
    return score_array * multipliers
