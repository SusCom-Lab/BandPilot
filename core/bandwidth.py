"""Bandwidth computation core module.

Encapsulates bandwidth lookup, GPU config statistics, and model-input construction logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import pickle

from data_process.preprocessing import preprocess_gpu_data, find_matching_bandwidth, analyze_gpu_pattern

logger = logging.getLogger(__name__)


@dataclass
class SwitchBandwidthConfig:
    """Switch bandwidth configuration."""

    num_machines: int
    cluster_type: str | None = None
    bw_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bw_matrix = np.zeros((self.num_machines, self.num_machines), dtype=float)

    def set_bandwidth(self, i: int, j: int, bw: float) -> None:
        """Set bandwidth between two machine groups."""
        if 0 <= i < self.num_machines and 0 <= j < self.num_machines:
            self.bw_matrix[i, j] = bw
            self.bw_matrix[j, i] = bw

    def get_bandwidth(self, i: int, j: int) -> float:
        """Get bandwidth between two machine groups."""
        if 0 <= i < self.num_machines and 0 <= j < self.num_machines:
            return float(self.bw_matrix[i, j])
        return 0.0

    def get_path_bandwidth(self, path: Sequence[int]) -> float:
        """Return the minimum bandwidth along a path."""
        if len(path) < 2:
            return float("inf")
        bandwidths = [
            self.get_bandwidth(path[idx], path[idx + 1]) for idx in range(len(path) - 1)
            if self.get_bandwidth(path[idx], path[idx + 1]) > 0
        ]
        return min(bandwidths) if bandwidths else 0.0


class BandwidthLookupCache:
    """Cache lookup table to avoid repeated CSV reads."""

    _lookup_table: Dict | None = None
    _loaded_path: Path | None = None

    @classmethod
    def ensure_loaded(cls, data_path: Path) -> Dict:
        normalized_path = Path(data_path).resolve()
        if cls._lookup_table is None or cls._loaded_path != normalized_path:
            logger.info("Loading bandwidth lookup table: %s", normalized_path)
            cls._lookup_table = preprocess_gpu_data(str(normalized_path))
            cls._loaded_path = normalized_path
            if cls._lookup_table is None:
                raise RuntimeError(f"Failed to load bandwidth lookup table: {normalized_path}")
        return cls._lookup_table

    @classmethod
    def reset(cls) -> None:
        """Reset cache to allow reload after switching data sources."""
        cls._lookup_table = None
        cls._loaded_path = None


def load_gpu_bw_dict(file_path: Path) -> Dict:
    """Load a GPU bandwidth dictionary from pickle."""
    if not file_path.exists():
        raise FileNotFoundError(f"Bandwidth dictionary not found: {file_path}")
    with file_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"File {file_path} did not contain a dictionary.")
    return data


CUSTOM_CLUSTER_NODE_TYPES = {
    "Het-4Mix": ["4090", "V100", "A6000","A800"],
}


def _expand_gpu_types_for_nodes(node_types: Sequence[str], repeat: int) -> List[str]:
    if repeat <= 0 or not node_types:
        return []
    cycles = math.ceil(repeat / len(node_types))
    ordered = list(node_types) * cycles
    return [f"{gpu}_gpu_bw_dict.pkl" for gpu in ordered[:repeat]]


def get_gpu_dict_files(cluster_type: str, repeat: int) -> List[str]:
    """List required bandwidth dictionary files for a cluster type."""
    if cluster_type in CUSTOM_CLUSTER_NODE_TYPES:
        node_types = CUSTOM_CLUSTER_NODE_TYPES[cluster_type]
        return _expand_gpu_types_for_nodes(node_types, repeat)

    known_gpu_types = ["4090", "V100", "A6000", "A800", "H100_26", "H100_27", "H100_28", "H100_29"]
    gpu_types = [gpu for gpu in known_gpu_types if gpu in cluster_type]
    if not gpu_types:
        logger.warning("No known GPU type found in cluster_type: %s", cluster_type)
        return []
    return _expand_gpu_types_for_nodes(gpu_types, repeat)


def get_gpu_counts_for_model(gpu_config: np.ndarray, total_gpu: int) -> Tuple[List[int], int]:
    """Compute per-node active GPU counts and total."""
    if total_gpu % 8 != 0:
        raise ValueError("total_gpu must be divisible by 8")
    num_machines = total_gpu // 8
    per_node_counts = [
        int(np.sum(gpu_config[idx * 8 : (idx + 1) * 8])) for idx in range(num_machines)
    ]
    total_active = int(np.sum(per_node_counts))
    return per_node_counts, total_active


def calculate_bandwidth_values(
    gpu: Sequence[int],
    total_gpu: int,
    gpu_bw_dict_list: Sequence[Dict[Tuple[int, ...], float]],
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
) -> Tuple[float, List[float], SwitchBandwidthConfig | float | None]:
    """Look up and compute end-to-end bandwidth for a given GPU configuration."""
    if total_gpu % 8 != 0:
        raise ValueError("total_gpu must be a multiple of 8")
    if len(gpu) != total_gpu:
        raise ValueError("gpu length must equal total_gpu")

    # Ensure gpu is a Python list, not np.ndarray.
    # This is important for subsequent slicing and type conversion operations.
    if isinstance(gpu, np.ndarray):
        gpu = gpu.tolist()
    else:
        gpu = list(gpu)

    # Early return: gpu_sum 0 or 1 -> 0 bandwidth (single-GPU combos typically not in table)
    gpu_sum = sum(gpu)
    if gpu_sum == 0:
        # Empty config -> 0 bandwidth
        parts = [tuple(int(x) for x in gpu[idx : idx + 8]) for idx in range(0, total_gpu, 8)]
        part_bandwidths = [0.0] * len(parts)
        return 0.0, part_bandwidths, switch_config
    elif gpu_sum == 1:
        # Single GPU config -> 0 bandwidth; avoid lookup warnings
        parts = [tuple(int(x) for x in gpu[idx : idx + 8]) for idx in range(0, total_gpu, 8)]
        part_bandwidths = [0.0] * len(parts)
        return 0.0, part_bandwidths, switch_config

    lookup_table = BandwidthLookupCache.ensure_loaded(Path(data_path))
    # Ensure each node config is a Python list of ints
    nodes_config = []
    for idx in range(0, total_gpu, 8):
        node_slice = gpu[idx : idx + 8]
        # Ensure list of ints
        node_list = [int(x) for x in node_slice]
        # Pad to length 8 if shorter
        if len(node_list) < 8:
            node_list.extend([0] * (8 - len(node_list)))
        nodes_config.append(node_list)
    
    result = find_matching_bandwidth(nodes_config, lookup_table)
    if result is not None:
        _, bandwidth = result
        final_bandwidth = float(bandwidth)  # ensure float
    else:
        # If no match, log debug info and return 0 bandwidth
        key = analyze_gpu_pattern(nodes_config)
        logger.warning(
            "No matching GPU config in bandwidth table: nodes_config=%s, key=%s, total_gpu=%s, gpu_sum=%s",
            nodes_config,
            key,
            total_gpu,
            sum(gpu),
        )
        final_bandwidth = 0.0

    parts = [tuple(int(x) for x in gpu[idx : idx + 8]) for idx in range(0, total_gpu, 8)]
    part_bandwidths: List[float] = []
    for idx, part_tuple in enumerate(parts):
        current_dict = gpu_bw_dict_list[idx]
        part_bandwidths.append(float(round(current_dict.get(part_tuple, 0.0), 2)))

    cluster_label = getattr(switch_config, "cluster_type", None)
    if cluster_label in CUSTOM_CLUSTER_NODE_TYPES:
        # Only positive intra-node bandwidth participates in bottleneck:
        # - Single-card/missing patterns yielding 0 should not suppress cross-node bandwidth
        active_bws = [
            bw
            for bw, part in zip(part_bandwidths, parts)
            if any(part) and bw > 0.0
        ]
        if active_bws:
            intra_bottleneck = min(active_bws)
            final_bandwidth = float(min(final_bandwidth, intra_bottleneck))

    return final_bandwidth, part_bandwidths, switch_config


def config_to_bandwidth(
    gpu_config_list: Iterable[Sequence[int]],
    total_gpu: int,
    gpu_bw_dict_list: Sequence[Dict[Tuple[int, ...], float]],
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a set of GPU configs to bandwidths and per-part bandwidths."""
    bandwidths: List[float] = []
    part_bandwidths: List[List[float]] = []
    for gpu_config in gpu_config_list:
        final_bw, part_bws, _ = calculate_bandwidth_values(
            gpu_config, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        bandwidths.append(final_bw)
        part_bandwidths.append(part_bws)
    return np.array(bandwidths), np.array(part_bandwidths)


def prepare_model_inputs(
    gpu_config_list: Sequence[Sequence[int]],
    total_gpu: int,
    gpu_bw_dict_list: Sequence[Dict[Tuple[int, ...], float]],
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build part bandwidths, node counts, and total counts for model inputs."""
    part_bws_list: List[List[float]] = []
    node_counts_list: List[List[int]] = []
    total_counts_list: List[int] = []

    for gpu_config in gpu_config_list:
        _, part_bws, _ = calculate_bandwidth_values(
            gpu_config, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        node_counts, total_counts = get_gpu_counts_for_model(np.array(gpu_config), total_gpu)

        part_bws_list.append(part_bws)
        node_counts_list.append(node_counts)
        total_counts_list.append(total_counts)

    return (
        np.array(part_bws_list),
        np.array(node_counts_list),
        np.array(total_counts_list).reshape(-1, 1),
    )

