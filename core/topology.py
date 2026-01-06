"""Topology-related utilities."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def parse_topo_matrix(filepath: str | Path) -> pd.DataFrame:
    """Parse a single-node topology file (e.g., A6000_topo.txt)."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Topology file not found: {filepath}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split("|")[2:-1]
    data_lines = [line.split("|")[1:-1] for line in lines[2:] if not line.startswith("|--")]
    row_labels = [row[0].strip() for row in data_lines]
    matrix = [[cell.strip() for cell in row[1:]] for row in data_lines]

    if len(row_labels) != len(matrix):
        raise ValueError("Row count does not match labels in topology file")
    if matrix and len(matrix[0]) != len(header):
        raise ValueError("Column count does not match labels in topology file")

    return pd.DataFrame(matrix, index=row_labels, columns=header)


def build_composite_topo_matrix(node_configs: Sequence[Tuple[str, int]]) -> Tuple[pd.DataFrame, int]:
    """Build a composite topology matrix."""
    parsed: Dict[str, pd.DataFrame] = {}
    total_gpu = 0
    offsets = [0]

    for topo_file, gpus_on_node in node_configs:
        if topo_file not in parsed:
            parsed[topo_file] = parse_topo_matrix(topo_file)
            if parsed[topo_file].shape[0] != gpus_on_node:
                raise ValueError(f"GPU count in {topo_file} mismatches configuration")
        total_gpu += gpus_on_node
        offsets.append(total_gpu)

    if total_gpu == 0:
        return pd.DataFrame(), 0

    composite = pd.DataFrame(
        "INTER",
        index=[f"GPU{i}" for i in range(total_gpu)],
        columns=[f"GPU{i}" for i in range(total_gpu)],
    )

    for idx, (topo_file, gpus_on_node) in enumerate(node_configs):
        start, end = offsets[idx], offsets[idx + 1]
        composite.iloc[start:end, start:end] = parsed[topo_file].values

    for diag in range(total_gpu):
        composite.iloc[diag, diag] = "X"

    return composite, total_gpu


def get_link_weight(link_type: str) -> int:
    """Return weight for a given link type.

    Supported link types:
    - X, INTER: weight 0 (no link or cross-node)
    - SYS: weight 1
    - PIX: weight 1.5
    - PXB: weight 2
    - NV<N>: NVLink; weight by version (NV16+ -> 6, NV8+ -> 5, NV4+ -> 4, NV1+ -> 3)
    """
    mapping = {"X": 0, "INTER": 0, "SYS": 1, "PIX": 1.5, "PXB": 2}
    link_type = link_type.strip().upper()
    # First check predefined mapping
    if link_type in mapping:
        return mapping[link_type]
    # Handle NVLink types (e.g., NV16, NV8, NV4)
    if link_type.startswith("NV"):
        # Should use \d+ in the original regex
        match = re.match(r"NV(\d+)", link_type)
        if match:
            num = int(match.group(1))
            # Weight based on NVLink version
            if num >= 16:
                return 6
            if num >= 8:
                return 5
            if num >= 4:
                return 4
            if num >= 1:
                return 3
    # Unknown link type: warn and return 0
    logger.warning("Unknown link type %s, treating as 0", link_type)
    return 0


def calculate_connectivity_score(gpu_indices: Sequence[int], topo_matrix: pd.DataFrame) -> float:
    """Compute connectivity weight score for a given GPU set."""
    score = 0.0
    valid_indices = [idx for idx in sorted(gpu_indices) if 0 <= idx < topo_matrix.shape[0]]
    if len(valid_indices) != len(gpu_indices):
        logger.warning("GPU indices out of range: all=%s, valid=%s", gpu_indices, valid_indices)

    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            link = topo_matrix.iloc[valid_indices[i], valid_indices[j]]
            score += get_link_weight(link)
    return score


def convert_cluster_type_to_node_configs(cluster_type: str, gpu_num: int) -> List[Tuple[str, int]]:
    """Generate node configs from cluster_type."""
    # Import custom cluster type configurations
    from core.bandwidth import CUSTOM_CLUSTER_NODE_TYPES
    
    node_configs: List[Tuple[str, int]] = []
    
    # Check custom cluster types (e.g., Het-4Mix)
    if cluster_type in CUSTOM_CLUSTER_NODE_TYPES:
        node_types = CUSTOM_CLUSTER_NODE_TYPES[cluster_type]
        for model in node_types:
            node_configs.append((f"Data/Topology/{model}_topo.txt", 8))
    else:
        # Legacy logic: extract GPU models from cluster_type string
        gpu_models = ["4090", "V100", "A6000", "A800", "H100_26", "H100_27", "H100_28", "H100_29"]
        extracted = [model for model in gpu_models if model in cluster_type]

        if "H100_26" in extracted:
            node_configs = [("Data/H100/H100_topo.txt", 8) for _ in extracted]
        else:
            for model in extracted:
                node_configs.append((f"Data/Topology/{model}_topo.txt", 8))

        # If nodes are insufficient, repeat in round-robin
        total = sum(count for _, count in node_configs)
        idx = 0
        while total < gpu_num and extracted:
            model = extracted[idx % len(extracted)]
            if "H100_26" in extracted:
                node_configs.append(("Data/H100/H100_topo.txt", 8))
            else:
                node_configs.append((f"Data/Topology/{model}_topo.txt", 8))
            total += 8
            idx += 1

    return node_configs


def create_gpu_to_node_map(node_configs: Sequence[Tuple[str, int]]) -> Dict[int, int]:
    """Create mapping from global GPU index to node index."""
    mapping: Dict[int, int] = {}
    start = 0
    for node_idx, (_, gpu_count) in enumerate(node_configs):
        for local_idx in range(gpu_count):
            mapping[start + local_idx] = node_idx
        start += gpu_count
    return mapping

