"""Dataset generation utilities."""
from __future__ import annotations

import logging
import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values
from core.gpu_config import (
    generate_data_minmax_restricted,
    generate_random_gpu_config,
)

logger = logging.getLogger(__name__)


def _compute_bandwidths(
    gpu_configs: Sequence[Sequence[int]],
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
) -> List[float]:
    bandwidths: List[float] = []
    for config in gpu_configs:
        final_bw, _, _ = calculate_bandwidth_values(
            config, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
        )
        bandwidths.append(final_bw)
    return bandwidths


def get_balanced_train_dataset(
    num_samples: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data with wide coverage and balanced allocation samples."""
    gpu_configs: List[np.ndarray] = []
    bandwidths: List[float] = []
    seen = set()
    num_parts = total_gpu // 8

    densities = [round(0.1 + 0.02 * i, 3) for i in range(0, 21)] + [
        round(0.52 + 0.06 * i, 3) for i in range(0, 7)
    ]
    target_counts = [max(2, int(total_gpu * densities[i % len(densities)])) for i in range(num_samples)]
    random.shuffle(target_counts)

    num_balanced = num_samples // 2 if num_parts >= 2 else 0
    counts_balanced = target_counts[:num_balanced]
    counts_random = target_counts[num_balanced:]

    print(f"Start generating {len(counts_balanced)} balanced-allocation samples...")
    for gpus_to_allocate in counts_balanced:
        while True:
            nodes_to_use = random.randint(2, num_parts) if num_parts >= 2 else 1
            base_alloc = gpus_to_allocate // nodes_to_use
            remainder = gpus_to_allocate % nodes_to_use
            if base_alloc + (1 if remainder else 0) > 8:
                continue

            allocation_plan = [base_alloc + 1] * remainder + [base_alloc] * (nodes_to_use - remainder)
            chosen_nodes = random.sample(range(num_parts), nodes_to_use)
            config = np.zeros(total_gpu, dtype=int)
            for idx, node_id in enumerate(chosen_nodes):
                num_to_take = allocation_plan[idx]
                node_indices = range(node_id * 8, (node_id + 1) * 8)
                selected = random.sample(list(node_indices), num_to_take)
                config[selected] = 1
            key = tuple(config)
            if key not in seen:
                seen.add(key)
                gpu_configs.append(config)
                break

    print(f"Start generating {len(counts_random)} random cross-node samples...")
    for gpus_to_allocate in counts_random:
        while True:
            config = np.zeros(total_gpu, dtype=int)
            active_indices = np.random.choice(total_gpu, gpus_to_allocate, replace=False)
            config[active_indices] = 1
            parts = [int(np.sum(config[i * 8 : (i + 1) * 8])) for i in range(num_parts)]
            if num_parts >= 2 and sum(1 for s in parts if s > 0) < 2:
                continue
            key = tuple(config)
            if key not in seen:
                seen.add(key)
                gpu_configs.append(config)
                break

    print("Start computing bandwidths for all generated configs...")
    bandwidths = _compute_bandwidths(
        gpu_configs,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
    )
    print(f"Dataset generation finished, total {len(gpu_configs)} samples.")
    logger.info("Balanced training data generation finished, %s samples", len(gpu_configs))
    return np.array(gpu_configs), np.array(bandwidths)


def get_simple_balanced_train_dataset(
    num_samples: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dataset suitable for the Simple model."""
    gpu_configs: List[np.ndarray] = []
    bandwidths: List[float] = []
    seen = set()
    num_parts = total_gpu // 8

    densities = [round(0.1 + 0.02 * i, 3) for i in range(0, 21)] + [
        round(0.52 + 0.06 * i, 3) for i in range(0, 7)
    ]
    target_counts = [max(2, int(total_gpu * densities[i % len(densities)])) for i in range(num_samples)]
    random.shuffle(target_counts)
    num_balanced = num_samples // 2 if num_parts >= 2 else 0
    counts_balanced = target_counts[:num_balanced]
    counts_random = target_counts[num_balanced:]

    print(f"Start generating {len(counts_balanced)} balanced-allocation samples...")
    for gpus_to_allocate in counts_balanced:
        while True:
            nodes_to_use = random.randint(2, num_parts) if num_parts >= 2 else 1
            base_alloc = gpus_to_allocate // nodes_to_use
            remainder = gpus_to_allocate % nodes_to_use
            if base_alloc + (1 if remainder else 0) > 8:
                continue
            allocation_plan = [base_alloc + 1] * remainder + [base_alloc] * (nodes_to_use - remainder)
            chosen_nodes = random.sample(range(num_parts), nodes_to_use)
            config = np.zeros(total_gpu, dtype=int)
            for idx, node_id in enumerate(chosen_nodes):
                num_to_take = allocation_plan[idx]
                node_indices = range(node_id * 8, (node_id + 1) * 8)
                selected = random.sample(list(node_indices), num_to_take)
                config[selected] = 1
            key = tuple(config)
            if key not in seen:
                seen.add(key)
                gpu_configs.append(config)
                break

    print(f"Start generating {len(counts_random)} random samples...")
    for gpus_to_allocate in counts_random:
        while True:
            config = np.zeros(total_gpu, dtype=int)
            active_indices = np.random.choice(total_gpu, gpus_to_allocate, replace=False)
            config[active_indices] = 1
            key = tuple(config)
            if key not in seen:
                seen.add(key)
                gpu_configs.append(config)
                break

    print("Start computing bandwidths for all generated configs...")
    bandwidths = _compute_bandwidths(
        gpu_configs,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
    )
    print(f"Dataset generation finished, total {len(gpu_configs)} samples.")
    logger.info("Simple training data generation finished, %s samples", len(gpu_configs))
    return np.array(gpu_configs), np.array(bandwidths)


def get_random_train_dataset(
    num_samples: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random cross-node dataset."""
    gpu_configs: List[np.ndarray] = []
    bandwidths: List[float] = []
    num_parts = total_gpu // 8

    print(f"Start randomly generating {num_samples} cross-node samples...")
    while len(gpu_configs) < num_samples:
        density = np.random.uniform(0.1, 0.9)
        gpu_config = generate_random_gpu_config(total_gpu, density)
        if num_parts >= 2:
            parts = [int(np.sum(gpu_config[i * 8 : (i + 1) * 8])) for i in range(num_parts)]
            if sum(1 for s in parts if s > 0) < 2:
                continue
        gpu_configs.append(gpu_config)

    print("Start computing bandwidths for all random configs...")
    bandwidths = _compute_bandwidths(
        gpu_configs,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
    )
    print(f"Random dataset generation finished, total {len(gpu_configs)} samples.")
    return np.array(gpu_configs), np.array(bandwidths)


# ---------------------------------------------------------------------------
# Sampling strategies for sensitivity analysis
# ---------------------------------------------------------------------------

def get_stratified_train_dataset(
    num_samples: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified sampling: equal samples per GPU-density bin.

    Divide the density range [0.1, 0.9] into 8 equal-width bins and allocate
    ``num_samples // 8`` samples (remainder distributed round-robin) to each bin.
    Within each bin, half use balanced allocation (when possible) and half use
    random cross-node allocation - mirroring *get_balanced_train_dataset* but
    with strictly uniform density coverage.
    """
    gpu_configs: List[np.ndarray] = []
    seen: set = set()
    num_parts = total_gpu // 8

    # Divide density into 8 equal-width bins
    num_bins = 8
    bin_edges = np.linspace(0.1, 0.9, num_bins + 1)
    per_bin = num_samples // num_bins
    remainder = num_samples % num_bins
    bin_quotas = [per_bin + (1 if i < remainder else 0) for i in range(num_bins)]

    print(f"Stratified sampling: {num_bins} density bins, quotas={bin_quotas}")

    for bin_idx in range(num_bins):
        low, high = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        quota = bin_quotas[bin_idx]
        balanced_count = quota // 2 if num_parts >= 2 else 0
        random_count = quota - balanced_count

        # Balanced allocation portion
        for _ in range(balanced_count):
            attempts = 0
            while attempts < 500:
                attempts += 1
                density = np.random.uniform(low, high)
                gpus_to_allocate = max(2, int(total_gpu * density))
                nodes_to_use = random.randint(2, num_parts) if num_parts >= 2 else 1
                base_alloc = gpus_to_allocate // nodes_to_use
                rem = gpus_to_allocate % nodes_to_use
                if base_alloc + (1 if rem else 0) > 8:
                    continue
                allocation_plan = [base_alloc + 1] * rem + [base_alloc] * (nodes_to_use - rem)
                chosen_nodes = random.sample(range(num_parts), nodes_to_use)
                config = np.zeros(total_gpu, dtype=int)
                for idx, node_id in enumerate(chosen_nodes):
                    num_to_take = allocation_plan[idx]
                    node_indices = range(node_id * 8, (node_id + 1) * 8)
                    selected = random.sample(list(node_indices), num_to_take)
                    config[selected] = 1
                key = tuple(config)
                if key not in seen:
                    seen.add(key)
                    gpu_configs.append(config)
                    break

        # Random cross-node portion
        for _ in range(random_count):
            attempts = 0
            while attempts < 500:
                attempts += 1
                density = np.random.uniform(low, high)
                gpus_to_allocate = max(2, int(total_gpu * density))
                config = np.zeros(total_gpu, dtype=int)
                active_indices = np.random.choice(total_gpu, gpus_to_allocate, replace=False)
                config[active_indices] = 1
                if num_parts >= 2:
                    parts = [int(np.sum(config[i * 8 : (i + 1) * 8])) for i in range(num_parts)]
                    if sum(1 for s in parts if s > 0) < 2:
                        continue
                key = tuple(config)
                if key not in seen:
                    seen.add(key)
                    gpu_configs.append(config)
                    break

    print(f"Stratified generation done: {len(gpu_configs)} configs (requested {num_samples})")

    bandwidths = _compute_bandwidths(
        gpu_configs, total_gpu, gpu_bw_dict_list, switch_config, training_data_path,
    )
    return np.array(gpu_configs), np.array(bandwidths)


def get_worst_case_train_dataset(
    num_samples: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Worst-case sampling: adversarial density distribution + highly unbalanced allocations.

    - 70 % of samples from extreme density regions ([0.06, 0.15] or [0.85, 0.95])
    - 30 % from moderate density [0.3, 0.7]
    - All cross-node allocations are intentionally unbalanced:
      nearly all GPUs on one node while others have 1-2 GPUs.
    """
    gpu_configs: List[np.ndarray] = []
    seen: set = set()
    num_parts = total_gpu // 8

    extreme_count = int(num_samples * 0.7)
    moderate_count = num_samples - extreme_count

    print(f"Worst-case sampling: extreme={extreme_count}, moderate={moderate_count}")

    def _gen_unbalanced(gpus_to_allocate: int) -> np.ndarray:
        """Generate a highly unbalanced cross-node allocation."""
        config = np.zeros(total_gpu, dtype=int)
        if num_parts < 2:
            indices = np.random.choice(total_gpu, gpus_to_allocate, replace=False)
            config[indices] = 1
            return config

        # Pick a primary node to load most GPUs
        nodes = list(range(num_parts))
        random.shuffle(nodes)
        primary = nodes[0]
        # Put as many GPUs as possible on the primary node (up to 8)
        primary_count = min(gpus_to_allocate - 1, 8)  # at least 1 left for other nodes
        remaining = gpus_to_allocate - primary_count

        # Distribute remaining 1 or 2 GPUs across other nodes
        other_nodes = nodes[1:]
        alloc = {}
        alloc[primary] = primary_count
        for nd in other_nodes:
            if remaining <= 0:
                break
            take = min(remaining, random.choice([1, 2]))
            alloc[nd] = take
            remaining -= take
        if remaining > 0:
            # Dump any leftover on the last node used
            last = [n for n in alloc if n != primary][-1] if len(alloc) > 1 else primary
            alloc[last] = alloc.get(last, 0) + remaining

        for node_id, count in alloc.items():
            count = min(count, 8)
            node_indices = list(range(node_id * 8, (node_id + 1) * 8))
            selected = random.sample(node_indices, count)
            config[selected] = 1
        return config

    # Extreme density samples
    for _ in range(extreme_count):
        attempts = 0
        while attempts < 500:
            attempts += 1
            if random.random() < 0.5:
                density = np.random.uniform(0.06, 0.15)
            else:
                density = np.random.uniform(0.85, 0.95)
            gpus_to_allocate = max(2, min(total_gpu, int(total_gpu * density)))
            config = _gen_unbalanced(gpus_to_allocate)
            # Ensure multi-node
            if num_parts >= 2:
                parts = [int(np.sum(config[i * 8 : (i + 1) * 8])) for i in range(num_parts)]
                if sum(1 for s in parts if s > 0) < 2:
                    continue
            key = tuple(config)
            if key not in seen:
                seen.add(key)
                gpu_configs.append(config)
                break

    # Moderate density samples (still unbalanced)
    for _ in range(moderate_count):
        attempts = 0
        while attempts < 500:
            attempts += 1
            density = np.random.uniform(0.3, 0.7)
            gpus_to_allocate = max(2, min(total_gpu, int(total_gpu * density)))
            config = _gen_unbalanced(gpus_to_allocate)
            if num_parts >= 2:
                parts = [int(np.sum(config[i * 8 : (i + 1) * 8])) for i in range(num_parts)]
                if sum(1 for s in parts if s > 0) < 2:
                    continue
            key = tuple(config)
            if key not in seen:
                seen.add(key)
                gpu_configs.append(config)
                break

    print(f"Worst-case generation done: {len(gpu_configs)} configs (requested {num_samples})")

    bandwidths = _compute_bandwidths(
        gpu_configs, total_gpu, gpu_bw_dict_list, switch_config, training_data_path,
    )
    return np.array(gpu_configs), np.array(bandwidths)

