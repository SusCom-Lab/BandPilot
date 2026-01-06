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

