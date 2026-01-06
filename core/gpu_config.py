"""GPU configuration generation and utilities."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np


def generate_random_gpu_config(total_gpu: int, active_ratio: float = 0.5) -> np.ndarray:
    """Randomly generate a GPU config according to active_ratio."""
    if not 0 <= active_ratio <= 1:
        raise ValueError("active_ratio must be in [0, 1]")
    gpu_config = np.zeros(total_gpu, dtype=int)
    active_count = int(total_gpu * active_ratio)
    if active_count:
        active_indices = np.random.choice(total_gpu, active_count, replace=False)
        gpu_config[active_indices] = 1
    return gpu_config


def generate_data_minmax(num_samples: int, num_dimensions: int, min_ones: int, max_ones: int) -> np.ndarray:
    """Batch-generate configs with 1-count between [min_ones, max_ones]."""
    if min_ones > max_ones:
        raise ValueError("min_ones cannot be greater than max_ones")
    data = np.zeros((num_samples, num_dimensions), dtype=int)
    num_ones_array = np.random.randint(min_ones, max_ones + 1, size=num_samples)
    for idx, num_ones in enumerate(num_ones_array):
        indices = np.random.choice(num_dimensions, num_ones, replace=False)
        data[idx, indices] = 1
    return data


def generate_data_minmax_restricted(
    num_samples: int,
    num_dimensions: int,
    min_ones: int,
    max_ones: int,
    avail_gpu: Iterable[int],
) -> np.ndarray:
    """Generate configs within avail_gpu whose 1-count lies in [min_ones, max_ones]."""
    avail_gpu = list(avail_gpu)
    if not avail_gpu:
        raise ValueError("avail_gpu cannot be empty")
    if max_ones > len(avail_gpu):
        raise ValueError("max_ones exceeds number of available GPUs")

    data = np.zeros((num_samples, num_dimensions), dtype=int)
    num_ones_array = np.random.randint(min_ones, max_ones + 1, size=num_samples)
    for idx, num_ones in enumerate(num_ones_array):
        indices = np.random.choice(avail_gpu, num_ones, replace=False)
        data[idx, indices] = 1
    return data

