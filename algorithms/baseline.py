"""Baseline GPU allocation algorithms."""
from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np


def random_algo(total_gpu: int, avail_gpu: Sequence[int], gpu_need: int) -> np.ndarray:
    """Randomly select GPUs from available set."""
    if gpu_need > len(avail_gpu):
        raise ValueError("Requested GPU count exceeds available GPUs")
    best_gpu = np.zeros(total_gpu, dtype=int)
    selected_gpu = np.random.choice(avail_gpu, gpu_need, replace=False)
    for gpu in selected_gpu:
        best_gpu[gpu] = 1
    return best_gpu


def default_algo(total_gpu: int, avail_gpu: Sequence[int], gpu_need: int, verbose: bool = False) -> np.ndarray:
    """Node/host-priority GPU allocation algorithm."""
    if gpu_need > len(avail_gpu):
        raise ValueError("Requested GPU count exceeds available GPUs")

    avail_gpu = list(avail_gpu)
    best_gpu = np.zeros(total_gpu, dtype=int)
    sorted_avail_gpu = sorted(avail_gpu)
    node_size = 4
    host_size = 8

    def _group_by(size: int) -> dict[int, List[int]]:
        grouped: dict[int, List[int]] = {}
        for gpu in sorted_avail_gpu:
            group_id = gpu // size
            grouped.setdefault(group_id, []).append(gpu)
        return grouped

    grouped_nodes = _group_by(node_size)
    grouped_hosts = _group_by(host_size)

    def _select_from_host(host_id: int, need: int) -> List[int]:
        host_gpus = grouped_hosts.get(host_id, [])
        if not host_gpus or need <= 0:
            return []
        node_groups: dict[int, List[int]] = {}
        for gpu in host_gpus:
            node_id = gpu // node_size
            node_groups.setdefault(node_id, []).append(gpu)

        node_items = sorted(node_groups.items(), key=lambda item: (-len(item[1]), item[0]))
        taken: List[int] = []
        remaining = need
        used_full_nodes = set()

        for node_id, g_list in node_items:
            if remaining < node_size:
                break
            if len(g_list) == node_size:
                taken.extend(sorted(g_list))
                remaining -= node_size
                used_full_nodes.add(node_id)
                if remaining == 0:
                    return taken

        for node_id, g_list in node_items:
            if remaining == 0:
                break
            if node_id in used_full_nodes:
                continue
            available = sorted(g_list)
            take_now = min(len(available), remaining)
            taken.extend(available[:take_now])
            remaining -= take_now

        if remaining > 0:
            for node_id in used_full_nodes:
                if remaining == 0:
                    break
                g_list = sorted(node_groups[node_id])
                already_taken = set(taken)
                leftover = [gpu for gpu in g_list if gpu not in already_taken]
                if not leftover:
                    continue
                take_now = min(len(leftover), remaining)
                taken.extend(leftover[:take_now])
                remaining -= take_now

        return taken

    def _finalize(selection: List[int]) -> np.ndarray:
        best_gpu.fill(0)
        for gpu in sorted(selection):
            best_gpu[gpu] = 1
        return best_gpu

    # 1) Try to use single-node configurations
    if gpu_need <= node_size:
        node_candidates = [
            (node_id, gpus) for node_id, gpus in grouped_nodes.items() if len(gpus) >= gpu_need
        ]
        node_candidates.sort(key=lambda item: (-len(item[1]), item[0]))
        if node_candidates:
            selected = sorted(node_candidates[0][1])[:gpu_need]
            return _finalize(selected)

    # 2) Try to use single-host configurations
    host_candidates = [
        (host_id, gpus) for host_id, gpus in grouped_hosts.items() if len(gpus) >= gpu_need
    ]
    host_candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    if host_candidates:
        host_id = host_candidates[0][0]
        selected = _select_from_host(host_id, gpu_need)
        if len(selected) == gpu_need:
            return _finalize(selected)

    # 3) Fill multi-host configurations: take nodes from remaining hosts sequentially
    ordered_hosts = sorted(grouped_hosts.items(), key=lambda item: (-len(item[1]), item[0]))
    aggregated: List[int] = []
    for host_id, gpus in ordered_hosts:
        if len(aggregated) >= gpu_need:
            break
        need = min(gpu_need - len(aggregated), len(gpus))
        aggregated.extend(_select_from_host(host_id, need))

    if len(aggregated) >= gpu_need:
        return _finalize(aggregated[:gpu_need])

    # 4) Fallback: use the lowest-numbered available cards
    return _finalize(sorted_avail_gpu[:gpu_need])

