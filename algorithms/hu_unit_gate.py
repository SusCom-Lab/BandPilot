"""Shared HU coarse-stage unit gate helpers.

The helpers centralize topology-aligned removal units for PTS. The default
public policy uses `8 -> 1`. Aggressive scalability sidecars may try larger
aligned units first, then degrade to smaller legal units when required by the
`gpu_need + 8` target-capacity rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class HUUnitGateConfig:
    """Configuration for HU coarse-stage unit selection."""

    aggressive: bool = False
    host_gpu_count: int = 8
    pod_gpu_count: int = 32
    pods_per_upper_switch: int = 4


def resolve_hu_unit_gate_config(
    config: Optional[Mapping[str, Any]] = None,
) -> HUUnitGateConfig:
    """Resolve a raw mapping into `HUUnitGateConfig`."""

    raw = dict(config or {})
    return HUUnitGateConfig(
        aggressive=bool(raw.get("aggressive", False)),
        host_gpu_count=max(1, int(raw.get("host_gpu_count", 8))),
        pod_gpu_count=max(1, int(raw.get("pod_gpu_count", 32))),
        pods_per_upper_switch=max(2, int(raw.get("pods_per_upper_switch", 4))),
    )


def normalize_hu_unit_gate_config(
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Return a JSON/YAML-serializable HU unit-gate config."""

    resolved = resolve_hu_unit_gate_config(config)
    return {
        "aggressive": bool(resolved.aggressive),
        "host_gpu_count": int(resolved.host_gpu_count),
        "pod_gpu_count": int(resolved.pod_gpu_count),
        "pods_per_upper_switch": int(resolved.pods_per_upper_switch),
    }


def resolve_hu_unit_sizes(
    *,
    num_dimensions: int,
    gate_config: Optional[Mapping[str, Any]] = None,
) -> List[int]:
    """Resolve candidate HU unit sizes for the current problem dimension.

    Default mode returns only the host-level unit. Aggressive mode adds larger
    topology-aligned units such as pod and upper-switch groups when legal.
    """

    resolved = resolve_hu_unit_gate_config(gate_config)
    if int(num_dimensions) <= 0:
        return [int(resolved.host_gpu_count)]
    if not resolved.aggressive:
        return [int(resolved.host_gpu_count)]

    levels = {int(resolved.host_gpu_count)}
    pod_gpu_count = int(resolved.pod_gpu_count)
    if int(num_dimensions) >= pod_gpu_count:
        levels.add(pod_gpu_count)
        current = pod_gpu_count * int(resolved.pods_per_upper_switch)
        while current <= int(num_dimensions):
            levels.add(int(current))
            current *= int(resolved.pods_per_upper_switch)
    return sorted(levels, reverse=True)


def build_active_unit_groups(
    current_combo: np.ndarray,
    *,
    num_dimensions: int,
    unit_size: int,
) -> Tuple[List[List[int]], List[int]]:
    """Group selected GPUs into active topology units and return their unit ids."""

    combo = np.asarray(current_combo, dtype=int)
    selected_indices = np.where(combo == 1)[0].astype(int).tolist()
    group_count = max(1, math.ceil(int(num_dimensions) / int(unit_size)))
    groups: List[List[int]] = [[] for _ in range(group_count)]
    for gpu_idx in selected_indices:
        groups[min(group_count - 1, int(gpu_idx) // int(unit_size))].append(int(gpu_idx))

    active_groups: List[List[int]] = []
    active_group_ids: List[int] = []
    for group_id, members in enumerate(groups):
        if members:
            active_groups.append(members)
            active_group_ids.append(int(group_id))
    return active_groups, active_group_ids


def build_unit_candidate_combos(
    current_combo: np.ndarray,
    *,
    removable_units: Sequence[Sequence[int]],
) -> np.ndarray:
    """Build coarse candidates by removing one active topology unit at a time."""

    base = np.asarray(current_combo, dtype=int)
    candidate_count = len(removable_units)
    candidate_combos = np.repeat(base[np.newaxis, :], candidate_count, axis=0)
    for row_idx, unit_members in enumerate(removable_units):
        candidate_combos[row_idx, np.asarray(list(unit_members), dtype=int)] = 0
    return candidate_combos
