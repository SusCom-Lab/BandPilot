"""Compatibility re-exports for network-aware baselines.

The implementation lives in `algorithms.network_baselines`; this module keeps
the historical baseline-suite import path stable.
"""
from __future__ import annotations

from algorithms.network_baselines import (
    CASCORE_NAME,
    bw_greedy_algo,
    cascore_algo,
    network_locality_algo,
    normalize_network_baseline_name,
)

__all__ = [
    "CASCORE_NAME",
    "cascore_algo",
    "network_locality_algo",
    "bw_greedy_algo",
    "normalize_network_baseline_name",
]
