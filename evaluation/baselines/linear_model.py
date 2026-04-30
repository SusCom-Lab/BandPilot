"""Compatibility re-export for the `LinearBW` baseline model.

The implementation lives in `algorithms.linear_bw`; this module keeps the
historical baseline-suite import path stable.
"""
from __future__ import annotations

from algorithms.linear_bw import LinearBandwidthRegressor

__all__ = ["LinearBandwidthRegressor"]
