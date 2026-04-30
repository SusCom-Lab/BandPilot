"""Resolve legacy threshold-style PTS trigger settings.

The current runtime-adaptive search path uses `RuntimeAdaptiveKNNState`. This
module remains for compatibility with threshold-style sidecars that still need
the old `cv_threshold`, `gap_threshold`, and `min_candidates_for_cv` tuple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from algorithms.contention_score import is_homogeneous_cluster
from core.bandwidth import SwitchBandwidthConfig


@dataclass(frozen=True)
class AdaptiveThresholds:
    """Resolved legacy threshold tuple."""

    cv_threshold: float
    gap_threshold: float
    min_candidates_for_cv: int
    policy_mode: str
    policy_bucket: str


# Preserve the pre-policy threshold tuple as the explicit fallback.
LEGACY_DEFAULT_THRESHOLDS = AdaptiveThresholds(
    cv_threshold=0.05,
    gap_threshold=0.03,
    min_candidates_for_cv=5,
    policy_mode="legacy_global",
    policy_bucket="global",
)

# Default reviewer-facing policy: split thresholds by cluster homogeneity.
DEFAULT_ADAPTIVE_THRESHOLD_POLICY = {
    "mode": "homogeneity_split",
    "homogeneous": {
        "cv_threshold": 0.10,
        "gap_threshold": 0.08,
        "min_candidates_for_cv": 4,
    },
    "heterogeneous": {
        "cv_threshold": 0.05,
        "gap_threshold": 0.03,
        "min_candidates_for_cv": 5,
    },
}


def _coerce_threshold_block(
    block: Optional[Mapping[str, Any]],
    *,
    fallback: AdaptiveThresholds,
    policy_mode: str,
    policy_bucket: str,
) -> AdaptiveThresholds:
    """Normalize a threshold block and fill missing fields from the fallback."""
    block = block or {}
    return AdaptiveThresholds(
        cv_threshold=float(block.get("cv_threshold", fallback.cv_threshold)),
        gap_threshold=float(block.get("gap_threshold", fallback.gap_threshold)),
        min_candidates_for_cv=int(
            block.get("min_candidates_for_cv", fallback.min_candidates_for_cv)
        ),
        policy_mode=policy_mode,
        policy_bucket=policy_bucket,
    )


def resolve_adaptive_thresholds(
    *,
    policy_cfg: Optional[Mapping[str, Any]],
    switch_config: Optional[SwitchBandwidthConfig],
    cluster_manager: Optional[Any] = None,
    fallback_thresholds: Optional[Mapping[str, Any]] = None,
) -> AdaptiveThresholds:
    """Resolve legacy threshold values from policy config and fallbacks."""
    fallback = _coerce_threshold_block(
        fallback_thresholds,
        fallback=LEGACY_DEFAULT_THRESHOLDS,
        policy_mode="fallback",
        policy_bucket="global",
    )
    if not policy_cfg:
        return fallback

    mode = str(policy_cfg.get("mode", "global")).strip().lower()
    if mode == "global":
        # Supported global-policy shapes:
        # 1) {"mode": "global", "global": {...}}
        # 2) {"mode": "global", "cv_threshold": ...}
        block = policy_cfg.get("global")
        if not isinstance(block, Mapping):
            block = policy_cfg
        return _coerce_threshold_block(
            block,
            fallback=fallback,
            policy_mode=mode,
            policy_bucket="global",
        )

    if mode == "homogeneity_split":
        is_homogeneous = is_homogeneous_cluster(
            switch_config=switch_config,
            cluster_manager=cluster_manager,
        )
        bucket = "homogeneous" if is_homogeneous else "heterogeneous"
        return _coerce_threshold_block(
            policy_cfg.get(bucket),
            fallback=fallback,
            policy_mode=mode,
            policy_bucket=bucket,
        )

    raise ValueError(
        f"Unsupported adaptive threshold policy mode: {mode!r}. "
        "Expected 'global' or 'homogeneity_split'."
    )
