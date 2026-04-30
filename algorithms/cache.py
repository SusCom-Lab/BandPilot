"""Dispatch-scoped caching for bandwidth prediction and search results.

Three-layer cache design for eliminating redundant computation within a single
algorithm execution (one dispatch event):

- L1 (InferenceCache): config vector -> contention-aware bandwidth score.
  Eliminates repeated predict_with_contention calls for the same GPU config.

- L2 (SubproblemCache): (node_gpus, target_count) -> best sub-config.
  Eliminates repeated _run_subset_tree_search calls in EHA Phase 2.

- C0 (SuperComboCache): canonical super_combo -> raw bandwidth prediction.
  Eliminates repeated _predict_combo_bandwidth calls on the same canonical
  super combo inside ClusterStateManager contention logic.

All caches are dispatch-scoped: created at algorithm entry, destroyed on return.
No cross-dispatch or cross-algorithm sharing.

Usage:
    bundle = DispatchCacheBundle()
    # pass bundle.l1 to search functions
    # pass bundle.l2 to EHA _run_subset_tree_search
    # pass bundle.c0 to ClusterStateManager via set_super_combo_cache()
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np


class InferenceCache:
    """L1: config -> contention-aware bandwidth cache.

    Caches the final output of predict_with_contention (or equivalent) for a
    given GPU config vector.  Within a single dispatch, the same config always
    yields the same bandwidth because the cluster state (background jobs) does
    not change.
    """

    __slots__ = ("_cache", "_max_size", "hits", "misses")

    def __init__(self, max_size: int = 200_000) -> None:
        self._cache: Dict[Tuple[int, ...], float] = {}
        self._max_size = max_size
        self.hits: int = 0
        self.misses: int = 0

    def get(self, config: np.ndarray) -> Optional[float]:
        key = tuple(config.astype(int).flat)
        val = self._cache.get(key)
        if val is not None:
            self.hits += 1
        else:
            self.misses += 1
        return val

    def put(self, config: np.ndarray, bandwidth: float) -> None:
        if len(self._cache) >= self._max_size:
            # Simple eviction: clear all.  In practice the limit is generous
            # enough that this should rarely trigger within a single dispatch.
            self._cache.clear()
        self._cache[tuple(config.astype(int).flat)] = float(bandwidth)

    def get_batch(
        self, configs: np.ndarray
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """Partition a batch into cache hits and misses.

        Returns:
            (hit_indices, miss_indices, hit_values)
        """
        hit_idx: List[int] = []
        miss_idx: List[int] = []
        hit_vals: List[float] = []
        for i in range(len(configs)):
            val = self.get(configs[i])
            if val is not None:
                hit_idx.append(i)
                hit_vals.append(val)
            else:
                miss_idx.append(i)
        return hit_idx, miss_idx, np.asarray(hit_vals, dtype=float) if hit_vals else np.array([], dtype=float)

    def put_batch(self, configs: np.ndarray, bandwidths: np.ndarray) -> None:
        """Insert multiple entries at once."""
        for i in range(len(configs)):
            self.put(configs[i], float(bandwidths[i]))

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": self.size}


class SubproblemCache:
    """L2: (node_gpus, target_count) -> best sub-config cache.

    Caches the result of _run_subset_tree_search so that identical
    (node_gpus, target_count) sub-problems in EHA Phase 2 are solved only once.
    """

    __slots__ = ("_cache", "hits", "misses")

    def __init__(self) -> None:
        self._cache: Dict[Tuple[FrozenSet[int], int], np.ndarray] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(
        self, node_gpus: Sequence[int], target_count: int
    ) -> Optional[np.ndarray]:
        key = (frozenset(int(g) for g in node_gpus), int(target_count))
        val = self._cache.get(key)
        if val is not None:
            self.hits += 1
            return val.copy()
        self.misses += 1
        return None

    def put(
        self, node_gpus: Sequence[int], target_count: int, config: np.ndarray
    ) -> None:
        key = (frozenset(int(g) for g in node_gpus), int(target_count))
        self._cache[key] = config.copy()

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": self.size}


class SuperComboCache:
    """C0: canonical super_combo -> raw bandwidth prediction cache.

    Caches _predict_combo_bandwidth results on canonical super combos inside
    ClusterStateManager.  Multiple candidate combos may produce the same
    canonical super combo when contending with the same background job, so
    this layer avoids redundant model/data lookups.
    """

    __slots__ = ("_cache", "hits", "misses")

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, ...], float] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, combo: np.ndarray) -> Optional[float]:
        key = tuple(combo.astype(int).flat)
        val = self._cache.get(key)
        if val is not None:
            self.hits += 1
        else:
            self.misses += 1
        return val

    def put(self, combo: np.ndarray, bandwidth: float) -> None:
        self._cache[tuple(combo.astype(int).flat)] = float(bandwidth)

    def get_batch(
        self, combos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deduplicate and partition a batch of combos.

        Returns:
            (hit_mask, hit_values, unique_miss_indices, miss_inverse)
            - hit_mask: bool array, True for cached entries
            - hit_values: float array with cached BWs (0.0 for misses)
            - unique_miss_indices: indices of unique misses to predict
            - miss_inverse: for each miss entry, index into unique_miss_indices;
              hit entries are set to -1
        """
        n = len(combos)
        hit_mask = np.zeros(n, dtype=bool)
        results = np.zeros(n, dtype=float)
        # Track unique misses to avoid duplicate predictions
        miss_key_to_idx: Dict[Tuple[int, ...], int] = {}
        unique_miss_idx: List[int] = []
        miss_inverse = np.full(n, -1, dtype=int)

        for i in range(n):
            key = tuple(combos[i].astype(int).flat)
            val = self._cache.get(key)
            if val is not None:
                hit_mask[i] = True
                results[i] = val
                self.hits += 1
            else:
                if key not in miss_key_to_idx:
                    miss_key_to_idx[key] = len(unique_miss_idx)
                    unique_miss_idx.append(i)
                miss_inverse[i] = miss_key_to_idx[key]
                self.misses += 1

        return hit_mask, results, np.array(unique_miss_idx, dtype=int), miss_inverse

    def put_batch(self, combos: np.ndarray, bandwidths: np.ndarray) -> None:
        for i in range(len(combos)):
            self.put(combos[i], float(bandwidths[i]))

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": self.size}


class DispatchCacheBundle:
    """Container for all dispatch-scoped caches.

    Created at the entry point of each algorithm call and passed down to
    sub-functions.  Destroyed when the algorithm returns.
    """

    __slots__ = ("l1", "l2", "c0")

    def __init__(self) -> None:
        self.l1 = InferenceCache()
        self.l2 = SubproblemCache()
        self.c0 = SuperComboCache()

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {
            "l1_inference": self.l1.stats(),
            "l2_subproblem": self.l2.stats(),
            "c0_super_combo": self.c0.stats(),
        }
