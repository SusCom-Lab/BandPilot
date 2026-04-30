# Algorithms Module Guide

This directory implements the search algorithms and baselines used by BandPilot.

## Main Files

- `search.py`: main BandPilot scheduler, PTS path, EHA integration, and runtime-adaptive hooks.
- `eha.py`: equilibrium-driven heuristic candidate generation and internal EHA ranking.
- `hu_unit_gate.py`: shared helper for topology-aligned hierarchical-unit coarse-stage selection.
- `runtime_adaptive.py`: runtime state machine for search-path adaptive kNN.
- `adaptive_knn.py`: compare-time replay implementation of the banked adaptive kNN policy.
- `adaptive_policy.py`: resolver for legacy threshold-style PTS trigger settings.
- `contention_score.py`: structural rerank helpers used only where the evidence protocol allows them.
- `baseline.py`: random and default GPU-selection baselines.
- `slurm.py`: Slurm-style best-fit and topology-aware heuristic baselines.
- `network_baselines.py`: `CasCore` and `BWGreedy` implementations.
- `linear_bw.py`: linear predictor baseline used by `LinearBW`.

## Implementation Boundary

`if_real_data` selects real CSV lookup versus model prediction at the algorithm layer. When background contention is modeled, pass a `ClusterStateManager` so candidate evaluation uses the same contention semantics as `evaluation.compare`.

`BandPilot` uses the main `improved_searching_algo(...)` path. `PTS` bypasses final EHA comparison and is used for sidecar scalability studies. Legacy threshold-adaptive wrappers remain available for compatibility but are not the default public path.

## Public Cleanup Note

Internal optimization notes and exploratory analysis files were removed. The public directory now keeps executable algorithm source and this concise implementation map.
