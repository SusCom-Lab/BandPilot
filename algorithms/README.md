# algorithms Module Guide

Implements GPU-combination search, heuristics, and baselines for training/evaluation.

## `baseline.py`
- `default_algo(total_gpu, avail_gpu, gpu_need)`: Tiered attempts by node/host structure (full node → full host → cross-node → random).
- `random_algo(total_gpu, avail_gpu, gpu_need)`: Randomly samples the required number of GPUs from available GPUs.

## `search.py`
- `generate_next_combos`: Given a 0/1 combo, generate all candidates by removing one GPU.
- `greedy_recursive_search`: Starting from an initial combo, recursively remove one GPU until meeting `gpu_need`.
- `_try_node_insert_optimization`: When full 4-GPU nodes/8-GPU hosts exist, start from whole nodes/hosts to shrink the search space.
- `_evaluate_bandwidth` / `_evaluate_global_bandwidth`: Unified bandwidth scoring via model prediction, real lookup, or `ClusterStateManager` contention-aware prediction.
- `_compare_and_select_best`: Compare multiple paths (e.g., “largest-set removal” and EHA candidates) and pick the higher-bandwidth combo.
- `tree_search_only`: One-direction “largest-set removal” searcher; can enable global bandwidth evaluation (`global_mode`).
- `improved_searching_algo`: Main scheduler combining node insertion optimization, decremental search, and EHA candidates, with support for:
  - `cluster_manager`: Uses `predict_with_contention` for bandwidth under background jobs/contention.
  - `global_mode`: Scores “current combo + remaining GPUs” as total gain.
  - `global_mode_all`: When `global_mode=True`, also adds bandwidth of already allocated combos to the global score.

## `eha.py`
- `eha_search`: Equilibrium-driven Heuristic Algorithm; generates limited candidates per node distribution and filters with model/real bandwidth.

## `slurm.py`
- `slurm_best_fit_algo`: Slurm-style best-fit; prefer single-node, then cross-node greedy.
- `k_clique_bandwidth_sampling_search` (extensible): Example weighted-sampling search flow.

## Design Notes

### Use of if_real_data

`if_real_data` is an algorithm-level flag to choose real data vs. model prediction for bandwidth.

**Principles**:
- `if_real_data` lives at the algorithm layer and should not be hardcoded inside `ClusterStateManager`.
- When using `cluster_manager`, build the bandwidth predictor via `create_bandwidth_predictor()` first, then inject it.
- If `cluster_manager` is provided, prefer `cluster_manager.predict_with_contention()` to stay aligned with online contention logic.

**Usage patterns**:
1. **Without `cluster_manager`**:  
   `improved_searching_algo` / `tree_search_only` switch between real lookup (`calculate_bandwidth_values`) and model prediction (`predict_with_model`) based on `if_real_data`.
2. **With `cluster_manager` (background jobs/contention)**:  
   - When creating `cluster_manager`, specify real vs. model via `create_bandwidth_predictor()`.  
   - Algorithms receiving `cluster_manager` call `predict_with_contention` for candidate evaluation.  
   - Under `global_mode`, they also incorporate “remaining GPUs” and historical task bandwidth into the global score.

## Recommendations
- For training and most evaluations, prefer model scoring via `predict_with_model`; set `if_real_data=True` when upper-bound estimates must align with real data.
- When `evaluation.compare` collects offline max_bw or `get_single_dispatch_with_contention_data` runs, the main model loads automatically and compares `improved_searching_algo`, `tree_search_only`, `eha_search`, `slurm_best_fit_algo`, `default_algo`, `random_algo`, etc.
- In experiments with background tasks/contention (e.g., single contention), pass `cluster_manager` (a `ClusterStateManager` instance) so algorithms account for contention and `contention_mode` occupancy modeling.

