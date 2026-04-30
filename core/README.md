# core Module Guide

This directory encapsulates all low-level logic for **cluster bandwidth modeling**, serving as the foundation for other modules.

## Files

### `bandwidth.py`
- `SwitchBandwidthConfig`: Maintains switch bandwidth matrix; query/set links between any two racks.
- `BandwidthLookupCache`: Global cache for CSV lookups to avoid repeated parsing.
- `calculate_bandwidth_values`: Given a GPU mask (0/1), split into 8-GPU nodes and query global and per-node bandwidth.
- `config_to_bandwidth` / `prepare_model_inputs`: Batch-convert GPU configs to model inputs (part bandwidths, node active counts, total active count).
- `get_gpu_counts_for_model`: Count node activity per config for higher-level analysis.

### `topology.py`
- `parse_topo_matrix`: Parse single-node NVLink topology text into a DataFrame.
- `build_composite_topo_matrix`: Stitch node lists into a full-cluster topology matrix; cross-node links marked as `INTER`.
- `get_link_weight` / `calculate_connectivity_score`: Map topology connectivity to weights for Slurm scoring.
- `convert_cluster_type_to_node_configs`: Generate node configs (topology file path + GPU count) from a `cluster_type` string.
- `create_gpu_to_node_map`: Build lookup from global GPU index to node index.

### `gpu_config.py`
- `generate_random_gpu_config`: Randomly generate 0/1 masks by activation ratio.
- `generate_data_minmax` / `generate_data_minmax_restricted`: Generate samples within specified count ranges, optionally constrained to available GPUs.

### `cluster_state.py`
- `create_bandwidth_predictor`: Factory for bandwidth prediction functions, supporting real-data and model modes.
- `ClusterStateManager`: Manages GPU allocation and contention in multi-tenant scenarios.
  - **Contention logic**:
    - Only cross-node jobs interfere; single-node jobs do not contend with others.
    - Cross-node jobs use the full standalone bandwidth in contention calculations (not partial bandwidth).
    - Two cross-node jobs with shared nodes are considered contending.
    - If a super combo in either direction exceeds 8 GPUs on any node, that GPU set is deemed non-concurrent and skipped-no canonicalization to "full".
    - **Bidirectional super combo capacity**:
      - For candidate C and existing job J (sharing nodes), compute both directions:
        - From C's view: `super_C = C + project(J, nodes_of_C)`
        - From J's view: `super_J = J + project(C, nodes_of_J)`
      - Compute bandwidth for both and take the minimum: `capacity = min(f(super_C), f(super_J))`
      - If multiple related jobs exist, compute bidirectional capacity per pair; the bottleneck capacity is the minimum across pairs.
    - Total demand = candidate standalone bandwidth + demand of all related jobs:
      - `contention_mode='common'`: allocated jobs use 25%-75% of peak as "actual occupied bandwidth". The new job is modeled with peak bandwidth to ensure probe/commit `final_bw` reflects potential upper bound. This occupancy is sampled once at commit time and stored for future reads; the new job does not immediately lower its own demand in this contention round.
      - Other modes: all jobs use standalone bandwidth.
    - If total demand exceeds bottleneck capacity, bandwidth is proportionally split.
    - `predict_with_contention` and `allocate_job` stay symmetric: commit no longer pairs the new job with itself for super combos, only with other active jobs.
  - `predict_with_contention`: Probe API predicting bandwidth under contention.
  - `allocate_job`: Commit API that allocates GPUs, updates state, and adjusts impacted jobs' bandwidth.

## Recommendations
- Before reading real CSVs, ensure paths in `config/default_config.yaml` are correct.
- Reuse `prepare_model_inputs` in training/evaluation to keep feature normalization consistent.
- When using `ClusterStateManager`, create the predictor via `create_bandwidth_predictor` first, then pass it in.

