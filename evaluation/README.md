# evaluation Module Guide

Provides helpers for bandwidth upper-bound estimation and algorithm comparison.

## `metrics.py`
- `find_max_bw_for_k_gpus(k, gpu_bw_dict_list, total_gpu, switch_config, avail_gpu, data_path)`  
  - Enumerates best local bandwidth per node activation count from `gpu_bw_dict_list` (per-node pkl).  
  - Combines with H100 CSV cross-node lookup to build node distributions under available-GPU constraints and takes the bottleneck between cross-node and intra-node.  
  - Preserves H100 cross-node symmetry while respecting true single-node bandwidth for heterogeneous clusters like Het-4Mix, so `max_bw` is comparable under consistent physical assumptions.

## `compare.py`
- `_load_predictor`: Load `BandwidthPredictor` from weights and switch to eval mode.
- `_sample_available_gpu`: Sample available GPUs within `[test_num, total_gpu]` based on `if_dynamic`.
- `get_single_dispatch_with_contention_data(...) -> pd.DataFrame`:  
  - Single-dispatch + background-contention experiment now records row-by-row; each row corresponds to `(test_num, repeat_idx, algorithm)`.  
  - Context fields include `contention_mode/search_if_real_data_effective/background_signature/occupancy_seed/probe_job_id/max_bw_cache_file`, enabling analysis of mask/search-mode effects.  
  - Metrics include `final_bw/standalone_bw/final_utilization/standalone_utilization/elapsed_time/predict_time/contention_time/combo_signature`.  
  - Online algorithms share `probe_job_id` with offline `max_bw`; background jobs use `job_id=-1`, keeping `final_bw / max_bw` comparisons at the same occupancy ratio.  
  - `max_bw` is still derived from `ClusterStateManager`-based multi-layer heuristics: keep local top-N per node, use sorting + branch-and-bound over node distributions, and return the best-so-far if the budget exhausts—balancing accuracy and reproducibility.  
  - Typical usage:
    ```python
    df = get_single_dispatch_with_contention_data(
        repeat_num=10,
        total_gpu=32,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        model_path=Path("Models/best.pth"),
        model_cfg=model_cfg,
        cluster_type="H100_32",
        data_path="Data/H100.csv",
        bw_type="contention",
        artifact_dir=artifact_dir,
        if_dynamic=True,
        random_seed=42,
        contention_mode="common",
        search_if_real_data=False,
    )
    df.to_csv("Data/Evaluation/Single_contention_42RS.csv", index=False)
    ```
- `contention_mode` is normalized for case/whitespace and supports `common` / `intensive` / `idle`. YAML values like `Intensive` or those with spaces are lowercased to keep cache filenames, result files, and `ClusterStateManager` aligned.
- Offline cache: after setting `enable/local_top_k/...` under `config/evaluation/max_bw_offline`, run the main program to produce `MaxBW_*` CSVs. `enable_single_contention` will then force-read the cache; missing files will prompt offline collection first.
