# evaluation 模块说明

提供带宽上界估计与算法比较的辅助工具。

## `metrics.py`
- `find_max_bw_for_k_gpus(k, gpu_bw_dict_list, total_gpu, switch_config, avail_gpu, data_path)`  
  依据查表结果和可用 GPU 约束，计算当前设定下可实现的最大带宽，用作各算法对比的基准。

## `compare.py`
- `_load_predictor`：从权重文件加载 `BandwidthPredictor` 并切换到 eval 模式。
- `_sample_available_gpu`：根据 `if_dynamic` 在 `[test_num, total_gpu]` 范围内采样可用 GPU。
- `get_compare_utilization_data(...) -> pd.DataFrame`：  
  逐个测试 `test_num`（2..total_gpu-1），比较 `BandDisp`、`Default`、`Tree`、`EHA`、`Topo`、`Random` 等算法的**带宽占最优比例**。  
  返回 DataFrame，`main.py` 会写入 `Data/Evaluation/.../Part_mean_*.csv`。
- `get_compare_accumulation_data(...) -> pd.DataFrame`：  
  与上类似，但记录 `max_bw - 实际带宽` 的差值（越小越好），输出 `Part_sum_*.csv`。

## `multi_tenant_sim.py`
- `run_multi_tenant_simulation(...) -> pd.DataFrame`：  
  多租户仿真模块，模拟多个任务依次到达并分配GPU资源的场景。该模块分为两个阶段：
  
  **阶段一：搜索阶段（可配置评估模式）**
  - 使用 `search_if_real_data` 指定的评估模式创建 `ClusterStateManager`
  - 对每个任务使用传入的 `search_algo` 搜索最优GPU组合（考虑多租户争用）
  - 分配资源并记录GPU组合和带宽值
  
  **阶段二：评估阶段（使用真实数据）**
  - 使用真实数据（`if_real_data=True`）创建新的 `ClusterStateManager`
  - 按照相同的顺序重新分配所有任务，使用真实数据计算带宽
  - 使用真实数据重新计算争用情况下的带宽值
  - 记录真实数据下的带宽值、争用比例等
  
  返回 DataFrame 包含每个任务的详细信息：
  - 搜索阶段：`predicted_standalone_bw`, `predicted_final_bw`
  - 评估阶段：`real_standalone_bw`, `real_final_bw`, `real_contention_ratio`（真实数据值）
  
  **算法适配器**：
  - `create_search_algo_adapter(...) -> Callable`：创建算法适配器，统一不同算法的接口
  - 支持需要模型参数的算法（如 `improved_searching_algo`, `tree_search_only`, `eha_search`）
  - 支持需要拓扑参数的算法（如 `slurm_best_fit_algo`）
  - 支持简单算法（如 `default_algo`, `random_algo`）
  
  `main.py` 会写入 `Data/Evaluation/.../multi_tenant_simulation.csv`。

## `compare.py` 中的多租户对比函数
- `get_multi_tenant_compare_data(...) -> pd.DataFrame`：  
  运行多个算法的多租户仿真并对比结果。
  
  **功能**：
  - 接受多个算法配置（算法函数 + 名称 + search_if_real_data）
  - 对每个算法运行 `run_multi_tenant_simulation`
  - 收集所有结果并合并为统一的 DataFrame
  - 添加 `algorithm_name` 和 `search_mode` 列，便于对比分析
  
  **使用示例**：
  ```python
  from evaluation.compare import get_multi_tenant_compare_data
  from algorithms.search import improved_searching_algo
  from algorithms.eha import eha_search
  from algorithms.baseline import default_algo
  
  algorithm_configs = [
      {"name": "BandDisp_Model", "algo": improved_searching_algo, "search_if_real_data": False},
      {"name": "BandDisp_GT", "algo": improved_searching_algo, "search_if_real_data": True},
      {"name": "EHA", "algo": eha_search, "search_if_real_data": False},
      {"name": "Default", "algo": default_algo, "search_if_real_data": False},
  ]
  
  df = get_multi_tenant_compare_data(
      ...,
      algorithm_configs=algorithm_configs,
  )
  ```

## 使用方式
- 在 `config/default_config.yaml` 中开启：
  ```yaml
  model:
    type: 'full'        # 使用 BandwidthPredictor
  evaluation:
    enable_utilization: true
    enable_accumulation: true
    enable_multi_tenant: true  # 启用多租户仿真
    multi_tenant:
      workload_mode: 'fixed_sum'  # 或 'random'
      total_gpu_sum: 32
      job_sizes: [1, 2, 4, 8]
      repeat_num: 1
      if_real_data: false
  ```
- 运行 `python main.py` 后，训练完成即会生成对应的 CSV 文件，可再结合 notebook 绘图。

