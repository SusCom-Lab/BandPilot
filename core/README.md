# core 模块说明

该目录封装所有与**集群带宽建模**相关的底层逻辑，是其它模块的基础。

## 文件说明

### `bandwidth.py`
- `SwitchBandwidthConfig`：维护交换机带宽矩阵，可查询/设置任意两机架之间的链路。
- `BandwidthLookupCache`：全局缓存 CSV 查表结果，避免重复解析。
- `calculate_bandwidth_values`：给定 GPU 掩码（0/1），分割为 8 卡节点并查表得到全局带宽、节点带宽。
- `config_to_bandwidth` / `prepare_model_inputs`：批量将 GPU 配置转换为模型输入（part 带宽、节点活跃 GPU 数、总激活数）。
- `get_gpu_counts_for_model`：统计单个配置的节点活跃数，供上层分析。

### `topology.py`
- `parse_topo_matrix`：解析单节点 NVLink 拓扑文本为 DataFrame。
- `build_composite_topo_matrix`：根据节点列表拼接成整集群拓扑矩阵，跨节点链路统一记为 `INTER`。
- `get_link_weight` / `calculate_connectivity_score`：将拓扑连通性映射为权重，用于 Slurm 算法打分。
- `convert_cluster_type_to_node_configs`：根据 `cluster_type` 字符串生成节点配置（拓扑文件路径 + GPU 数量）。
- `create_gpu_to_node_map`：构建 “全局 GPU 索引 → 节点索引” 的查找表。

### `gpu_config.py`
- `generate_random_gpu_config`：按激活比例随机生成 0/1 掩码。
- `generate_data_minmax` / `generate_data_minmax_restricted`：生成指定数量范围的样本，可限定可用 GPU 集合。

### `cluster_state.py`
- `create_bandwidth_predictor`：创建带宽预测函数工厂，支持真实数据模式和模型预测模式。
- `ClusterStateManager`：集群状态管理器，管理多租户场景下的GPU资源分配和争用检测。
  - **争用计算逻辑**：
    - 只有跨节点的任务才会相互干扰，单节点任务不与其他任务产生争用。
    - 对于跨节点任务，使用**完整任务的独立带宽**参与争用计算，而不是部分带宽。
    - 当两个跨节点任务有共享节点时，它们会参与争用计算。
    - **双向Super Combo容量计算**：
      - 对于候选任务C和已有任务J（有共享节点），计算两个方向的super combo：
        - 从C视角：`super_C = C + project(J, C的节点集合)`
        - 从J视角：`super_J = J + project(C, J的节点集合)`
      - 分别计算这两个super combo的带宽容量，取最小值：`capacity = min(f(super_C), f(super_J))`
      - 如果有多个相关任务，对每个任务对都计算双向容量，最终的瓶颈容量 = 所有双向容量的最小值
    - 总需求 = 候选任务的独立带宽 + 所有相关任务的完整独立带宽之和。
    - 若总需求 > 瓶颈容量，按比例瓜分带宽。
  - `predict_with_contention`：Probe接口，预测候选组合在考虑争用后能获得的带宽。
  - `allocate_job`：Commit接口，正式分配GPU给任务，更新状态，并修正受影响任务的带宽。

## 使用建议
- 读取真实 CSV 前请确认 `config/default_config.yaml` 中的数据路径与文件一致。
- 训练、评估阶段都应复用 `prepare_model_inputs`，确保特征归一化维度一致。
- 使用 `ClusterStateManager` 时，应先通过 `create_bandwidth_predictor` 创建预测函数，再传入管理器。

