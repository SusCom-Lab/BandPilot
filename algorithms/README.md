# algorithms 模块说明

实现 GPU 组合搜索、启发式与基线算法，供训练/评估时调用。

## `baseline.py`
- `default_algo(total_gpu, avail_gpu, gpu_need)`：根据节点/主机结构逐级尝试（完整节点→完整主机→跨节点→随机）。
- `random_algo(total_gpu, avail_gpu, gpu_need)`：在可用 GPU 中随机抽取指定数量。

## `search.py`
- `generate_next_combos` / `generate_add_combos`：用于递归搜索的候选生成。
- `find_best_2gpu_combo`：遍历两卡组合作为搜索起点。
- `greedy_recursive_search`：每次移除一个 GPU，直至满足需求。
- `tree_search_only`：单方向搜索，实现原脚本的“最大集合剔除”策略。
- `improved_searching_algo`：综合递减与启发式路径，是主调度算法。

## `eha.py`
- `eha_search`：Equilibrium-driven Heuristic Algorithm，根据节点资源分布生成有限候选并使用模型/真实带宽筛选最优解。

## `slurm.py`
- `slurm_best_fit_algo`：模仿 Slurm 的 best-fit 策略，优先单节点，再跨节点贪心。
- `k_clique_bandwidth_sampling_search`（可扩展）：示例性的带权采样搜索流程。

## 架构设计说明

### if_real_data 参数的使用

`if_real_data` 是算法级别的参数，用于决定使用真实数据还是模型预测来计算带宽。

**重要原则**：
- `if_real_data` 是算法绑定的，不应该硬编码在 `ClusterStateManager` 中
- 当使用 `cluster_manager` 时，算法的 `if_real_data` 参数应该与创建 `cluster_manager` 时使用的 `if_real_data` 一致
- `ClusterStateManager` 通过 `create_bandwidth_predictor()` 工厂函数接收预测函数，保持通用性

**使用模式**：
1. **不使用 cluster_manager**：算法根据 `if_real_data` 参数直接使用真实数据或模型预测
2. **使用 cluster_manager**：
   - 创建 `cluster_manager` 时，使用 `create_bandwidth_predictor()` 根据 `if_real_data` 创建预测函数
   - 算法传入 `cluster_manager` 参数，并保持 `if_real_data` 参数与创建 `cluster_manager` 时一致
   - 算法优先使用 `cluster_manager.predict_with_contention()` 进行带宽评估（考虑多租户争用）

## 使用建议
- 训练过程中（预测模型打分）使用 `predict_with_model`，评估真实数据则设 `if_real_data=True`。
- 启用 `evaluation.compare` 时会自动加载主模型并调用 `improved_searching_algo`、`tree_search_only`、`eha_search`、`slurm_best_fit_algo` 等进行对比。
- 多租户场景下，使用 `cluster_manager` 参数传入 `ClusterStateManager` 实例，算法会自动考虑资源争用。

