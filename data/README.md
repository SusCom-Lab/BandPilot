# data 模块说明

数据相关逻辑分为三层：原始 CSV 解析、样本生成、批量加载与归一化。

## `preprocessing.py`
- `analyze_gpu_pattern(pattern)`：将多节点 GPU 活跃情况映射为 `(总激活数, 激活节点数, 节点分布)` 的键。
- `preprocess_gpu_data(file_path)`：读取真实采样 CSV（`GPU_Mapping_Across_Nodes`, `Bandwidth(GB/s)`），构建查找表 `dict[key] -> [(mapping, bw), ...]`。
- `find_matching_bandwidth(test_data, lookup_table)`：根据节点分布查找对应带宽，供 `core.bandwidth` 使用。

## `dataset.py`
- `get_balanced_train_dataset`：生成覆盖多种密度/节点分布的训练样本，保证跨节点数据占 50%。
- `get_simple_balanced_train_dataset`：为 `SimpleBandwidthPredictor` 提供均衡样本集。
- `get_random_train_dataset`：纯随机方式生成跨节点测试/训练样本。
- `_compute_bandwidths`：内部工具，调用 `calculate_bandwidth_values` 获取真实带宽。

## `dataloader.py`
- `get_group_data_loader` / `get_group_test_loader`：面向主模型（有节点特征），会将 part 带宽、节点活跃数、总活跃数分别做 `StandardScaler` 归一化，并将 scaler 序列化到 `artifact_dir`。
- `get_simple_group_data_loader` / `get_simple_group_test_loader`：面向简化模型（直接用 0/1 掩码）。
- `_save_scaler` / `_load_scaler`：统一管理 scaler 文件，保证训练/评估一致。

## 使用建议
- 所有 scaler 会保存在 `artifact_dir`（通常是 `model/<cluster_type>`）。评估或推理前需确保对应文件存在。
- 若替换新的 CSV，请先运行 `BandPilot` 中的生成脚本或自行调用 `preprocess_gpu_data` 以验证格式正确。

