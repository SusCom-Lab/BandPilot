# SC_BandPilot 重构版

该目录是对 `BandPilot/Auto_experiment_H100.py` 3885 行单体脚本的模块化重构，实现 GPU 带宽建模、搜索与评估全流程。

## 目录结构

```
SC_BandPilot/
├── main.py                  # 主入口，读取配置，驱动训练/评估
├── config/
│   └── default_config.yaml  # 默认配置
├── core/                    # 核心逻辑：带宽查表、拓扑、GPU配置
├── models/                  # 神经网络模型
├── data/                    # 数据预处理与加载
├── algorithms/              # 搜索与启发式算法
├── training/                # 训练与评估封装
├── evaluation/              # 利用率/累积对比
├── utils/                   # IO、工具函数
└── Data/, model/            # 与原目录兼容的数据/模型文件
```

## 快速开始

```bash
cd SC_BandPilot
python main.py --config config/default_config.yaml
```

默认配置会在 `config/default_config.yaml` 中设置数据路径、模型结构、训练超参以及集群/带宽参数，可按需修改。

## 主要模块

- `core.bandwidth`：带宽查表缓存、Switch 配置、模型输入构造
- `core.topology`：拓扑矩阵解析、复合矩阵拼接、节点映射
- `models.bandwidth_predictor` / `models.simple_predictor`：Transformer 模型及简化版本
- `data.preprocessing/dataset/dataloader`：数据预处理、样本生成、归一化与 DataLoader
- `algorithms.*`：包括贪心、树搜索、EHA、Slurm BestFit 等策略
- `training.trainer/evaluator`：统一的训练循环与评估方法
- `evaluation.metrics/compare`：带宽上界估计、利用率和累积差距统计

## 评估开关

`config/default_config.yaml` 的 `evaluation` 段包含：

- `enable_utilization`: 是否在训练后运行 `get_compare_utilization_data`（生成利用率对比）
- `enable_accumulation`: 是否运行 `get_compare_accumulation_data`（生成累积差距对比）
- `enable_multi_tenant`: 是否运行多租户仿真（生成 `multi_tenant_simulation.csv`）
- `repeat_num`: 每种 GPU 数量的采样次数（用于 utilization/accumulation）
- `if_dynamic`: 是否按动态可用 GPU 数量采样（用于 utilization/accumulation）

> 只有在 `model.type` 为 `full`（即 `BandwidthPredictor`）时才会执行上述评估。

### 多租户仿真配置

多租户仿真模拟多个任务依次到达并分配GPU资源的场景，能够检测和计算资源争用对带宽的影响。在 `evaluation.multi_tenant` 子配置段中可设置：

- `workload_mode`: 工作负载生成模式
  - `'fixed_sum'`: 生成总和为 `total_gpu_sum` 的随机任务序列（默认）
  - `'random'`: 随机生成 `num_jobs` 个任务
- `total_gpu_sum`: fixed_sum 模式下的总GPU数（默认32）
- `num_jobs`: random 模式下的任务数量（默认10）
- `job_sizes`: 允许的任务大小列表（默认 `[1, 2, 4, 8]`）
- `repeat_num`: 重复仿真次数（默认1）
- `if_real_data`: 是否使用真实数据（默认false，使用模型预测）

**输出文件**: `Data/Evaluation/{cluster_type}/multi_tenant_simulation.csv`

CSV 文件包含以下列：
- `job_id`: 任务ID
- `gpu_need`: 需要的GPU数量
- `combo`: GPU组合（格式如 "0,1,2,3"）
- `predicted_standalone_bw`: 搜索阶段的独占带宽（模型预测）
- `predicted_final_bw`: 搜索阶段的最终带宽（模型预测，考虑争用）
- `real_standalone_bw`: 评估阶段的独占带宽（真实数据）
- `real_final_bw`: 评估阶段的最终带宽（真实数据，考虑争用）
- `real_contention_ratio`: 真实数据下的争用比例 (real_final_bw / real_standalone_bw)
- `real_cluster_throughput`: 真实数据下的集群总吞吐量
- `num_active_jobs`: 当前活跃任务数

**重要说明**：
- 搜索阶段可以使用模型预测或真实数据（通过 `search_if_real_data` 参数控制）
- 评估阶段始终使用真实数据（`if_real_data=True`）重新计算所有任务的带宽值
- 评估阶段的争用计算也使用真实数据，确保结果准确反映实际性能
- 支持多种搜索算法的对比，通过 `evaluation.compare.get_multi_tenant_compare_data` 函数可以运行多个算法并对比结果
- 多租户仿真新增 GPU 合法性校验：`ClusterStateManager` 会在 probe/commit 前检查组合是否超出单节点容量（默认 8 张卡）或复用已分配 GPU；`evaluation.multi_tenant_sim` 也会在算法返回组合后再次校验，若发现非法资源请求会直接跳过并记录日志，避免脏数据污染仿真结果

**使用示例**:
```yaml
evaluation:
  enable_multi_tenant: true
  multi_tenant:
    workload_mode: 'fixed_sum'
    total_gpu_sum: 32
    job_sizes: [1, 2, 4, 8]
    repeat_num: 1
    if_real_data: false
```

## 依赖

详见 `requirements.txt`。安装示例：

```bash
pip install -r requirements.txt
```

## 数据与模型

为兼容原始脚本，仍需以下目录：

- `SC_BandPilot/Data`：带宽CSV、拓扑文件等
- `SC_BandPilot/model`：模型参数及 scaler artifacts 输出目录

## 后续工作

- 按需扩展 `evaluation` 目录中的输出统计、可视化
- 增补单元测试覆盖核心模块
- 将更多原脚本中的实验入口迁移至 `main.py`
