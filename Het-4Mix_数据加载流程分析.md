# Het-4Mix 数据加载流程详细分析

## 概述

本文档详细追踪了当 `cluster_types` 为 `Het-4Mix` 时的数据加载和构造流程，用于核查潜在的逻辑问题。

## 1. 配置读取阶段

**位置**: `main.py:166-172`

```python
for cluster_type in cluster_cfg["cluster_types"]:  # cluster_type = 'Het-4Mix'
    switch_config = SwitchBandwidthConfig(
        num_machines=total_gpu // 8,  # 32 // 8 = 4
        cluster_type=cluster_type,   # 'Het-4Mix'
    )
    file_list = get_gpu_dict_files(cluster_type, repeat=total_gpu // 8)  # repeat=4
    gpu_bw_dict_list = load_gpu_bandwidth_dicts(bandwidth_dir, file_list)
```

**关键参数**:
- `total_gpu = 32` (从 config 读取)
- `num_machines = 4` (32 // 8)
- `cluster_type = 'Het-4Mix'`
- `repeat = 4` (需要4个节点的字典文件)

## 2. GPU 字典文件列表生成

**位置**: `core/bandwidth.py:95-99`

```python
def get_gpu_dict_files(cluster_type: str, repeat: int) -> List[str]:
    if cluster_type in CUSTOM_CLUSTER_NODE_TYPES:  # 'Het-4Mix' 匹配
        node_types = CUSTOM_CLUSTER_NODE_TYPES[cluster_type]  # ["4090", "A800", "A6000", "V100"]
        return _expand_gpu_types_for_nodes(node_types, repeat)  # repeat=4
```

**CUSTOM_CLUSTER_NODE_TYPES 定义** (`core/bandwidth.py:82-84`):
```python
CUSTOM_CLUSTER_NODE_TYPES = {
    "Het-4Mix": ["4090", "A800", "A6000", "V100"],
}
```

**文件列表生成逻辑** (`core/bandwidth.py:87-92`):
```python
def _expand_gpu_types_for_nodes(node_types: Sequence[str], repeat: int) -> List[str]:
    if repeat <= 0 or not node_types:
        return []
    cycles = math.ceil(repeat / len(node_types))  # ceil(4/4) = 1
    ordered = list(node_types) * cycles  # ["4090", "A800", "A6000", "V100"] * 1
    return [f"{gpu}_gpu_bw_dict.pkl" for gpu in ordered[:repeat]]  # 取前4个
```

**生成的文件列表**:
```python
file_list = [
    "4090_gpu_bw_dict.pkl",    # 节点0
    "A800_gpu_bw_dict.pkl",    # 节点1
    "A6000_gpu_bw_dict.pkl",   # 节点2
    "V100_gpu_bw_dict.pkl"     # 节点3
]
```

**⚠️ 潜在问题 1**: 文件顺序是硬编码的 `["4090", "A800", "A6000", "V100"]`，这意味着：
- 节点0（GPU 0-7）**固定**使用 4090 的字典
- 节点1（GPU 8-15）**固定**使用 A800 的字典
- 节点2（GPU 16-23）**固定**使用 A6000 的字典
- 节点3（GPU 24-31）**固定**使用 V100 的字典

如果实际的物理节点顺序不同，会导致节点与字典的映射错误。

## 3. 字典文件加载

**位置**: `main.py:62-66`

```python
def load_gpu_bandwidth_dicts(bandwidth_dir: Path, file_list: List[str]):
    dicts = []
    for filename in file_list:
        dicts.append(load_gpu_bw_dict(bandwidth_dir / filename))
    return dicts
```

**加载过程** (`core/bandwidth.py:71-79`):
```python
def load_gpu_bw_dict(file_path: Path) -> Dict:
    """从pickle文件中加载GPU带宽字典。"""
    if not file_path.exists():
        raise FileNotFoundError(f"Bandwidth dictionary not found: {file_path}")
    with file_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"File {file_path} did not contain a dictionary.")
    return data
```

**最终结果**:
```python
gpu_bw_dict_list = [
    {4090的8卡模式字典},  # 索引0，对应节点0
    {A800的8卡模式字典},  # 索引1，对应节点1
    {A6000的8卡模式字典}, # 索引2，对应节点2
    {V100的8卡模式字典}   # 索引3，对应节点3
]
```

每个字典的键是长度为8的元组（表示节点内8个GPU的使用模式），值是带宽值（GB/s）。

## 4. 跨节点带宽查找表加载

**位置**: `core/bandwidth.py:155`

```python
lookup_table = BandwidthLookupCache.ensure_loaded(Path(data_path))
```

**数据路径**: `config/default_config.yaml:3`
```yaml
h100_data_path: './Data/H100_Real/Pune_H100_16M_binary.csv'
```

**⚠️ 潜在问题 2**: Het-4Mix 使用的是 **H100 的跨节点带宽数据**，而不是异构集群的实际跨节点带宽。这是设计上的选择（见 README.md:124），但需要确认是否符合预期。

**查找表结构** (`data/preprocessing.py:37-76`):
- 从 CSV 读取 `GPU_Mapping_Across_Nodes` 和 `Bandwidth(GB/s)` 列
- 使用 `analyze_gpu_pattern` 生成查找键：`(total_active, num_active_nodes, sorted_active_counts)`
- 键值对：`{(total_active, num_nodes, counts): [(mapping_str, bandwidth), ...]}`

## 5. 带宽计算流程

**位置**: `core/bandwidth.py:121-198` - `calculate_bandwidth_values`

### 5.1 GPU 配置解析

```python
# 将 total_gpu=32 的配置按每8个GPU分组为4个节点
nodes_config = []
for idx in range(0, total_gpu, 8):  # idx = 0, 8, 16, 24
    node_slice = gpu[idx : idx + 8]  # 每个节点8个GPU
    node_list = [int(x) for x in node_slice]
    nodes_config.append(node_list)
```

**示例**: 如果 `gpu = [1,1,0,0,0,0,0,0, 0,0,1,1,0,0,0,0, ...]`
- `nodes_config[0] = [1,1,0,0,0,0,0,0]` (节点0，GPU 0-7)
- `nodes_config[1] = [0,0,1,1,0,0,0,0]` (节点1，GPU 8-15)
- `nodes_config[2] = [0,0,0,0,0,0,0,0]` (节点2，GPU 16-23)
- `nodes_config[3] = [0,0,0,0,0,0,0,0]` (节点3，GPU 24-31)

### 5.2 跨节点带宽查找

```python
result = find_matching_bandwidth(nodes_config, lookup_table)
if result is not None:
    _, bandwidth = result
    final_bandwidth = float(bandwidth)  # 从H100 CSV查到的跨节点带宽
else:
    final_bandwidth = 0.0  # 查找失败
```

**查找逻辑** (`data/preprocessing.py:79-111`):
1. 使用 `analyze_gpu_pattern` 分析 `nodes_config`，生成键：`(total_active, num_active_nodes, sorted_active_counts)`
2. 在 `lookup_table`（H100数据）中查找匹配的键
3. 返回第一个匹配项的带宽值

**⚠️ 潜在问题 3**: 查找是基于**模式匹配**（活跃GPU总数、活跃节点数、各节点活跃GPU数），而不是精确的节点分配。这意味着：
- 不同的节点分配可能匹配到相同的键
- 返回的是第一个匹配项，可能不是最准确的

### 5.3 节点内带宽查找

```python
parts = [tuple(int(x) for x in gpu[idx : idx + 8]) for idx in range(0, total_gpu, 8)]
part_bandwidths: List[float] = []
for idx, part_tuple in enumerate(parts):
    current_dict = gpu_bw_dict_list[idx]  # 根据节点索引选择字典
    part_bandwidths.append(float(round(current_dict.get(part_tuple, 0.0), 2)))
```

**关键逻辑**:
- `parts[0]` (节点0的8卡模式) → 使用 `gpu_bw_dict_list[0]` (4090字典)
- `parts[1]` (节点1的8卡模式) → 使用 `gpu_bw_dict_list[1]` (A800字典)
- `parts[2]` (节点2的8卡模式) → 使用 `gpu_bw_dict_list[2]` (A6000字典)
- `parts[3]` (节点3的8卡模式) → 使用 `gpu_bw_dict_list[3]` (V100字典)

**⚠️ 潜在问题 4**: 这里假设了节点索引与GPU类型的一一对应关系：
- 节点0 → 4090
- 节点1 → A800
- 节点2 → A6000
- 节点3 → V100

如果实际的物理拓扑不是这个顺序，会导致错误的带宽计算。

### 5.4 Het-4Mix 特殊处理：瓶颈带宽计算

```python
cluster_label = getattr(switch_config, "cluster_type", None)
if cluster_label in CUSTOM_CLUSTER_NODE_TYPES:  # 'Het-4Mix' 匹配
    active_bws = [
        part_bandwidths[idx]
        for idx, part in enumerate(parts)
        if any(part)  # 只考虑有活跃GPU的节点
    ]
    if active_bws:
        intra_bottleneck = min(active_bws)  # 节点内最小带宽
        final_bandwidth = float(min(final_bandwidth, intra_bottleneck))  # 取跨节点和节点内的最小值
```

**逻辑说明**:
1. 收集所有有活跃GPU的节点的节点内带宽
2. 取最小值作为节点内瓶颈带宽 (`intra_bottleneck`)
3. 最终带宽 = `min(跨节点带宽, 节点内瓶颈带宽)`

**⚠️ 潜在问题 5**: 这个逻辑假设：
- 跨节点带宽来自H100数据（可能不准确）
- 节点内瓶颈是正确的（取决于字典映射是否正确）
- 取最小值是合理的（符合瓶颈理论，但需要验证）

## 6. 数据流总结

```
配置文件 (default_config.yaml)
  ↓
main.py: 读取 cluster_types = ['Het-4Mix']
  ↓
get_gpu_dict_files('Het-4Mix', repeat=4)
  ↓
生成文件列表: ["4090_gpu_bw_dict.pkl", "A800_gpu_bw_dict.pkl", "A6000_gpu_bw_dict.pkl", "V100_gpu_bw_dict.pkl"]
  ↓
load_gpu_bandwidth_dicts: 加载4个字典文件
  ↓
gpu_bw_dict_list[0] = 4090字典 (节点0)
gpu_bw_dict_list[1] = A800字典  (节点1)
gpu_bw_dict_list[2] = A6000字典 (节点2)
gpu_bw_dict_list[3] = V100字典  (节点3)
  ↓
calculate_bandwidth_values 被调用时:
  1. 从H100 CSV查找跨节点带宽 (lookup_table)
  2. 从对应字典查找节点内带宽 (gpu_bw_dict_list[idx])
  3. 对于Het-4Mix，取两者的最小值作为最终带宽
```

## 7. 发现的潜在问题汇总

### 问题1: 节点与GPU类型的硬编码映射
- **位置**: `core/bandwidth.py:83` 和 `_expand_gpu_types_for_nodes`
- **问题**: 节点索引与GPU类型的对应关系是固定的，如果物理拓扑不同会导致错误
- **影响**: 节点内带宽查找可能使用错误的字典

### 问题2: 跨节点带宽使用H100数据
- **位置**: `core/bandwidth.py:155` 和 `config/default_config.yaml:3`
- **问题**: Het-4Mix使用H100的跨节点带宽，可能不准确
- **影响**: 跨节点带宽可能被高估或低估

### 问题3: 模式匹配的模糊性
- **位置**: `data/preprocessing.py:19-34` 和 `find_matching_bandwidth`
- **问题**: 查找基于模式而非精确分配，可能返回不准确的带宽
- **影响**: 跨节点带宽可能不准确

### 问题4: 节点索引与字典索引的强耦合
- **位置**: `core/bandwidth.py:184`
- **问题**: `gpu_bw_dict_list[idx]` 直接使用节点索引，假设了固定的GPU类型分配
- **影响**: 如果节点顺序不同，会使用错误的字典

### 问题5: 瓶颈带宽计算的假设
- **位置**: `core/bandwidth.py:188-196`
- **问题**: 假设取最小值是正确的，但需要验证是否符合实际物理限制
- **影响**: 最终带宽可能不准确

## 8. 建议的验证方法

1. **验证节点顺序**: 检查实际的物理节点顺序是否与代码中的假设一致
2. **验证字典映射**: 确认每个节点的GPU类型是否与 `gpu_bw_dict_list` 的索引对应
3. **验证跨节点带宽**: 检查H100数据是否适用于Het-4Mix的跨节点通信
4. **验证模式匹配**: 检查 `analyze_gpu_pattern` 生成的键是否能准确区分不同的配置
5. **验证瓶颈计算**: 通过实际测试验证 `min(跨节点, 节点内)` 是否合理

## 9. 代码关键位置索引

- **配置定义**: `config/default_config.yaml:31-33`
- **集群类型定义**: `core/bandwidth.py:82-84`
- **文件列表生成**: `core/bandwidth.py:95-106`
- **字典加载**: `main.py:62-66`, `core/bandwidth.py:71-79`
- **跨节点查找**: `core/bandwidth.py:155`, `data/preprocessing.py:37-76`, `data/preprocessing.py:79-111`
- **节点内查找**: `core/bandwidth.py:181-185`
- **瓶颈计算**: `core/bandwidth.py:187-196`

