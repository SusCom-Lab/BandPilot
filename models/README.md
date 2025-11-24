# models 模块说明

封装所有神经网络模型及基础层，当前包含 Transformer 主模型与简化版两套实现。

## `layers.py`
- `PositionalEncoding`：标准正弦/余弦位置编码，支持最多 100 个序列元素。
- `AttentionPooling`：对 Transformer 输出进行加权聚合，生成固定长度向量。

## `bandwidth_predictor.py`
- `BandwidthPredictor`：
  - 输入：`x_bw`（part 带宽序列）、`x_node_counts`（对应节点活跃 GPU 数）、`x_total_counts`（总活跃数）。
  - 将带宽与节点计数拼接后投影到 Transformer 编码器，再用 `AttentionPooling` + MLP 预测。
  - 内置 EWC（Elastic Weight Consolidation）支持：`save_old_params()`、`update_fisher()`、`ewc_loss()`。

## `simple_predictor.py`
- `SimpleBandwidthPredictor`：
  - 输入直接为 GPU 0/1 掩码（例如 32 维），reshape 为 `(batch, num_nodes, 8)`。
  - 使用单一 Transformer 编码 + 注意力池化输出带宽预测。

## 使用建议
- 主模型与简化模型共享大部分超参（`hidden_dim`, `num_layers`, `num_heads`），在 `config/default_config.yaml` 中配置。
- 若要启用 EWC，训练循环需在任务切换前调用 `save_old_params` 和 `update_fisher`，当前默认只调用 `save_old_params`（可按需扩展）。

