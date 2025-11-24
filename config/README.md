# config 目录说明

保存所有运行时配置，默认通过 `config/default_config.yaml` 传入 `main.py`。

## default_config.yaml 字段

```yaml
data:
  h100_data_path         # 带宽 CSV / 查表数据
  bandwidth_dict_dir     # GPU 局部带宽字典（pkl）目录
  model_save_dir         # 训练过程中生成模型与 scaler 的输出目录
  evaluation_dir         # 利用率/累积比较结果输出目录

model:
  type                   # 'simple' 使用 SimpleBandwidthPredictor；'full' 使用 BandwidthPredictor
  hidden_dim / ...       # Transformer 结构超参

training:
  batch_size, num_epochs, learning_rate 等常规训练参数
  num_train_samples      # 调用 dataset 生成的训练样本量
  num_test_samples       # 评估/比较阶段的随机样本数量

cluster:
  total_gpu              # 集群 GPU 总数（需为 8 的倍数）
  cluster_types          # 列表，可配置多个不同的拓扑组合
  bw_switch              # 交换机带宽类型（同步传入评估文件名）

evaluation:
  enable_utilization     # true 时输出利用率 (`Part_mean_*.csv`)
  enable_accumulation    # true 时输出与最优差值 (`Part_sum_*.csv`)
  repeat_num             # 每个 test_num 的采样次数
  if_dynamic             # true 则随机采样可用 GPU 数量；false 表示所有 GPU 可用

random_seed / device     # 全局随机数种子与训练设备
```

> 若要复现原脚本 “训练 + 对比评估” 的行为：  
> 1. 将 `model.type` 设置为 `'full'`；  
> 2. 在 `evaluation` 中至少开启 `enable_utilization`；  
> 3. 运行 `python main.py --config config/default_config.yaml`，结果会落在 `data.evaluation_dir/<cluster_type>/`。

