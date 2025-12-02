## 多租户仿真结果文件命名规则说明

多租户仿真结果保存在 `Data/Evaluation/<cluster_type>/` 下面，各个集群类型仍然使用单独的子文件夹，例如：

- `Data/Evaluation/H100_Real/H100_26H100_27H100_28H100_29/`
- `Data/Evaluation/H100_Real/Het-4Mix/` （如启用）

在每个 `cluster_type` 子目录中，多租户仿真结果 CSV 文件的命名规则为：

`MTS_{random_seed}RS_{num_train_samples}TD_{contention_mode}CM_{repeat_num}RN.csv`

其中字段含义为：

- **MTS_{random_seed}RS**: `random_seed`，来自顶层配置 `config.random_seed`
- **{num_train_samples}TD**: 训练样本数，来自 `training.num_train_samples`
- **{contention_mode}CM**: 多租户仿真争用模式，来自 `evaluation.multi_tenant.contention_mode`
- **{repeat_num}RN**: 多租户仿真重复次数，来自 `evaluation.multi_tenant.repeat_num`

示例：

- `MTS_1111RS_500TD_commonCM_100RN.csv`

表示：

- 随机种子 `random_seed = 1111`
- 训练样本数 `num_train_samples = 500`
- 争用模式 `contention_mode = common`
- 多租户仿真重复次数 `repeat_num = 100`


