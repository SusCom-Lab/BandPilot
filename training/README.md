# training 模块说明

封装训练循环、评估与推理工具，供 `main.py` 调用。

## `trainer.py`
- `train_model`：主模型（带节点特征）的训练模板，支持 Huber 损失、EWC、Cosine 学习率、早停。
- `train_simple_model`：简化模型的训练模板。
- `model_train_pipeline`：
  - 调用 `get_balanced_train_dataset` + `get_group_data_loader` 生成数据。
  - 训练 `BandwidthPredictor` 并在随机测试集上评估、保存 `bandwidth_predictor.pth`。
  - 返回 `(mse, mae, model_path)` 供 `main.py` 后续评估使用。
- `simple_model_train_pipeline`：
  - 对 `SimpleBandwidthPredictor` 进行同样的流程，保存 `simple_bandwidth_predictor.pth`。

## `evaluator.py`
- `predict_with_model`：针对多节点配置使用 scaler 归一化后批量推理；单节点直接返回局部带宽。
- `evaluate_model` / `evaluate_simple_model`：在测试集上计算真实 MSE/MAE，并将预测结果反归一化。

## 使用建议
- `artifact_dir`（一般为 `model/<cluster_type>`）同时保存 scaler 与模型权重，确保训练与评估一致。
- 当 `config.model.type` 设置为 `full` 时，`main.py` 会基于 `model_path` 继续执行利用率/累积对比。

