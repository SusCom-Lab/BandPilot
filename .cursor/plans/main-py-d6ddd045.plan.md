<!-- d6ddd045-acd0-4277-bf9f-ad7149e0dd60 089d09ab-6251-4fd6-b097-5e5ead19e326 -->
# 修复随机种子问题，确保每次repeat使用不同的随机序列

## 问题分析

当前问题：

1. `main.py` 中设置了全局随机种子 `set_seed(random_seed)`
2. `compare.py` 中的 `get_compare_utilization_data` 和 `get_compare_accumulation_data` 函数没有接受 `random_seed` 参数
3. `_sample_available_gpu` 函数使用 `np.random`，但没有为每次 repeat 设置不同的随机种子
4. 这导致每次 repeat 都产生完全相同的随机序列，repeat 失去意义

参考实现：

- `Auto_experiment_H100.py` 中使用 `random.seed(random_seed + r)` 为每次 repeat 设置不同的随机种子
- `multi_tenant_sim.py` 中已经为工作负载生成使用了 `random_seed + repeat_id`（正确）

## 修复方案

### 1. 修改 `compare.py` 中的 `_sample_available_gpu` 函数

- 添加 `random_seed` 参数（可选）
- 如果提供了 `random_seed`，在函数内部设置 `np.random.seed(random_seed)`
- 保持向后兼容（如果未提供 random_seed，使用全局随机状态）

### 2. 修改 `get_compare_utilization_data` 函数

- 添加 `random_seed: Optional[int] = None` 参数
- 在 `for repeat_idx in range(repeat_num)` 循环中：
- 如果 `random_seed` 不为 None，计算 `current_seed = random_seed + repeat_idx`
- 调用 `_sample_available_gpu` 时传入 `current_seed`
- 对于使用随机数的算法（如 `random_algo`, `default_algo`），也需要设置随机种子

### 3. 修改 `get_compare_accumulation_data` 函数

- 同样的修改：添加 `random_seed` 参数
- 在 repeat 循环中使用 `random_seed + repeat_idx`

### 4. 修改 `main.py` 中的调用

- 在调用 `get_compare_utilization_data` 和 `get_compare_accumulation_data` 时，传递 `random_seed=config.get("random_seed", 123)`

### 5. 检查 `multi_tenant_sim.py`

- 确认工作负载生成已经正确使用 `random_seed + repeat_id`
- 检查算法内部是否还有其他随机操作需要设置种子（如 `random_algo`, `default_algo`）

## 实施步骤

1. 修改 `_sample_available_gpu` 函数，添加 random_seed 参数
2. 修改 `get_compare_utilization_data`，添加 random_seed 参数并在 repeat 循环中使用
3. 修改 `get_compare_accumulation_data`，添加 random_seed 参数并在 repeat 循环中使用
4. 修改 `main.py`，传递 random_seed 参数
5. 检查并修复 `multi_tenant_sim.py` 中算法内部的随机操作

### To-dos

- [ ] 创建算法适配器函数，统一不同算法的接口
- [ ] 重构 run_multi_tenant_simulation，添加 search_algo 和 search_if_real_data 参数
- [ ] 修改搜索阶段逻辑，使用传入的算法函数和 search_if_real_data
- [ ] 在 compare.py 中创建 get_multi_tenant_compare_data 函数
- [ ] 更新 main.py 中的调用，保持向后兼容