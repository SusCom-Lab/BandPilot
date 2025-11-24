# utils 模块说明

收集与业务无关的通用辅助函数。

## `io_utils.py`
- `load_pickle_dict(path)`：加载 pickle 文件并校验结果必须是 `dict`。
- `load_json(path)` / `save_json(data, path)`：JSON 读写工具，`save_json` 会自动创建父目录。

## `helpers.py`
- `ensure_directory(path)`：确保目录存在（`mkdir -p` 的 Python 版本）。

以上工具主要被 `main.py`（创建评估输出目录）及其它模块在序列化 scaler、配置、指标时复用。需要扩展新工具时，请优先放在此目录集中管理，避免散落在业务代码中。

