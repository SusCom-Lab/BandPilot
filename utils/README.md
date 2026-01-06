# utils Module Guide

Holds general-purpose helpers unrelated to core business logic.

## `io_utils.py`
- `load_pickle_dict(path)`: Load a pickle file and ensure the result is a `dict`.
- `load_json(path)` / `save_json(data, path)`: JSON IO helpers; `save_json` auto-creates parent dirs.

## `helpers.py`
- `ensure_directory(path)`: Ensure directory existence (Python `mkdir -p`).

These helpers are used by `main.py` (e.g., creating evaluation output dirs) and other modules for scaler/config/metric serialization. Add new utilities here to keep helpers centralized.

