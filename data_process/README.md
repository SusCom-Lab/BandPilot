# data Module Guide

Data logic in three layers: raw CSV parsing, sample generation, batch loading & normalization.

## `preprocessing.py`
- `analyze_gpu_pattern(pattern)`: Map multi-node GPU activity to `(total_active, active_nodes, node_distribution)` keys.
- `preprocess_gpu_data(file_path)`: Read real-sampled CSV (`GPU_Mapping_Across_Nodes`, `Bandwidth(GB/s)`) and build lookup `dict[key] -> [(mapping, bw), ...]`.
- `find_matching_bandwidth(test_data, lookup_table)`: Find bandwidth by node distribution for `core.bandwidth`.

## `dataset.py`
- `get_balanced_train_dataset`: Generate training samples covering various densities/node distributions with 50% cross-node data.
- `get_simple_balanced_train_dataset`: Balanced samples for `SimpleBandwidthPredictor`.
- `get_random_train_dataset`: Pure-random cross-node test/train samples.
- `_compute_bandwidths`: Internal helper calling `calculate_bandwidth_values` for real bandwidths.

## `dataloader.py`
- `get_group_data_loader` / `get_group_test_loader`: For the main model (with node features); standardize part bandwidths, node counts, and total counts via `StandardScaler`, and serialize scalers to `artifact_dir`.
- `get_simple_group_data_loader` / `get_simple_group_test_loader`: For the simplified model (direct 0/1 masks).
- `_save_scaler` / `_load_scaler`: Centralized scaler file management for consistent train/eval.
- Normalization artifacts and model files carry sample-size suffixes: `*_ns{num_train_samples}.pkl/.pth`, and `artifact_dir` stores `active_num_train_samples.txt` to avoid mixing scales.

## Recommendations
- All scalers live in `artifact_dir` (usually `model/<cluster_type>`). Ensure they exist before evaluation or inference.
- If you swap in a new CSV, run the BandPilot generation script or call `preprocess_gpu_data` to validate formatting.

## New dataset: `Data/H100_24/Pune_H100_16M_binary.csv`
- Goal: Build a 3-node subset (8 GPUs per node, up to 24 GPUs) from H100 real data, covering all `Total_GPU_Count`=2–24 mapping patterns uniquely.
- Source: `Data/H100_Real/Pune_H100_16M_binary.csv`, truncating the 4th node from `GPU_Mapping_Across_Nodes` and recomputing active counts.
- Rules:
  - Keep samples whose truncated active count is between 2–24.
  - Rewrite `Total_GPU_Count` using the 3-node active count.
  - Deduplicate identical truncated mappings (keep first) to ensure uniqueness per pattern.
  - Sort by `Total_GPU_Count` and mapping string for reproducibility.
- Generation example (run after `conda activate gpu_dp_opt` from repo root):
  ```
  python - <<'PY'
  import os, ast, json
  import pandas as pd
  src = 'Data/H100_Real/Pune_H100_16M_binary.csv'
  out_dir = 'Data/H100_24'
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, 'Pune_H100_16M_binary.csv')
  df = pd.read_csv(src)
  truncated = []
  counts = []
  for mapping_str in df['GPU_Mapping_Across_Nodes']:
      mapping = ast.literal_eval(mapping_str)
      mapping3 = mapping[:3]                       # Truncate to the first 3 nodes
      truncated.append(mapping3)
      counts.append(sum(sum(row) for row in mapping3))
  df['GPU_Mapping_Across_Nodes'] = [json.dumps(m) for m in truncated]
  df['Total_GPU_Count'] = counts
  df = df[df['Total_GPU_Count'].between(2, 24)].copy()
  df['__key'] = df['GPU_Mapping_Across_Nodes']
  df = df.drop_duplicates(subset='__key', keep='first').drop(columns='__key')
  df = df[['OP','Total_GPU_Count','GPU_Mapping_Across_Nodes','data_size(B)','Bandwidth(GB/s)']]
  df = df.sort_values(by=['Total_GPU_Count','GPU_Mapping_Across_Nodes']).reset_index(drop=True)
  df.to_csv(out_path, index=False)
  print('saved to', out_path)
  PY
  ```
  After running, verify in terminal: unique active counts cover `[2..24]`, total rows 163.

## New dataset: `Data/H100_16/Pune_H100_16M_binary.csv`
- Goal: Build a 2-node subset (8 GPUs per node, up to 16 GPUs) from H100 real data, covering all `Total_GPU_Count`=2–16 mapping patterns uniquely.
- Source: `Data/H100_Real/Pune_H100_16M_binary.csv`, truncating to the first 2 nodes and recomputing active counts.
- Rules:
  - Keep samples whose truncated active count is between 2–16.
  - Rewrite `Total_GPU_Count` using the 2-node active count.
  - Deduplicate identical truncated mappings (keep first) to ensure uniqueness per pattern.
  - Sort by `Total_GPU_Count` and mapping string for reproducibility.
- Generation example (run after `conda activate gpu_dp_opt` from repo root):
  ```
  python - <<'PY'
  import os, ast, json
  import pandas as pd
  src = 'Data/H100_Real/Pune_H100_16M_binary.csv'
  out_dir = 'Data/H100_16'
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, 'Pune_H100_16M_binary.csv')
  df = pd.read_csv(src)
  truncated = []
  counts = []
  for mapping_str in df['GPU_Mapping_Across_Nodes']:
      mapping = ast.literal_eval(mapping_str)
      mapping2 = mapping[:2]                       # Truncate to the first 2 nodes
      truncated.append(mapping2)
      counts.append(sum(sum(row) for row in mapping2))
  df['GPU_Mapping_Across_Nodes'] = [json.dumps(m) for m in truncated]
  df['Total_GPU_Count'] = counts
  df = df[df['Total_GPU_Count'].between(2, 16)].copy()
  df['__key'] = df['GPU_Mapping_Across_Nodes']
  df = df.drop_duplicates(subset='__key', keep='first').drop(columns='__key')
  df = df[['OP','Total_GPU_Count','GPU_Mapping_Across_Nodes','data_size(B)','Bandwidth(GB/s)']]
  df = df.sort_values(by=['Total_GPU_Count','GPU_Mapping_Across_Nodes']).reset_index(drop=True)
  df.to_csv(out_path, index=False)
  print('saved to', out_path)
  PY
  ```
  After running, verify in terminal: unique active counts cover `[2..16]`, total rows 43.

