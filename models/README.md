# models Module Guide

Contains all neural network models and base layers; currently includes the Transformer main model and a simplified version.

## `layers.py`
- `PositionalEncoding`: Standard sinusoidal positional encoding, supports up to 100 sequence elements.
- `AttentionPooling`: Weighted aggregation over Transformer outputs to produce a fixed-length vector.

## `bandwidth_predictor.py`
- `BandwidthPredictor`:
  - Inputs: `x_bw` (part bandwidth sequence), `x_node_counts` (active GPU count per node), `x_total_counts` (total active GPUs).
  - Concatenates bandwidth and node counts, projects into a Transformer encoder, then uses `AttentionPooling` + MLP to predict bandwidth.
  - Built-in EWC (Elastic Weight Consolidation) support: `save_old_params()`, `update_fisher()`, `ewc_loss()`.

## `simple_predictor.py`
- `SimpleBandwidthPredictor`:
  - Input is a GPU 0/1 mask (e.g., 32-dim), reshaped to `(batch, num_nodes, 8)`.
  - Uses a single Transformer encoder + attention pooling to predict bandwidth.

## Recommendations
- Main and simplified models share most hyperparameters (`hidden_dim`, `num_layers`, `num_heads`), configured in `config/default_config.yaml`.
- To enable EWC, training loops should call `save_old_params` and `update_fisher` before task switches; currently only `save_old_params` is called by default (extend as needed).

