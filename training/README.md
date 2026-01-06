# training Module Guide

Wraps training loops, evaluation, and inference utilities for `main.py`.

## `trainer.py`
- `train_model`: Training template for the main model (with node features); supports Huber loss, EWC, cosine LR, and early stopping.
- `train_simple_model`: Training template for the simplified model.
- `model_train_pipeline`:
  - Uses `get_balanced_train_dataset` + `get_group_data_loader` to build data.
  - Trains `BandwidthPredictor`, evaluates on a random test set, saves `bandwidth_predictor_ns{num_train_samples}.pth`, and writes `active_num_train_samples.txt` so inference can auto-select the scaler.
  - Returns `(mse, mae, model_path)` for downstream evaluation in `main.py`.
- `simple_model_train_pipeline`:
  - Same flow for `SimpleBandwidthPredictor`, saving `simple_bandwidth_predictor_ns{num_train_samples}.pth`.

## `evaluator.py`
- `predict_with_model`: Batch inference for multi-node configs with scaler normalization; single-node returns local bandwidth directly.
- `evaluate_model` / `evaluate_simple_model`: Compute true MSE/MAE on test sets and inverse-transform predictions.

## Recommendations
- `artifact_dir` (typically `model/<cluster_type>`) stores both scalers and model weights to keep training and evaluation consistent.
- When `config.model.type` is `full`, `main.py` continues utilization/cumulative comparisons based on `model_path`.

## `variable_length_experiment.py`
- Purpose: Validate Transformer transfer when GPU counts change (16→32, 24→32).
- Features:
  - Reuse `model_train_pipeline` to train 16-GPU/24-GPU base models.
  - Load `Data/H100_Real/Pune_H100_16M_binary.csv` and exclude GPU configs seen in `Data/H100_16/` and `Data/H100_24/` so 32-GPU stage only sees unseen samples.
  - Derive reproducible random seeds from one master seed to run repeated “base training → 32-GPU finetune → 32-GPU eval”.
  - Outputs live in `Data/Evaluation/VariableLengthStudy/<H100_XXGPU>/seed_xxx/`, including model checkpoints, scalers, prediction tables, and metric summaries.
- Run example (activate `conda activate gpu_dp_opt` first):
  ```bash
  python training/variable_length_experiment.py \
    --config config/default_config.yaml \
    --master-seed 1111 \
    --num-runs 3 \
    --output-dir Data/Evaluation/VariableLengthStudy \
    --base-num-train-samples 300 \
    --base-num-test-samples 400 \
    --finetune-train-count 100 \
    --finetune-eval-count 200 \
    --finetune-epochs 200 \
    --finetune-lr 5e-4
  ```
- For finer-grained data sizing:
  - Use `--base-num-train-samples / --base-num-test-samples` to override base training sizes.
  - Use `--finetune-train-count / --finetune-eval-count` to set 32-GPU finetune/validation sizes (must provide both; overrides `--finetune-train-ratio`).
- Outputs:
  - `VariableLengthStudy/variable_length_metrics.csv`: cumulative MSE/MAE/RMSE/MAPE/R² for each base training and 32-GPU eval.
  - `pred_vs_actual_H100_xxGPU_seed*.csv`: prediction vs. ground-truth comparison on 32-GPU validation sets for error analysis.

