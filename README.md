## BandPilot Refactored

This directory is a modular refactor of the 3,885-line monolithic script `BandPilot/Auto_experiment_H100.py`, covering the full pipeline of GPU bandwidth modeling, search, and evaluation.

It is intended as the official implementation for our TPDS paper on GPU bandwidth-aware dispatching and BandPilot.  
Please adapt the paper information and BibTeX below to your final publication details.

## Paper

This codebase accompanies the paper:

> **[BandPilot: Towards Performance- and Contention-Aware GPU Dispatching in AI Clusters]**  
> Kunming Zhang, Hanlong Liao, Junyu Xue, Deke Guo, and Guoming Tang
> [Paper link](https://arxiv.org/abs/2506.15595) 

### Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{zhang2026bandpilot,
  title   = {BandPilot: Towards Performance- and Contention-Aware GPU Dispatching in AI Clusters},
  author  = {Zhang, Kunming and Liao, Hanlong and Xue, Junyu and Guo, Deke and Tang, Guoming},
  journal = {arXiv preprint arXiv:2506.15595},
  year    = {2026},
  url     = {https://arxiv.org/abs/2506.15595}
}
```

## Environment

- **Python**: recommended 3.8+ (see `requirements.txt` for details)
- **Deep learning framework**: PyTorch (version as specified in `requirements.txt`)
- **OS**: Linux (the experiments in the paper were conducted on Linux-based clusters)
- **Hardware**:
  - H100 GPU cluster with multiple 8-GPU nodes for H100 experiments
  - Heterogeneous 4-node Het-4Mix cluster (4090 / A800 / A6000 / V100, each with 8 GPUs) for heterogeneous experiments

We strongly recommend creating a fresh virtual environment and installing dependencies via:

```bash
pip install -r requirements.txt
```

## Directory Structure

```text
BandPilot/
├── main.py                  # Entry point to read config and drive training/evaluation
├── config/                  # YAML configs
├── core/                    # Bandwidth lookup, topology, GPU configs, and cluster state
├── models/                  # Neural network models (BandwidthPredictor)
├── data_process/            # Data preprocessing, sample generation, and DataLoader
├── algorithms/              # Search and heuristic algorithms
├── training/                # Training and offline evaluation wrappers
├── evaluation/              # Upper-bound estimation and single-contention experiments
├── utils/                   # IO, path, and other helpers
├── Data/                    # Raw/processed data, bandwidth dictionaries, evaluation results
└── model/                   # Model weights and scaler directories per cluster type
```

## Installation

```bash
cd BandPilot    # or the repo root
pip install -r requirements.txt
```

## Quick Start

To run a complete training + evaluation pipeline with the default configuration:

```bash
cd BandPilot
python main.py --config config/default_config.yaml
```

The default config in `config/default_config.yaml` sets data paths, model structure, training hyperparameters, and cluster/bandwidth parameters. Adjust as needed.

## Main Modules

- `core.bandwidth`: Bandwidth lookup cache, switch bandwidth configuration, and model input construction.
- `core.topology`: Topology matrix parsing, composite matrix stitching, and node mapping.
- `core.cluster_state`: `ClusterStateManager` and bandwidth modeling for contention modes (`intensive/common/idle`).
- `models.bandwidth_predictor`: Transformer-based main model (with EWC support).
- `data_process.preprocessing/dataset/dataloader`: Data preprocessing, sample generation, normalization, and grouped `DataLoader`.
- `algorithms.*`: BandPilot search (`improved_searching_algo`), tree search (`tree_search_only`), EHA, Slurm BestFit, baseline/random strategies.
- `training.trainer` / `training.evaluator`: Unified training loops, evaluation, and inference tools.
- `evaluation.metrics` / `evaluation.compare`: Bandwidth upper-bound estimation, single-contention experiments, and offline `max_bw` cache.

## Training Config

The `training` section of `config/default_config.yaml` contains:

- `enable_training`: Whether to run model training (default `true`).
  - `true`: Run training and save the model to  
    `{model_save_dir}/{cluster_type}/bandwidth_predictor_ns{num_train_samples}.pth`,  
    and write `active_num_train_samples.txt` to auto-select the matching scaler during inference/evaluation.
  - `false`: Skip training and directly load an existing model (from the same-suffix default path).  
    - If the file is missing, a `FileNotFoundError` will be raised.  
    - Suitable when a trained model already exists and only evaluation is needed.

Other key training parameters:

- `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`, `patience`, `lambda_ewc`, etc.
- `num_train_samples`: Number of training samples.
- `num_test_samples`: Number of random test samples for quick post-training checks.

## Evaluation and Experiment Config

The current version evaluates algorithms via **offline `max_bw` upper-bound estimation + single-contention experiments**, configured in the `evaluation` section:

- `max_bw_offline`: Offline collection of theoretical bandwidth upper bound `max_bw` under given contention modes. Results are saved to  
  `Data/Evaluation/{cluster_type}/MaxBW_*.csv`, and single-contention experiments always consume this cache.
- `enable_single_contention`: Whether to run single-dispatch + background-contention experiments (`single_dispatch_with_contention`):
  - `single_contention.repeat_num`: Repetitions per `test_num`.
  - `single_contention.if_dynamic`: Whether to sample available GPUs dynamically.
  - `single_contention.contention_mode`: Contention mode list (e.g., `['intensive', 'common', 'idle']`).
  - `single_contention.search_if_real_data`: Whether search uses real data directly.

In `max_bw_offline` you can further control search accuracy and budget:

- `local_top_k`: Local candidates retained per node.
- `max_combos_per_distribution`: Candidate cap per node distribution.
- `max_total_combos`: Global cap on evaluated combinations to keep runtime bounded.

The main program first collects upper bounds via `max_bw_offline`, then (if `enable_single_contention = true`) reads the cache and runs single-contention experiments, outputting  
`Data/Evaluation/{cluster_type}/Single_contention_*.csv`.

> The current repo only provides the `model.type = 'full'` (`BandwidthPredictor`) training and evaluation path.

### Sequential execution for multiple configs (single_contention / max_bw_offline)

- `evaluation.single_contention` and `evaluation.max_bw_offline` allow `repeat_num` and `contention_mode` to be lists; they will be enumerated in order.
- `max_bw_offline` generates a separate cache file for each `(contention_mode, repeat_num)`, and `single_contention` automatically matches the cache and writes result files.
- Example:

  ```yaml
  evaluation:
    enable_single_contention: true
    single_contention:
      repeat_num: [30, 50]
      contention_mode: ['common', 'intensive']
      if_dynamic: true
      search_if_real_data: false
    max_bw_offline:
      enable: true
      repeat_num: [30, 50]
      contention_mode: ['common', 'intensive']
      if_dynamic: true
      search_if_real_data: true
      local_top_k: 10
      max_combos_per_distribution: 2048
      max_total_combos: 200000
  ```

- `contention_mode` is case/whitespace-insensitive and will be normalized to lowercase for cache and result filenames (e.g., YAML `Intensive` is treated as `intensive`).

## Reproducing the Paper Results

This repository is designed so that the main experiments in the paper can be reproduced via configuration files under `config/`.
At a high level, the workflow is:

1. **Prepare data and models** (see the section “Data and Models” below).
2. **Select the target cluster type and experiment settings** in a YAML config under `config/` (e.g., cluster topology, contention modes, training hyperparameters).
3. **Run the pipeline**:

   ```bash
   python main.py --config config/default_config.yaml
   ```

4. **Collect evaluation outputs** from:
   - `Data/Evaluation/{cluster_type}/MaxBW_*.csv` for offline `max_bw` upper-bound estimation.
   - `Data/Evaluation/{cluster_type}/Single_contention_*.csv` for single-dispatch + background-contention experiments.

Depending on your exact config, these CSVs correspond to the tables and figures reported in the paper (e.g., main comparison on H100, ablation on search budget, Het-4Mix experiments).
You can create additional YAML configs (e.g., variants of `default_config.yaml`) to mirror specific experimental settings described in the paper.

## Data and Models

To stay compatible with the original script, the following directories are required:

- `Data/`: Bandwidth CSVs, topology files, bandwidth dictionaries, etc.
- `model/`: Model weights and scaler artifacts per cluster type.

Typical usage patterns:

- For **training from scratch**:
  - Populate `Data/` with the bandwidth and topology files for your cluster.
  - Set `training.enable_training: true` in the config.
  - After training, the model and scalers are saved under `model/{cluster_type}/`.

- For **evaluation only** (using a pretrained model):
  - Ensure `model/{cluster_type}/bandwidth_predictor_ns{num_train_samples}.pth` and corresponding scaler artifacts exist.
  - Set `training.enable_training: false` in the config to skip training and load existing weights.

If you plan to release public datasets or pretrained weights, we recommend adding download links here (e.g., institutional server, cloud storage) and instructions on where to place them inside `Data/` and `model/`.

## Het-4Mix Cluster

- `Het-4Mix` stitches four 8-GPU servers (4090 / A800 / A6000 / V100) into a 32-GPU heterogeneous cluster. It can be configured alongside H100 in `config/default_config.yaml` under `cluster.cluster_types`, e.g.:

  ```yaml
  cluster:
    total_gpu: 32
    cluster_types:
      - 'H100_26H100_27H100_28H100_29'
      - 'Het-4Mix'
  ```

- The four GPU types only provide intra-node (8-GPU) bandwidth dictionaries; cross-node bandwidth reuses the default H100 CSV.
  Final communication bandwidth takes the bottleneck between H100 cross-node results and the minimum intra-node bandwidth, preventing overestimation for heterogeneous nodes.
- All Het-4Mix evaluation results are written to `Data/Evaluation/Het-4Mix/`, and model weights are saved to `model/Het-4Mix/`.

## Next Steps / Roadmap

- Extend evaluation statistics/visualization in `evaluation` as needed.
- Add unit tests to cover core modules.
- Migrate more experiment entry points from the original script into `main.py`.

## License

This project is currently intended for academic research use.  
We recommend adding an explicit open-source license file (e.g., MIT, Apache-2.0, or BSD-3-Clause) at the repository root and updating this section accordingly.

## Contact

For questions, feedback, or collaboration, please contact the corresponding author(s):  
`[your.name]@institution.edu`  <!-- TODO: replace with actual contact email -->

