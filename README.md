# BandPilot

This repository contains the public reproduction source for **BandPilot: Towards Performance- and Contention-Aware GPU Dispatching in AI Clusters**.

## Repository Layout

```text
.
├── main.py                  # Main training and evaluation entry point
├── config/                  # Runtime configuration templates
├── core/                    # Bandwidth lookup, topology, GPU config, cluster state
├── models/                  # Bandwidth predictor model definitions
├── data_process/            # Dataset generation and preprocessing helpers
├── algorithms/              # BandPilot, PTS, EHA, Slurm-style, and baseline search logic
├── training/                # Training, evaluation, and sensitivity experiment drivers
├── evaluation/              # Reproduction scripts for comparisons, baselines, scalability, and sidecars
├── Figures/Evaluation/      # Plotting source for regenerated evaluation figures
├── Data/                    # Lightweight source inputs: bandwidth dictionaries, H100 CSVs, topology files
└── requirements.txt         # Python dependency list
```

The following paths are generated locally and are excluded from the public source tree:

- `Data/Evaluation/` and `Data/legacy_Evaluation/` for evaluation CSV outputs.
- `model/` or `Models/` for checkpoints and scalers.
- `evaluation/**/artifacts/`, `evaluation/**/reports/`, `evaluation/**/figures/`, and `evaluation/**/cache/`.
- `Figures/search_overhead*/` and generated figure files under `Figures/Evaluation/`.

## Environment

Use Python 3.8 or newer on Linux. The implementation depends on PyTorch, NumPy, pandas, scikit-learn, matplotlib, and PyYAML; install the pinned project dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU-dependent workflows require an environment compatible with the target cluster and PyTorch build. CPU-only syntax checks and report generation can be run without GPU access.

## Quick Start

Run the default training and single-contention evaluation workflow:

```bash
python main.py --config config/default_config.yaml
```

The default workflow reads the public inputs under `Data/` and writes regenerated
checkpoints, CSVs, and figures to ignored output directories.

## Reproduction Workflows

The main entry point is `main.py`, which trains the bandwidth predictor and runs
the single-contention dispatch evaluation used by the paper workflow.

The reported algorithms are implemented under `algorithms/`. In regenerated
single-contention tables, `BandPilot` refers to the current mainline path in
`algorithms/search.py`; legacy variants are kept only for comparison labels.

Generated artifacts are intentionally not part of the public source tree. After
rerunning workflows, expect outputs under ignored paths such as `model/`,
`Data/Evaluation/`, and `evaluation/**/artifacts/`.

See the README files inside each subdirectory for command-level details when
rerunning a specific study.

## Citation

```bibtex
@article{zhang2026bandpilot,
  title   = {BandPilot: Towards Performance- and Contention-Aware GPU Dispatching in AI Clusters},
  author  = {Zhang, Kunming and Liao, Hanlong and Xue, Junyu and Guo, Deke and Tang, Guoming},
  journal = {arXiv preprint arXiv:2506.15595},
  year    = {2026},
  url     = {https://arxiv.org/abs/2506.15595}
}
```
