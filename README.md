# BandPilot

This repository contains the public reproduction source for **BandPilot: Towards Performance- and Contention-Aware GPU Dispatching in AI Clusters**.

The public tree is intentionally source-only. Generated experiment results, trained checkpoints, caches, internal revision logs, paper PDFs, and private workspace automation have been removed. The retained files are sufficient to inspect the implementation and rerun the reported workflows with the provided input data and configuration templates.

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

The default config reads source inputs from `Data/H100`, `Data/Bandwidth`, and `Data/Topology`. It writes regenerated checkpoints and evaluation results to ignored output directories, so a clean public checkout remains source-only after results are removed.

## Main Reproduction Workflows

Use `config/default_config.yaml` for the main H100 and Het-4Mix training/evaluation pipeline. It controls data paths, model hyperparameters, training sample budgets, cluster topology, contention modes, offline `MaxBW_*` cache generation, and single-contention evaluation.

In the single-contention workflow, the public `BandPilot` row is the current
`search.py` mainline path: EHA candidate pruning, runtime kNN PTS gating, and
current PTS refinement. The old exact-PTS BandPilot path is labeled
`legacy-BandPilot`; the current PTS primitive is labeled `PTS`; the old exact
PTS primitive is labeled `legacy-PTS`; and the legacy tree-search primitive is
labeled `tree`.

Use `evaluation/baselines/` to train and compare reviewer-facing baselines such as `CasCore`, `BWGreedy`, and `LinearBW`.

Use `evaluation/scalability/` to rebuild the scalability-latency pipeline. The evidence boundaries are explicit: 32-GPU dispatch latency is `measured`, 64-1024 GPU scaled traces are `simulated`, and 2048-4096 GPU control-plane bounds are `synthesized`.

Use `evaluation/sensitivity-analysis/` and `training/sample_sensitivity_*` for predictor-level and dispatch-level sample-sensitivity studies.

Use `evaluation/llm_tp_bandwidth/` as an optional `measured` sidecar for two-GPU tensor-parallel LLM bandwidth sensitivity. This sidecar requires local model paths and a CUDA/NCCL environment.

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

## Contact

For questions about the artifact, contact the corresponding author at `kzhang519@connect.hkust-gz.edu.cn`.
