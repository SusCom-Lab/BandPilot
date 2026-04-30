# Configuration Guide

Runtime configuration is YAML-based. The default public entry point is:

```bash
python main.py --config config/default_config.yaml
```

`config/default_config.yaml` keeps only source-reproducible paths. It reads lightweight inputs from `Data/` and writes regenerated outputs to ignored directories such as `model/`, `Data/Evaluation/`, and `evaluation/**/artifacts/`.

## Main Sections

- `data`: input data paths, checkpoint output root, and evaluation output root.
- `model`: Transformer predictor architecture.
- `training`: training switches, optimizer settings, sample counts, and patience.
- `cluster`: total GPU count, cluster templates, and switch-bandwidth filename tag.
- `evaluation.single_contention`: BandPilot/PTS/baseline comparison under background contention.
- `evaluation.max_bw_offline`: offline `MaxBW_*` cache generation consumed by single-contention evaluation.
- `evaluation.sensitivity_analysis`: predictor-level sample-sensitivity driver settings.
- `evaluation.scalability_benchmark`: measured/simulated/synthesized scalability-latency benchmark settings.
- `random_seed` and `device`: global reproducibility and execution-device controls.

## Reproduction Notes

The public default enables `max_bw_offline` and `single_contention` so a clean checkout can regenerate the missing `Data/Evaluation/` cache before running the comparison stage. For faster smoke checks, reduce `repeat_num`, `num_train_samples`, and `num_test_samples` before running.

Generated outputs are intentionally ignored by Git. Keep the source configs, scripts, and lightweight input data under version control; do not commit regenerated CSVs, checkpoints, cache files, or plots.
