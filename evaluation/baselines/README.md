# Baseline Suite

This directory contains the public baseline reproduction source.

## Files

- `run_suite.py`: orchestrates baseline comparisons from regenerated single-contention data.
- `train_linear_bw.py`: trains the `LinearBW` baseline checkpoint.
- `report_builder.py`: builds tabular baseline summaries from regenerated suite outputs.
- `common.py`: shared schema and IO helpers.
- `heuristics.py`: compatibility wrappers around `algorithms.network_baselines`.
- `linear_model.py`: compatibility wrapper around `algorithms.linear_bw`.
- `test_*.py`: unit tests for baseline behavior and report generation.

## Generated Outputs

Baseline models, CSVs, figures, and reports are written under ignored paths such as `evaluation/baselines/artifacts/`, `evaluation/baselines/figures/`, and `evaluation/baselines/reports/`.

## Public Cleanup Note

Precomputed baseline artifacts and ad hoc report builders were removed. The retained files are executable source plus tests.
