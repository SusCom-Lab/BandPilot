# Sensitivity Analysis

This directory keeps source for BandPilot sample-sensitivity studies.

## Workflows

- Predictor-level sensitivity: `training/sample_sensitivity_experiment.py`.
- Dispatch-level sidecar: `training/sample_sensitivity_dispatch_experiment.py`.
- Nested acceptance check: `check_nested_acceptance.py`.
- Hierarchical versus naive predictor ablation: `run_hier_vs_naive_ablation.py`.
- Batch rerun wrapper: `run_nested_cumulative_rerun.sh`.

## Generated Outputs

All CSVs, JSON summaries, plots, reports, nested manifests, and temporary models are generated under ignored `evaluation/sensitivity-analysis/artifacts/` paths.

## Public Cleanup Note

Precomputed sensitivity artifacts and reviewer-report drafts were removed. The retained files are source for regenerating the studies.
