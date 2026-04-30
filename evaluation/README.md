# Evaluation Module Guide

This directory contains source code for reproducing the BandPilot evaluation workflows. Generated CSVs, plots, reports, checkpoints, and caches are intentionally excluded from the public tree.

## Main Entry Points

- `compare.py`: single-contention comparison utilities and offline `MaxBW_*` cache generation.
- `metrics.py`: maximum-bandwidth helper functions used by the evaluation pipeline.
- `baselines/`: reviewer-facing baseline suite source.
- `scalability/`: scalability-latency benchmark source with measured, simulated, and synthesized stages.
- `sensitivity-analysis/`: predictor-level and dispatch-level sample-sensitivity source.
- `llm_tp_bandwidth/`: optional measured two-GPU tensor-parallel LLM sidecar.

## Generated Outputs

The following paths are generated locally and ignored by Git:

- `Data/Evaluation/`
- `evaluation/**/artifacts/`
- `evaluation/**/reports/`
- `evaluation/**/figures/`
- `evaluation/**/cache/`

Regenerate these outputs from the scripts and configs when reproducing the paper. Do not commit regenerated artifacts to the public source repository.

## Evidence Boundary

Keep evidence labels explicit:

- `measured`: directly collected on the target hardware or sidecar environment.
- `simulated`: replayed or scaled from measured/source traces.
- `synthesized`: bounded control-plane or analytical construction.
- `theoretical`: derived from formulas or model assumptions only.

## Public Cleanup Note

Internal reports, stale adaptive-diagnostic wrappers, and precomputed results were removed. This directory now keeps source needed to regenerate the evaluation evidence.
