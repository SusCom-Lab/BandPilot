# Scalability Benchmark

This directory contains the source for the BandPilot scalability-latency benchmark.

## Files

- `benchmark.py`: runs the staged scalability benchmark.
- `report_builder.py`: audits regenerated benchmark outputs and builds public summary material.
- `rebuild_artifacts.py`: rebuilds reviewer-facing tables and figures from regenerated raw CSVs.
- `online_policy.py`: benchmark-local adaptive-policy helper.
- `configs/`: YAML configs for full runs, refresh runs, sidecars, and smoke checks.
- `pts_sidecar/`: focused PTS versus legacy-PTS sidecar.
- `test_*.py`: unit tests for report and sidecar pipelines.

## Evidence Boundary

- 32-GPU dispatch latency is `measured`.
- 64-1024 GPU scaled-trace latency is `simulated`.
- 2048-4096 GPU control-plane bounds are `synthesized`.

## Generated Outputs

All benchmark artifacts are regenerated under ignored directories such as `evaluation/scalability/artifacts/`, `evaluation/scalability/reports/`, and `Figures/search_overhead*/`.

## Public Cleanup Note

Historical benchmark outputs were removed. The public tree keeps configs, scripts, and tests only.
