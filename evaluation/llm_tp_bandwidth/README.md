# LLM Tensor-Parallel Bandwidth Sidecar

This optional sidecar measures how two-GPU tensor-parallel LLM training changes across GPU-pair bandwidth classes.

## Files

- `runner.py`: validates the environment, selects GPU pairs, and launches workers.
- `worker.py`: executes the two-rank training step workload.
- `analyze.py`: converts raw JSONL records into CSV and JSON summaries.
- `plot.py`: regenerates English-labeled figures from summaries.
- `report_builder.py`: writes a Markdown sidecar report.
- `dataset.py`, `gpu_monitor.py`, and `io_utils.py`: shared support utilities.
- `configs/default.yaml`: editable local configuration template.

## Requirements

This workflow requires CUDA, NCCL, PyTorch distributed execution, local model weights, and a compatible tokenizer/dataset cache. The default config uses local paths that must be edited for each deployment.

## Evidence Boundary

Successful sidecar runs are `measured` for the local machine and GPU pair only. They should not be described as H100-cluster measurements unless collected on that target hardware.

## Public Cleanup Note

Generated run outputs, caches, figures, and reports were removed. The public tree keeps only sidecar source and configuration templates.
