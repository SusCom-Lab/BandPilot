# Data Processing Module Guide

This directory contains the data-loading and preprocessing code used by BandPilot.

## Files

- `preprocessing.py`: parses bandwidth CSV rows and builds lookup dictionaries keyed by GPU activity patterns.
- `dataset.py`: generates balanced, random, and simplified training samples from the retained source inputs.
- `dataloader.py`: creates PyTorch data loaders and serializes scalers beside regenerated checkpoints.

## Source Data Boundary

The public artifact keeps lightweight source inputs in:

- `Data/H100/`: H100 CSV and topology inputs.
- `Data/Bandwidth/`: per-node bandwidth dictionaries.
- `Data/Topology/`: topology text files for H100 and heterogeneous nodes.

Generated evaluation data and model/scaler artifacts are not stored in Git. They are written to ignored paths such as `Data/Evaluation/` and `model/` when the training or evaluation workflows run.

## Public Cleanup Note

Historical derived datasets and ad hoc generation notes were removed from this README because those outputs are not part of the source-only public artifact.
