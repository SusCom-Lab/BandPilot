# Evaluation Figure Sources

This directory keeps source code for regenerating evaluation figures. Generated CSV, PNG, and PDF files are intentionally ignored and should not be committed.

## Files

- `plot_single_contention_latency.py`: builds reviewer-facing single-contention latency plots from regenerated `Single_contention_*.csv` files.

## Expected Inputs

Run the evaluation pipeline first so `Data/Evaluation/<cluster_type>/Single_contention_*.csv` exists locally. The plot script reads those regenerated CSVs and writes figures under `Figures/Evaluation/<cluster_type>/`.

## Public Cleanup Note

Historical notebook-driven plotting artifacts were removed from the public tree. This directory now keeps only reusable plotting source.
