# Scalability PTS Sidecar

This sidecar isolates the scalability impact of PTS versus legacy exact PTS on representative large-cluster slices.

## Files

- `runner.py`: executes the sidecar cases and writes raw rows.
- `analyze.py`: computes paired summaries and speedups.
- `plot.py`: regenerates English-labeled figures.
- `report_builder.py`: writes a concise Markdown report from regenerated outputs.

## Evidence Boundary

This sidecar is `simulated` evidence because it scales the cluster template rather than measuring a physical thousand-GPU deployment.

## Public Cleanup Note

Generated sidecar artifacts were removed. Run the scripts to recreate `artifacts/` locally when needed.
