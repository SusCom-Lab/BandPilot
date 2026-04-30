"""Standalone sensitivity figure generator - regenerate figures from saved CSV.

Usage:
    python -m training.plot_sensitivity --csv evaluation/sensitivity-analysis/artifacts/predictor-level/sensitivity_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.sample_sensitivity_experiment import plot_sensitivity_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate sensitivity figures from CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Path to sensitivity_results.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same as CSV)")
    parser.add_argument("--total-gpu", type=int, default=32, help="Total GPU count for sparsity annotation")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    output_dir = args.output_dir or args.csv.parent
    plot_sensitivity_figures(df, output_dir, args.total_gpu)
    print("Done.")


if __name__ == "__main__":
    main()
