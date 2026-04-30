"""Acceptance-gate checker for nested/cumulative sensitivity reruns.

The checker reads a dispatch-sidecar `summary.csv` and verifies whether the
first-pass nested result is stable enough or should be rerun with a larger
`10 seeds x 10 repeats` protocol.

Rules:
1. `Het-4Mix + Stratified`: `util_500 >= util_250`.
2. `Het-4Mix + Random`: `util_500 > util_250`.
3. `H100 + Random/Stratified`: `abs(util_500 - util_250) <= 1.0 pt`.

Example:
```bash
conda run -n gpu_dp_opt python evaluation/sensitivity-analysis/check_nested_acceptance.py \
  --summary evaluation/sensitivity-analysis/artifacts/dispatch_sidecar/<run-tag>/summary.csv
```
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


H100_CLUSTER = "H100_26H100_27H100_28H100_29"
HET_CLUSTER = "Het-4Mix"


def _lookup_util(summary: pd.DataFrame, cluster_type: str, strategy: str, sample_size: int) -> float:
    """Fetch mean final utilization for one `(cluster, strategy, sample_size)` tuple."""

    subset = summary[
        (summary["cluster_type"] == cluster_type)
        & (summary["strategy"] == strategy)
        & (summary["sample_size"] == sample_size)
    ]
    if subset.empty:
        raise ValueError(
            f"Missing row for cluster={cluster_type}, strategy={strategy}, sample_size={sample_size}"
        )
    return float(subset.iloc[0]["mean_final_utilization"])


def evaluate_acceptance(summary: pd.DataFrame) -> list[str]:
    """Return a list of violated acceptance constraints."""

    failures: list[str] = []

    het_strat_250 = _lookup_util(summary, HET_CLUSTER, "Stratified", 250)
    het_strat_500 = _lookup_util(summary, HET_CLUSTER, "Stratified", 500)
    if het_strat_500 < het_strat_250:
        failures.append(
            f"Het-4Mix + Stratified violated: util_500={het_strat_500:.2f} < util_250={het_strat_250:.2f}"
        )

    het_rand_250 = _lookup_util(summary, HET_CLUSTER, "Random", 250)
    het_rand_500 = _lookup_util(summary, HET_CLUSTER, "Random", 500)
    if het_rand_500 <= het_rand_250:
        failures.append(
            f"Het-4Mix + Random violated: util_500={het_rand_500:.2f} <= util_250={het_rand_250:.2f}"
        )

    for strategy in ["Random", "Stratified"]:
        h100_250 = _lookup_util(summary, H100_CLUSTER, strategy, 250)
        h100_500 = _lookup_util(summary, H100_CLUSTER, strategy, 500)
        delta = h100_500 - h100_250
        if math.fabs(delta) > 1.0:
            failures.append(
                f"H100 + {strategy} violated: |util_500 - util_250| = {abs(delta):.2f} > 1.0"
            )

    return failures


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the acceptance checker."""

    parser = argparse.ArgumentParser(description="Check nested sensitivity acceptance gates")
    parser.add_argument("--summary", type=Path, required=True, help="Path to dispatch summary.csv")
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    summary = pd.read_csv(args.summary)
    failures = evaluate_acceptance(summary)
    if failures:
        print("Acceptance gate failed:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("Acceptance gate passed.")


if __name__ == "__main__":
    main()
