"""Generate best-found GBE summaries from regenerated comparison CSVs.

The script computes `GBE = final_bw / case_max_final_bw * 100` from local
`Single_contention_*.csv` artifacts and writes JSON, Markdown, and LaTeX summary
outputs under ignored evaluation artifact directories.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Protocol constants used to locate regenerated comparison artifacts.
TRAINING_DATA_NUM = 250
RANDOM_SEED = 777
REPEAT_NUM = 50
TOTAL_GPU = 32
FILE_ROOT = Path("Data/Evaluation")
OUTPUT_DIR = Path("evaluation/artifacts/baseline_best_found_gbe_777RS")

# Public cluster, mode, and algorithm order used by the generated table.
CLUSTERS = (
    "H100_26H100_27H100_28H100_29",
    "Het-4Mix",
)
MODES = (
    ("intensive", r"$\tau_{hvy}$"),
    ("common", r"$\tau_{mod}$"),
    ("idle", r"$\tau_{idle}$"),
)
TABLE_ALGORITHMS = (
    ("BandPilot", r"\sysname{}"),
    ("LinearBW", "LinProxy"),
    ("BWGreedy", "PairBW"),
    ("CasCore", "CasCore"),
    ("Topo", "Topo"),
    ("Default", "Default"),
)

ALGORITHM_ALIASES = {
    "HU-" + "BandPilot": "BandPilot",
    "HU-" + "Adaptive": "BandPilot",
    "HU-" + "PTS": "PTS",
    "HU-" + "PTS-only": "PTS",
    "PTS-only": "legacy-PTS",
}


def _csv_path(cluster_type: str, mode: str) -> Path:
    """Resolve the input comparison CSV for one cluster and contention mode."""

    return (
        FILE_ROOT
        / cluster_type
        / (
            f"Single_contention_{mode}CM_{RANDOM_SEED}RS_"
            f"{TRAINING_DATA_NUM}TD_TrueDy_{TOTAL_GPU}GPU_{REPEAT_NUM}RN.csv"
        )
    )


def _load_rows(path: Path) -> List[dict]:
    """Load a comparison CSV into a list of row dictionaries."""

    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_algorithm_name(name: str) -> str:
    """Normalize legacy artifact labels to the public naming scheme."""

    return ALGORITHM_ALIASES.get(str(name).strip(), str(name).strip())


def _case_key(row: dict) -> Tuple[str, str]:
    """Return the dispatch-case key `(test_num, repeat_idx)` within one mode."""

    return row["test_num"], row["repeat_idx"]


def _compute_mode_summary(rows: Iterable[dict]) -> Dict[str, object]:
    """Compute best-found GBE summaries for one contention mode."""

    # Find the best observed `final_bw` for each dispatch case.
    case_max_bw: Dict[Tuple[str, str], float] = {}
    for row in rows:
        key = _case_key(row)
        bw = float(row["final_bw"])
        if key not in case_max_bw or bw > case_max_bw[key]:
            case_max_bw[key] = bw

    # Track BandPilot GBE so the upper-bound adjustment uses the same public label.
    bandpilot_gbe: Dict[Tuple[str, str], float] = {}
    raw_gbe_rows: List[Tuple[int, str, Tuple[str, str], float]] = []
    for row in rows:
        key = _case_key(row)
        max_bw = case_max_bw[key]
        gbe = (float(row["final_bw"]) / max_bw * 100.0) if max_bw else 0.0
        algorithm = _normalize_algorithm_name(row["algorithm"])
        if algorithm == "BandPilot":
            bandpilot_gbe[key] = gbe
        raw_gbe_rows.append((int(row["selected_gpu_count"]), algorithm, key, gbe))

    # Keep UpperBandPilot no worse than BandPilot on the same case.
    adjusted_rows: List[Tuple[int, str, float]] = []
    for selected_gpu_count, algorithm, key, gbe in raw_gbe_rows:
        if algorithm == "UpperBandPilot" and key in bandpilot_gbe:
            gbe = max(gbe, bandpilot_gbe[key])
        adjusted_rows.append((selected_gpu_count, algorithm, gbe))

    # Match the notebook aggregation: first by `(selected_gpu_count, algorithm)`.
    per_k_algorithm_values: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for selected_gpu_count, algorithm, gbe in adjusted_rows:
        per_k_algorithm_values[(selected_gpu_count, algorithm)].append(gbe)

    per_k_algorithm_mean: Dict[Tuple[int, str], float] = {}
    for key, values in per_k_algorithm_values.items():
        per_k_algorithm_mean[key] = sum(values) / len(values)

    # Then average over selected GPU counts to obtain mode-level mean GBE.
    algorithm_to_k_means: Dict[str, List[float]] = defaultdict(list)
    for (_, algorithm), mean_value in per_k_algorithm_mean.items():
        algorithm_to_k_means[algorithm].append(mean_value)

    algorithm_mode_mean: Dict[str, float] = {}
    for algorithm, values in algorithm_to_k_means.items():
        algorithm_mode_mean[algorithm] = sum(values) / len(values)

    return {
        "case_count": len(case_max_bw),
        "row_count": len(adjusted_rows),
        "algorithm_mode_mean": algorithm_mode_mean,
    }


def _round1(value: float) -> str:
    """Format a value with one decimal place."""

    return f"{value:.1f}"


def _build_table_tex(summary: Dict[str, object]) -> str:
    """Build the LaTeX table for the best-found GBE summary."""

    cluster_titles = {
        "H100_26H100_27H100_28H100_29": "H100 Cluster",
        "Het-4Mix": "Het-4Mix Cluster",
    }
    mode_keys = [mode for mode, _ in MODES]
    mode_headers = [latex_name for _, latex_name in MODES]

    lines: List[str] = [
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \small",
        r"  \caption{Average Performance Comparison of GPU Dispatching Algorithms.}",
        r"  \label{tab:perf_comparison}",
        r"  \resizebox{\columnwidth}{!}{",
        r"  \begin{tabular}{@{}lccc@{}}",
        r"    \toprule",
        (
            r"    \textbf{Algorithm} & \textbf{Mean GBE with "
            + mode_headers[0]
            + r" (\%)} $\uparrow$ & "
            + r"\textbf{"
            + mode_headers[1]
            + r" (\%)} $\uparrow$ & "
            + r"\textbf{"
            + mode_headers[2]
            + r" (\%)} $\uparrow$  \\"
        ),
        r"    \midrule",
        r"    \addlinespace",
        "",
    ]

    for cluster in CLUSTERS:
        lines.append(
            rf"    \multicolumn{{4}}{{@{{}}l}}{{\textbf{{{cluster_titles[cluster]}}}}} \\"
        )
        lines.append(r"    \midrule")
        for algorithm, display_name in TABLE_ALGORITHMS:
            values = [
                summary[cluster]["modes"][mode_key]["algorithm_mode_mean"][algorithm]
                for mode_key in mode_keys
            ]
            row_label = rf"\textbf{{{display_name}}}" if algorithm == "BandPilot" else display_name
            row_values = " & ".join(_round1(value) for value in values)
            lines.append(f"    {row_label} & {row_values} \\\\")
        lines.append("")
        lines.append(r"    \addlinespace")
        lines.append("")

    # Remove the final spacer before appending the bottom rule.
    lines = lines[:-2]
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_report_md(summary: Dict[str, object]) -> str:
    """Build the Markdown report from the computed summary."""

    lines: List[str] = [
        "# Baseline Best-Found GBE Summary (`777RS`)",
        "",
        "## Fixed Protocol",
        "",
        f"- Data root: `{FILE_ROOT}`",
        f"- Training data num: `{TRAINING_DATA_NUM}`",
        f"- Random seed: `{RANDOM_SEED}`",
        f"- Repeat num: `{REPEAT_NUM}`",
        f"- Total GPU: `{TOTAL_GPU}`",
        "- Evidence kind: `simulated`",
        "- GBE formula: `final_bw / case_max_final_bw * 100`",
        "- Aggregation: `case -> (selected_gpu_count, algorithm) mean -> algorithm mean`",
        "",
        "## Cluster Summary",
        "",
    ]

    for cluster in CLUSTERS:
        lines.append(f"### {cluster}")
        lines.append("")
        for mode_key, mode_name in MODES:
            lines.append(f"- {mode_key} / {mode_name}:")
            mode_values = summary[cluster]["modes"][mode_key]["algorithm_mode_mean"]
            ranking = sorted(
                ((algorithm, value) for algorithm, value in mode_values.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            lines.append(
                "  "
                + ", ".join(f"`{algorithm}={value:.3f}`" for algorithm, value in ranking)
            )
        lines.append("")
        lines.append("- overall stronger-baseline subset:")
        overall_values = summary[cluster]["overall_table_subset"]
        ranking = sorted(overall_values.items(), key=lambda item: item[1], reverse=True)
        lines.append(
            "  " + ", ".join(f"`{algorithm}={value:.3f}`" for algorithm, value in ranking)
        )
        lines.append("")

    lines.append("## LaTeX Table")
    lines.append("")
    lines.append("```tex")
    lines.append(summary["table_tex"].rstrip())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_summary() -> Dict[str, object]:
    """Build the JSON summary and overall table subset."""

    summary: Dict[str, object] = {
        "metadata": {
            "file_root": str(FILE_ROOT),
            "training_data_num": TRAINING_DATA_NUM,
            "random_seed": RANDOM_SEED,
            "repeat_num": REPEAT_NUM,
            "total_gpu": TOTAL_GPU,
            "evidence_kind": "simulated",
            "gbe_formula": "final_bw / case_max_final_bw * 100",
            "aggregation": (
                "mean over rows per (selected_gpu_count, algorithm), "
                "then mean over selected_gpu_count"
            ),
            "upperbandpilot_adjustment": (
                "per-case max between UpperBandPilot GBE and BandPilot GBE"
            ),
        }
    }

    for cluster in CLUSTERS:
        cluster_summary: Dict[str, object] = {
            "input_files": {},
            "modes": {},
        }
        for mode_key, _ in MODES:
            csv_path = _csv_path(cluster, mode_key)
            rows = _load_rows(csv_path)
            cluster_summary["input_files"][mode_key] = str(csv_path)
            cluster_summary["modes"][mode_key] = _compute_mode_summary(rows)

        # Compute overall means across contention modes for each algorithm.
        overall_algorithm_values: Dict[str, List[float]] = defaultdict(list)
        for mode_key, _ in MODES:
            mode_summary = cluster_summary["modes"][mode_key]["algorithm_mode_mean"]
            for algorithm, value in mode_summary.items():
                overall_algorithm_values[algorithm].append(value)

        cluster_summary["overall_all_algorithms"] = {
            algorithm: sum(values) / len(values)
            for algorithm, values in overall_algorithm_values.items()
        }
        cluster_summary["overall_table_subset"] = {
            algorithm: cluster_summary["overall_all_algorithms"][algorithm]
            for algorithm, _ in TABLE_ALGORITHMS
        }
        summary[cluster] = cluster_summary

    # Embed derived Markdown and LaTeX artifacts in the JSON summary.
    summary["table_tex"] = _build_table_tex(summary)
    summary["report_markdown"] = _build_report_md(summary)
    return summary


def write_outputs(summary: Dict[str, object]) -> None:
    """Write JSON, Markdown, and LaTeX output artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep JSON as the machine-readable source for downstream checks.
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Markdown is for inspection; LaTeX is for manuscript/table reuse.
    (OUTPUT_DIR / "report.md").write_text(
        summary["report_markdown"],
        encoding="utf-8",
    )
    (OUTPUT_DIR / "perf_comparison_table.tex").write_text(
        summary["table_tex"],
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point: compute summaries and write artifacts."""

    summary = build_summary()
    write_outputs(summary)
    print(f"Wrote summary to: {OUTPUT_DIR / 'summary.json'}")
    print(f"Wrote markdown report to: {OUTPUT_DIR / 'report.md'}")
    print(f"Wrote LaTeX table to: {OUTPUT_DIR / 'perf_comparison_table.tex'}")


if __name__ == "__main__":
    main()
