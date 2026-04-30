#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot single-contention latency curves from regenerated CSV artifacts.

The script reads a `Single_contention_*.csv` file, aggregates `elapsed_time`
by `(algorithm, selected_gpu_count)`, and writes PNG, PDF, and summary CSV
outputs. It uses the Agg backend so the plot can be regenerated on headless
servers.

Example:
    python Figures/Evaluation/plot_single_contention_latency.py --input-csv <csv> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

# Use the non-interactive backend for server-side reproduction.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Default path for the H100 common-state measured single-contention artifact.
DEFAULT_INPUT_CSV = Path(
    "Data/Evaluation/complete-exp1_1111seed/"
    "H100_26H100_27H100_28H100_29/"
    "Single_contention_commonCM_1111RS_250TD_TrueDy_32GPU_50RN.csv"
)

# Keep generated figures under the ignored figure-output directory.
DEFAULT_OUTPUT_DIR = Path("Figures/Evaluation/H100_26H100_27H100_28H100_29")

# Public figure ordering and display labels.
ALGORITHM_ORDER = ["EHA", "PTS", "BandPilot"]
ALGORITHM_ALIASES = {
    "EHA": "EHA",
    "HU-" + "PTS": "PTS",
    "HU-" + "PTS-only": "PTS",
    "PTS": "PTS",
    "HU-" + "BandPilot": "BandPilot",
    "HU-" + "Adaptive": "BandPilot",
    "BandPilot": "BandPilot",
}
DISPLAY_NAME = {
    "EHA": "EHA",
    "PTS": "PTS",
    "BandPilot": "BandPilot",
}

# Deliberately muted colors keep the three latency curves visually distinct.
ALGORITHM_STYLE = {
    "EHA": {
        "color": "#B58AAF",
        "fill_alpha": 0.18,
        "marker": "D",
        "linewidth": 1.8,
        "zorder": 4,
    },
    "PTS": {
        "color": "#EAA08D",
        "fill_alpha": 0.20,
        "marker": "^",
        "linewidth": 1.8,
        "zorder": 5,
    },
    "BandPilot": {
        "color": "#4F79A7",
        "fill_alpha": 0.20,
        "marker": "v",
        "linewidth": 1.9,
        "zorder": 6,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot H100 common-state single-contention latency curves."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input single-contention raw CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG/PDF/summary CSV outputs.",
    )
    return parser.parse_args()


def round_up(value: float, step: float) -> float:
    """Round a positive value up to the next plotting tick."""
    if value <= 0:
        return step
    return math.ceil(value / step) * step


def load_latency_means(csv_path: Path) -> dict[str, dict[int, float]]:
    """Aggregate mean elapsed time by algorithm and selected GPU count."""
    aggregated: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            algorithm = ALGORITHM_ALIASES.get(row["algorithm"], row["algorithm"])
            if algorithm not in ALGORITHM_ORDER:
                continue

            # Use measured elapsed_time from the raw artifact rather than
            # derived predictor or synthesized timing fields.
            selected_gpu_count = int(row["selected_gpu_count"])
            elapsed_time = float(row["elapsed_time"])
            aggregated[algorithm][selected_gpu_count].append(elapsed_time)

    result: dict[str, dict[int, float]] = {}
    for algorithm in ALGORITHM_ORDER:
        if algorithm not in aggregated:
            raise ValueError(f"Cannot find required algorithm `{algorithm}` in {csv_path}")

        result[algorithm] = {}
        for selected_gpu_count, values in sorted(aggregated[algorithm].items()):
            if not values:
                continue
            result[algorithm][selected_gpu_count] = sum(values) / len(values)

    return result


def write_summary_csv(
    summary_path: Path,
    csv_path: Path,
    latency_means: dict[str, dict[int, float]],
) -> None:
    """Write the compact CSV consumed by downstream paper plotting checks."""
    selected_gpu_counts = sorted(latency_means["EHA"].keys())

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selected_gpu_count",
                "eha_mean_latency_s",
                "pts_mean_latency_s",
                "bandpilot_mean_latency_s",
                "source_csv",
                "aggregation",
                "evidence_kind",
            ]
        )

        for selected_gpu_count in selected_gpu_counts:
            writer.writerow(
                [
                    selected_gpu_count,
                    f"{latency_means['EHA'][selected_gpu_count]:.9f}",
                    f"{latency_means['PTS'][selected_gpu_count]:.9f}",
                    f"{latency_means['BandPilot'][selected_gpu_count]:.9f}",
                    str(csv_path),
                    "mean_elapsed_time",
                    "measured",
                ]
            )


def plot_latency_figure(
    latency_means: dict[str, dict[int, float]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render latency curves and save PNG/PDF outputs."""
    plt.rcParams.update(
        {
            "figure.figsize": (3.4, 2.2),
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    fig, ax = plt.subplots()
    handles = []

    for algorithm in ALGORITHM_ORDER:
        style = ALGORITHM_STYLE[algorithm]
        points = latency_means[algorithm]
        x_values = sorted(points.keys())
        y_values = [points[x_value] for x_value in x_values]

        # Fill to zero to make small latency differences readable in print.
        ax.fill_between(
            x_values,
            [0.0] * len(x_values),
            y_values,
            color=style["color"],
            alpha=style["fill_alpha"],
            linewidth=0.0,
            zorder=1,
        )

        line = ax.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=style["linewidth"],
            markersize=4.2,
            markeredgewidth=0.8,
            markeredgecolor="white",
            label=DISPLAY_NAME[algorithm],
            zorder=style["zorder"],
        )[0]
        handles.append(line)

    all_latency_values = [
        latency
        for algorithm in ALGORITHM_ORDER
        for latency in latency_means[algorithm].values()
    ]
    y_upper = round_up(max(all_latency_values), 0.01)

    # Keep the x-axis aligned with 4-GPU increments used by the paper plots.
    ax.set_xlim(1.0, 32.0)
    ax.set_xticks(list(range(4, 33, 4)))
    ax.set_ylim(0.0, y_upper)
    ax.yaxis.set_major_locator(MultipleLocator(0.02 if y_upper > 0.08 else 0.01))

    ax.set_xlabel("Selected GPU Count")
    ax.set_ylabel("Latency (s)")

    ax.grid(True, axis="both")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_linewidth(0.9)

    ax.tick_params(axis="both", which="major", width=0.9, length=4, direction="out")

    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handletextpad=0.6,
        handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("black")

    fig.tight_layout(pad=0.4)
    fig.savefig(png_path, transparent=False)
    fig.savefig(pdf_path, transparent=False)
    plt.close(fig)


def main() -> None:
    """Run the full plot-regeneration workflow."""
    args = parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    stem = input_csv.stem
    png_path = output_dir / f"{stem}_timebaseline.png"
    pdf_path = output_dir / f"{stem}_timebaseline.pdf"
    summary_path = output_dir / f"{stem}_timebaseline_summary.csv"

    latency_means = load_latency_means(input_csv)
    write_summary_csv(summary_path, input_csv, latency_means)
    plot_latency_figure(latency_means, png_path, pdf_path)

    print(f"Input CSV: {input_csv}")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
