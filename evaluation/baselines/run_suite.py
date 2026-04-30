"""Run the isolated baseline suite under ``evaluation/baselines``.

Function:
- Execute the reviewer baseline comparison without editing the repository's
  default compare pipeline.
- Reuse the existing single-contention runtime helpers, real-data re-evaluator,
  and search implementations in read-only mode.

Design:
- Existing algorithms (`BandPilot`, `Topo`, `Default`, `Random`) are invoked
  by imported helpers or direct read-only function calls.
- `CasCore`, `BWGreedy`, and `LinearBW` are imported from `algorithms/`; this
  runner only orchestrates the suite and writes regenerated outputs.
- The runtime-adaptive `BandPilot` path explicitly iterates
  `repeat_idx -> test_num`, so each repeat owns one natural KNN bank and
  `finish_bank()` is called at the repeat boundary.
- All outputs are written to local artifact/report/figure directories.

Usage:
- ``python -m evaluation.baselines.run_suite --config ...``
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import torch

from algorithms.baseline import default_algo, random_algo
from algorithms.hu_unit_gate import resolve_hu_unit_gate_config
from algorithms.network_baselines import (
    CASCORE_NAME,
    bw_greedy_algo,
    cascore_algo,
)
from algorithms.runtime_adaptive import RuntimeAdaptiveKNNState
from algorithms.search import improved_searching_algo
from algorithms.slurm import slurm_best_fit_algo
from core.cluster_state import SharedResourceCompatibilityScorer, contention_profiling_session
from evaluation.baselines.common import (
    build_run_tag,
    cluster_algorithm_order,
    load_cluster_resources,
    load_suite_config,
    prepare_output_layout,
    resolve_bandpilot_artifact_dir,
    resolve_bandpilot_model_path,
    resolve_device,
    resolve_external_max_bw_cache_path,
    resolve_linear_model_artifact_dir,
    resolve_linear_model_path,
    set_global_seed,
    write_json,
)
from evaluation.baselines.linear_model import LinearBandwidthRegressor
from evaluation.baselines.report_builder import main as build_report_main
from evaluation.compare import (
    build_single_contention_record,
    build_single_contention_real_manager,
    iter_single_contention_case_contexts,
    prepare_single_contention_runtime_context,
    run_single_contention_search_algorithm,
)
from training.evaluator import prediction_profiling_session

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the baseline-suite runner CLI."""
    parser = argparse.ArgumentParser(description="Run isolated baseline suite")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evaluation/baselines/config/baseline_suite.yaml"),
        help="Path to the isolated baseline-suite config",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip the automatic report-builder invocation",
    )
    return parser.parse_args()


def _run_with_timing(algo_fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float, float, float]:
    """Run a plain heuristic baseline with the same timing profilers used by compare."""
    with prediction_profiling_session() as pred_profiler, contention_profiling_session() as contention_profiler:
        # These baselines are primarily CPU-side heuristics, so wall-clock
        # timing must use perf_counter rather than CUDA events.
        import time

        time_start = time.perf_counter()
        combo = algo_fn(*args, **kwargs)
        elapsed = float(time.perf_counter() - time_start)
        predict_time = pred_profiler.total_time if pred_profiler is not None else 0.0
        contention_time = contention_profiler.total_time if contention_profiler is not None else 0.0
    return combo, elapsed, predict_time, contention_time


def _run_plain_baseline(
    *,
    runtime_context,
    case_context,
    real_manager,
    algorithm: str,
    algo_fn: Callable[..., Any],
    algo_args: Sequence[Any],
    algo_kwargs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Execute a non-search baseline locally, then convert it to a compare-compatible record."""
    combo, elapsed, predict_time, contention_time = _run_with_timing(
        algo_fn,
        *algo_args,
        **algo_kwargs,
    )
    return build_single_contention_record(
        runtime_context=runtime_context,
        case_context=case_context,
        algorithm=algorithm,
        combo=combo,
        elapsed=elapsed,
        predict_time=predict_time,
        contention_time=contention_time,
        search_if_real_data_effective=None,
        job_id=case_context.probe_job_id,
        real_manager=real_manager,
    )


def _load_linear_model(config: Dict[str, Any], cluster_type: str, device: torch.device) -> Tuple[LinearBandwidthRegressor, Path, Path]:
    """Load the previously trained local LinearBW checkpoint for one cluster."""
    artifact_dir = resolve_linear_model_artifact_dir(config, cluster_type)
    model_path = resolve_linear_model_path(config, cluster_type)
    if not model_path.exists():
        raise FileNotFoundError(
            f"LinearBW checkpoint not found: {model_path}. "
            "Run evaluation.baselines.train_linear_bw first."
        )
    model = LinearBandwidthRegressor()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_path, artifact_dir


def _build_bandpilot_runtime_context(
    config: Dict[str, Any],
    cluster_type: str,
    contention_mode: str,
):
    """Construct the read-only BandPilot runtime context reused by the local suite."""
    resources = load_cluster_resources(config, cluster_type)
    suite_cfg = config["suite"]
    model_path = resolve_bandpilot_model_path(config, cluster_type)
    artifact_dir = resolve_bandpilot_artifact_dir(config, cluster_type)
    return prepare_single_contention_runtime_context(
        repeat_num=int(suite_cfg["repeat_num"]),
        total_gpu=resources.total_gpu,
        gpu_bw_dict_list=resources.gpu_bw_dict_list,
        switch_config=resources.switch_config,
        model_path=model_path,
        model_cfg=dict(config["model"]),
        cluster_type=cluster_type,
        training_data_path=resources.training_data_path,
        evaluation_data_path=resources.evaluation_data_path,
        bw_type=str(suite_cfg["bw_type"]),
        artifact_dir=artifact_dir,
        if_dynamic=bool(suite_cfg["if_dynamic"]),
        random_seed=int(config["random_seed"]),
        contention_mode=str(contention_mode),
        search_if_real_data=False,
        max_bw_cache_file=resolve_external_max_bw_cache_path(config, cluster_type, contention_mode),
        adaptive_threshold_policy=None,
    )


def _build_runtime_adaptive_state(
    config: Mapping[str, Any],
    *,
    cluster_type: str,
    contention_mode: str,
) -> RuntimeAdaptiveKNNState:
    """Build the per-mode runtime-adaptive state used by `BandPilot`.

    The baseline suite now aligns with the compare/mainline protocol:
    one `(cluster_type, contention_mode, repeat_idx)` stream corresponds to one
    natural bank, and the caller owns the explicit `finish_bank()` boundary.
    """

    suite_cfg = dict(config["suite"])
    adaptive_runtime_policy = dict(suite_cfg.get("adaptive_runtime_policy", {}))
    return RuntimeAdaptiveKNNState.from_mapping(
        adaptive_runtime_policy,
        bank_id=f"evaluation.baselines:BandPilot:{cluster_type}:{contention_mode}",
    )


def _build_hu_bandpilot_algo_fn(
    runtime_state: RuntimeAdaptiveKNNState,
    *,
    aggressive: bool = False,
) -> Callable[..., Any]:
    """Wrap the mainline runtime-adaptive search entry for the baseline suite."""

    return lambda *args, **kwargs: improved_searching_algo(
        *args,
        aggressive=bool(aggressive),
        adaptive_pts=True,
        adaptive_runtime_state=runtime_state,
        return_metadata=True,
        **kwargs,
    )


def _run_cluster_mode(
    *,
    config: Dict[str, Any],
    cluster_type: str,
    contention_mode: str,
    algorithm_order: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the full local algorithm set for one cluster and one contention mode."""
    runtime_context = _build_bandpilot_runtime_context(config, cluster_type, contention_mode)
    linear_model, linear_model_path, linear_artifact_dir = _load_linear_model(
        config=config,
        cluster_type=cluster_type,
        device=resolve_device(config),
    )
    linear_runtime_context = replace(
        runtime_context,
        model=linear_model,
        model_path=linear_model_path,
        artifact_dir=linear_artifact_dir,
    )

    suite_cfg = config["suite"]
    pair_bw_cache: MutableMapping[Tuple[int, int], float] = {}
    rows: List[Dict[str, Any]] = []
    bank_summaries: List[Dict[str, Any]] = []
    hu_aggressive = resolve_hu_unit_gate_config(
        suite_cfg.get("hu_unit_gate")
    ).aggressive
    runtime_state = _build_runtime_adaptive_state(
        config=config,
        cluster_type=cluster_type,
        contention_mode=contention_mode,
    )
    hu_bandpilot_algo_fn = _build_hu_bandpilot_algo_fn(
        runtime_state,
        aggressive=hu_aggressive,
    )
    test_num_values = [int(value) for value in suite_cfg["test_num_values"]]
    repeat_num = int(suite_cfg["repeat_num"])

    for repeat_idx in range(repeat_num):
        case_iter = iter_single_contention_case_contexts(
            runtime_context,
            test_num_values=test_num_values,
            repeat_indices=[repeat_idx],
        )
        for case_context in case_iter:
            real_manager = build_single_contention_real_manager(runtime_context, case_context)

            for algorithm in algorithm_order:
                if algorithm == "BandPilot":
                    result = run_single_contention_search_algorithm(
                        runtime_context=runtime_context,
                        case_context=case_context,
                        algorithm="BandPilot",
                        algo_fn=hu_bandpilot_algo_fn,
                        use_real_data=False,
                        real_manager=real_manager,
                    )
                    if result["record"] is not None:
                        rows.append(result["record"])
                    continue

                if algorithm == "LinearBW":
                    result = run_single_contention_search_algorithm(
                        runtime_context=linear_runtime_context,
                        case_context=case_context,
                        algorithm="LinearBW",
                        algo_fn=improved_searching_algo,
                        use_real_data=False,
                        real_manager=real_manager,
                    )
                    if result["record"] is not None:
                        rows.append(result["record"])
                    continue

                if algorithm == "Topo":
                    record = _run_plain_baseline(
                        runtime_context=runtime_context,
                        case_context=case_context,
                        real_manager=real_manager,
                        algorithm="Topo",
                        algo_fn=slurm_best_fit_algo,
                        algo_args=(
                            runtime_context.total_gpu,
                            case_context.avail_gpu,
                            case_context.test_num,
                            runtime_context.topo_matrix,
                            runtime_context.gpu_to_node_map,
                        ),
                        algo_kwargs={},
                    )
                    if record is not None:
                        rows.append(record)
                    continue

                if algorithm == "Default":
                    record = _run_plain_baseline(
                        runtime_context=runtime_context,
                        case_context=case_context,
                        real_manager=real_manager,
                        algorithm="Default",
                        algo_fn=default_algo,
                        algo_args=(
                            runtime_context.total_gpu,
                            case_context.avail_gpu,
                            case_context.test_num,
                        ),
                        algo_kwargs={},
                    )
                    if record is not None:
                        rows.append(record)
                    continue

                if algorithm == "Random":
                    record = _run_plain_baseline(
                        runtime_context=runtime_context,
                        case_context=case_context,
                        real_manager=real_manager,
                        algorithm="Random",
                        algo_fn=random_algo,
                        algo_args=(
                            runtime_context.total_gpu,
                            case_context.avail_gpu,
                            case_context.test_num,
                        ),
                        algo_kwargs={},
                    )
                    if record is not None:
                        rows.append(record)
                    continue

                if algorithm == CASCORE_NAME:
                    record = _run_plain_baseline(
                        runtime_context=runtime_context,
                        case_context=case_context,
                        real_manager=real_manager,
                        algorithm=CASCORE_NAME,
                        algo_fn=cascore_algo,
                        algo_args=(
                            runtime_context.total_gpu,
                            case_context.avail_gpu,
                            case_context.test_num,
                            runtime_context.topo_matrix,
                            runtime_context.gpu_to_node_map,
                        ),
                        algo_kwargs={
                            "background_combo": case_context.background_combo,
                            "compatibility_scorer": SharedResourceCompatibilityScorer(real_manager),
                            "shortlist_limit": int(suite_cfg.get("cascore_shortlist_limit", 12)),
                            "extra_node_slack": int(suite_cfg.get("cascore_extra_node_slack", 1)),
                            "penalty_weight": float(suite_cfg["network_locality_penalty_weight"]),
                        },
                    )
                    if record is not None:
                        rows.append(record)
                    continue

                if algorithm == "BWGreedy":
                    record = _run_plain_baseline(
                        runtime_context=runtime_context,
                        case_context=case_context,
                        real_manager=real_manager,
                        algorithm="BWGreedy",
                        algo_fn=bw_greedy_algo,
                        algo_args=(
                            runtime_context.total_gpu,
                            case_context.avail_gpu,
                            case_context.test_num,
                            runtime_context.gpu_bw_dict_list,
                            runtime_context.switch_config,
                            runtime_context.evaluation_data_path,
                            runtime_context.gpu_to_node_map,
                        ),
                        algo_kwargs={
                            "background_combo": case_context.background_combo,
                            "pair_bw_cache": pair_bw_cache,
                            "penalty_weight": float(suite_cfg["bw_greedy_penalty_weight"]),
                        },
                    )
                    if record is not None:
                        rows.append(record)
                    continue

                raise ValueError(f"Unsupported algorithm in isolated baseline suite: {algorithm}")

        bank_summary = dict(runtime_state.finish_bank())
        bank_summary.update(
            {
                "cluster_type": str(cluster_type),
                "contention_mode": str(contention_mode),
                "repeat_idx": int(repeat_idx),
            }
        )
        logger.info(
            "Baseline BandPilot bank finished | cluster=%s | mode=%s | repeat=%s | version=%s | active_next=%s",
            cluster_type,
            contention_mode,
            repeat_idx,
            bank_summary.get("bank_version_after_finish"),
            bank_summary.get("bank_is_active_after_finish"),
        )
        bank_summaries.append(bank_summary)

    return rows, bank_summaries


def main() -> None:
    """Run the isolated baseline suite and save all new outputs locally."""
    args = parse_args()
    config = load_suite_config(args.config)
    set_global_seed(int(config["random_seed"]))
    output_layout = prepare_output_layout(config)
    algorithm_order = cluster_algorithm_order(config)

    all_rows: List[Dict[str, Any]] = []
    all_bank_summaries: List[Dict[str, Any]] = []
    for cluster_type in config["cluster"]["cluster_types"]:
        for contention_mode in config["suite"]["contention_modes"]:
            cluster_rows, cluster_bank_summaries = _run_cluster_mode(
                config=config,
                cluster_type=str(cluster_type),
                contention_mode=str(contention_mode),
                algorithm_order=algorithm_order,
            )
            all_rows.extend(cluster_rows)
            all_bank_summaries.extend(cluster_bank_summaries)

    rows_df = pd.DataFrame(all_rows)
    rows_path = output_layout["artifact_dir"] / "rows.csv"
    rows_df.to_csv(rows_path, index=False)

    reference_algorithm = str(config["suite"].get("reference_algorithm", "BandPilot"))
    metadata = {
        "run_tag": build_run_tag(config),
        "config_path": str(args.config),
        "rows_path": str(rows_path),
        "algorithm_order": list(algorithm_order),
        "reference_algorithm": reference_algorithm,
        "execution_order": "repeat_idx_outer_then_test_num",
        "bank_partition_mode": str(
            config["suite"].get("adaptive_runtime_policy", {}).get("bank_partition_mode", "repeat")
        ),
        "adaptive_runtime_policy": dict(config["suite"].get("adaptive_runtime_policy", {})),
        "hu_unit_gate": dict(config["suite"].get("hu_unit_gate", {})),
        "bank_summaries": all_bank_summaries,
        "cluster_types": list(config["cluster"]["cluster_types"]),
        "contention_modes": list(config["suite"]["contention_modes"]),
        "test_num_values": list(config["suite"]["test_num_values"]),
    }
    write_json(output_layout["artifact_dir"] / "metadata.json", metadata)
    print(f"Baseline suite rows saved to {rows_path}")

    if not args.skip_report:
        build_report_main_args = [
            "report_builder",
            "--artifact-dir",
            str(output_layout["artifact_dir"]),
            "--report-dir",
            str(output_layout["report_dir"]),
            "--figure-dir",
            str(output_layout["figure_dir"]),
        ]
        # The report builder is invoked via a direct module import below rather
        # than a subprocess so the suite stays self-contained in one Python run.
        import sys

        previous_argv = list(sys.argv)
        try:
            sys.argv = build_report_main_args
            build_report_main()
        finally:
            sys.argv = previous_argv


if __name__ == "__main__":
    main()
