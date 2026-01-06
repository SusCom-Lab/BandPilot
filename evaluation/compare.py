"""Algorithm comparison and statistical analysis utilities."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

from algorithms.baseline import default_algo, random_algo
from algorithms.eha import eha_search
from algorithms.search import improved_searching_algo, tree_search_only
from algorithms.slurm import slurm_best_fit_algo
from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values
from core.cluster_state import (
    ClusterStateManager,
    create_bandwidth_predictor,
    contention_profiling_session,
    normalize_contention_mode,
)
from core.topology import (
    build_composite_topo_matrix,
    convert_cluster_type_to_node_configs,
    create_gpu_to_node_map,
)
from evaluation.metrics import  find_max_bw_for_k_gpus_with_contention
from models.bandwidth_predictor import BandwidthPredictor
from training.evaluator import prediction_profiling_session

logger = logging.getLogger(__name__)

BACKGROUND_JOB_ID = -1  # Shared job_id for background tasks to avoid probe-job collision


def _load_predictor(model_path: Path, device: torch.device, model_cfg: dict) -> BandwidthPredictor:
    model = BandwidthPredictor(
        input_dim=model_cfg.get("input_dim", 1),
        node_count_embedding_dim=model_cfg.get("node_count_embedding_dim", 8),
        total_count_feature_dim=model_cfg.get("total_count_feature_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 3),
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.05),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _run_and_record(algo_fn, *args, **kwargs):
    with prediction_profiling_session() as pred_profiler, contention_profiling_session() as contention_profiler:
        start = time.perf_counter()
        combo = algo_fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        predict_time = pred_profiler.total_time if pred_profiler is not None else 0.0
        contention_time = contention_profiler.total_time if contention_profiler is not None else 0.0
    return combo, elapsed, predict_time, contention_time


def _sample_available_gpu(total_gpu: int, test_num: int, if_dynamic: bool, random_seed: Optional[int] = None) -> np.ndarray:
    """Sample available GPUs.

    Args:
        total_gpu: Total GPU count.
        test_num: Minimum GPUs required.
        if_dynamic: Whether to sample dynamically.
        random_seed: Optional RNG seed.

    Returns:
        Array of available GPU indices.
    """
    if if_dynamic:
        if random_seed is not None:
            np.random.seed(random_seed)
        avail_gpu_num = np.random.randint(test_num, total_gpu + 1)
        return np.random.choice(total_gpu, avail_gpu_num, replace=False)
    return np.arange(total_gpu)


def _build_contention_job_id(test_num: int, repeat_idx: int, algo_offset: int) -> int:
    """Build a stable job_id for contention experiments to keep common-mode ratios repeatable."""
    # Simple linear combination avoids Python hash randomization drift
    return int(test_num * 10_000 + repeat_idx * 100 + algo_offset)


def _build_avail_signature(avail_gpu: Sequence[int]) -> str:
    """Construct a signature from sorted available GPU indices for cache consistency verification."""
    return ",".join(str(int(idx)) for idx in sorted(int(x) for x in avail_gpu))


def _build_combo_signature(combo: Sequence[int]) -> str:
    """Construct a signature from selected GPU indices for subsequent identification."""
    return ",".join(str(idx) for idx, flag in enumerate(combo) if int(flag) == 1)


def _build_base_record(
    *,
    test_num: int,
    total_gpu: int,
    repeat_idx: int,
    bw_type: str,
    cluster_type: str,
    if_dynamic: bool,
    seed: Optional[int],
    avail_gpu: Sequence[int],
    max_bw: float,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Construct a generic experimental context field for reuse in result rows."""
    record: Dict[str, object] = {
        "test_num": test_num,
        "repeat_idx": repeat_idx,
        "total_gpu": total_gpu,
        "bw_type": bw_type,
        "cluster_type": cluster_type,
        "if_dynamic": if_dynamic,
        "seed_used": seed,
        "avail_gpu_count": int(len(avail_gpu)),
        "avail_signature": _build_avail_signature(avail_gpu),
        "max_bw": float(max_bw),
    }
    if extra:
        record.update(extra)
    return record


def _select_data_path(use_real_data: bool, training_data_path: str, evaluation_data_path: str) -> str:
    """Return the corresponding data path based on the evaluation mode."""
    return evaluation_data_path if use_real_data else training_data_path


def build_max_bw_cache_filename(
    random_seed: int,
    num_train_samples: int,
    total_gpu: int,
    repeat_num: int,
    if_dynamic: bool,
    contention_mode: str,
    search_if_real_data: bool,
    local_top_k: int,
    max_combos_per_distribution: int,
    max_total_combos: int,
) -> str:
    """Build the max_bw offline cache filename from the config."""
    contention_mode = normalize_contention_mode(contention_mode)
    return (
        "MaxBW_"
        f"{random_seed}RS_"
        f"{num_train_samples}TD_"
        f"{total_gpu}GPU_"
        f"{repeat_num}RN_"
        f"{if_dynamic}Dy_"
        f"{contention_mode}CM_"
        f"{'real' if search_if_real_data else 'model'}SM_"
        f"{local_top_k}LTK_"
        f"{max_combos_per_distribution}CPD_"
        f"{max_total_combos}TC.csv"
    )


def _create_cluster_manager(
    *,
    use_real_data: bool,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    evaluation_data_path: str,
    model,
    device,
    artifact_dir: Path,
    contention_mode: str,
    background_combo: np.ndarray,
    occupancy_seed: int,
) -> ClusterStateManager:
    contention_mode = normalize_contention_mode(contention_mode)  # normalize to avoid mode mismatch
    predictor = create_bandwidth_predictor(
        if_real_data=use_real_data,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
        evaluation_data_path=evaluation_data_path,
        model=model if not use_real_data else None,
        device=device if not use_real_data else None,
        artifact_dir=artifact_dir if not use_real_data else None,
    )
    manager = ClusterStateManager(
        total_gpu=total_gpu,
        bandwidth_predictor=predictor,
        contention_mode=contention_mode,
        occupancy_seed=occupancy_seed,
    )
    if np.any(background_combo):
        manager.allocate_job(job_id=BACKGROUND_JOB_ID, combo=background_combo)
    return manager


def collect_single_contention_max_bw_data(
    repeat_num: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    cluster_type: str,
    training_data_path: str,
    evaluation_data_path: str,
    artifact_dir: Path,
    if_dynamic: bool,
    random_seed: int,
    contention_mode: str,
    search_if_real_data: bool,
    max_bw_options: Optional[dict] = None,
) -> pd.DataFrame:
    """Collect offline max_bw for single-contention scenarios."""
    contention_mode = normalize_contention_mode(contention_mode)  # normalize for consistent cache naming
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_predictor(model_path, device, model_cfg)

    node_configs = convert_cluster_type_to_node_configs(cluster_type, total_gpu)
    _topo_matrix, _ = build_composite_topo_matrix(node_configs)

    options = max_bw_options or {}
    local_top_k = int(options.get("local_top_k", 3))
    max_combos_per_distribution = int(options.get("max_combos_per_distribution", 2048))
    max_total_combos = int(options.get("max_total_combos", 100000))

    records = []
    for test_num in range(2, total_gpu):
        print("===== Single-contention max_bw offline collection: current GPU count %s =====", test_num)
        for repeat_idx in range(repeat_num):
            current_seed = None
            if random_seed is not None:
                current_seed = random_seed + repeat_idx
                np.random.seed(current_seed)
                import random as _random

                _random.seed(current_seed)

            avail_gpu = _sample_available_gpu(total_gpu, test_num, if_dynamic, random_seed=current_seed)

            mask_gpu = sorted(set(range(total_gpu)) - set(int(idx) for idx in avail_gpu))
            background_combo = np.zeros(total_gpu, dtype=int)
            if mask_gpu:
                background_combo[mask_gpu] = 1

            if current_seed is not None:
                occupancy_seed = current_seed + test_num * 97
            else:
                occupancy_seed = int(np.random.randint(0, 2**31 - 1))

            real_manager = _create_cluster_manager(
                use_real_data=True,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=training_data_path,
                evaluation_data_path=evaluation_data_path,
                model=model,
                device=device,
                artifact_dir=artifact_dir,
                contention_mode=contention_mode,
                background_combo=background_combo,
                occupancy_seed=occupancy_seed,
            )

            # probe_job_id represents the foreground task; offline optimum and online algos
            # share the same job_id so common-mode occupancy stays consistent.
            probe_job_id = _build_contention_job_id(test_num, repeat_idx, 0)
            max_bw, _ = find_max_bw_for_k_gpus_with_contention(
                k=test_num,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                avail_gpu=avail_gpu,
                cluster_manager=real_manager,
                job_id=probe_job_id,
                data_path=evaluation_data_path,
                local_top_k=local_top_k,
                max_combos_per_distribution=max_combos_per_distribution,
                max_total_combos=max_total_combos,
            )

            records.append(
                {
                    "test_num": test_num,
                    "repeat_idx": repeat_idx,
                    "max_bw": max_bw,
                    "avail_signature": _build_avail_signature(avail_gpu),
                    "contention_mode": contention_mode,
                    "if_dynamic": if_dynamic,
                    "search_if_real_data": search_if_real_data,
                    "local_top_k": local_top_k,
                    "max_combos_per_distribution": max_combos_per_distribution,
                    "max_total_combos": max_total_combos,
                }
            )

    return pd.DataFrame(records)


def get_single_dispatch_with_contention_data(
    repeat_num: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    cluster_type: str,
    training_data_path: str,
    evaluation_data_path: str,
    bw_type: str,
    artifact_dir: Path,
    if_dynamic: bool = True,
    random_seed: Optional[int] = None,
    contention_mode: str = "common",
    search_if_real_data: bool = False,
    max_bw_cache_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Single-dispatch + background-contention experiments (returns per-run records).

    Assumes masked GPUs represent existing jobs and uses ClusterStateManager (common/intensive/idle)
    to simulate available bandwidth under contention. For each test_num and repeat:

    1) Sample available GPUs (others are treated as background-occupied).
    2) Build ClusterStateManager in model and real-data modes; register a background job covering masked GPUs.
    3) Run multiple algorithms. Cluster-aware ones score with contention-aware bandwidth; baselines keep their logic.
    4) Re-evaluate each combo with the real-data manager under current background; also record standalone bandwidth.
    5) Aggregate per-run results with elapsed/predict/contention timing.

    Args:
        repeat_num: Repetitions per test_num.
        total_gpu: Total GPUs.
        gpu_bw_dict_list: Per-node bandwidth dictionaries.
        switch_config: Switch configuration.
        model_path: Bandwidth predictor weights path.
        model_cfg: Model config.
        cluster_type: Cluster type (drives topology config).
        training_data_path: Data for training/model prediction.
        evaluation_data_path: Data for real-data evaluation.
        bw_type: Bandwidth tag in results.
        artifact_dir: Artifact directory.
        if_dynamic: Whether to sample available GPUs dynamically.
        random_seed: RNG seed.
        contention_mode: ClusterStateManager contention mode (default common).
        search_if_real_data: Whether search uses real data (default False -> model).

    Returns:
        pandas.DataFrame of per-run results with final_bw/standalone_bw/utilization and timing fields.
    """
    contention_mode = normalize_contention_mode(contention_mode)  # normalize for manager/logs
    if max_bw_cache_file is None:
        raise ValueError("single_dispatch_with_contention requires max_bw_cache_file.")
    if not max_bw_cache_file.exists():
        print(f"[MaxBW] cache file not found: {max_bw_cache_file}; run max_bw_offline first.")
        raise FileNotFoundError(f"max_bw cache missing: {max_bw_cache_file}")

    cache_df = pd.read_csv(max_bw_cache_file)
    max_bw_cache: Dict[Tuple[int, int], Dict[str, object]] = {}
    for _, row in cache_df.iterrows():
        key = (int(row["test_num"]), int(row["repeat_idx"]))
        max_bw_cache[key] = {
            "max_bw": float(row["max_bw"]),
            "avail_signature": str(row.get("avail_signature", "")),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_predictor(model_path, device, model_cfg)

    node_configs = convert_cluster_type_to_node_configs(cluster_type, total_gpu)
    topo_matrix, _ = build_composite_topo_matrix(node_configs)
    gpu_to_node_map = create_gpu_to_node_map(node_configs)

    algo_order = ["BandPilot", "Default", "EHA",  "Tree", "Topo", "Random", "UpperBandPilot"] 
    results = []

    for test_num in range(2, total_gpu):
        print("===== Single-contention experiment: current test GPU count %s =====", test_num)

        for repeat_idx in range(repeat_num):
            current_seed = None
            if random_seed is not None:
                current_seed = random_seed + repeat_idx
                np.random.seed(current_seed)
                import random
                random.seed(current_seed)

            avail_gpu = _sample_available_gpu(total_gpu, test_num, if_dynamic, random_seed=current_seed)
            if len(avail_gpu) < test_num:
                logger.warning(
                    "Available GPUs insufficient (test_num=%s, avail=%s); skip this repeat",
                    test_num,
                    len(avail_gpu),
                )
                continue

            mask_gpu = sorted(set(range(total_gpu)) - set(int(idx) for idx in avail_gpu))
            background_combo = np.zeros(total_gpu, dtype=int)
            if mask_gpu:
                background_combo[mask_gpu] = 1

            occupancy_seed = (
                current_seed + test_num * 97 if current_seed is not None else int(np.random.randint(0, 2**31 - 1))
            )

            real_manager = _create_cluster_manager(
                use_real_data=True,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=training_data_path,
                evaluation_data_path=evaluation_data_path,
                model=model,
                device=device,
                artifact_dir=artifact_dir,
                contention_mode=contention_mode,
                background_combo=background_combo,
                occupancy_seed=occupancy_seed,
            )
            if search_if_real_data:
                search_manager = real_manager
            else:
                search_manager = _create_cluster_manager(
                    use_real_data=False,
                    total_gpu=total_gpu,
                    gpu_bw_dict_list=gpu_bw_dict_list,
                    switch_config=switch_config,
                    training_data_path=training_data_path,
                    evaluation_data_path=evaluation_data_path,
                    model=model,
                    device=device,
                    artifact_dir=artifact_dir,
                    contention_mode=contention_mode,
                    background_combo=background_combo,
                    occupancy_seed=occupancy_seed,
                )

            cache_key = (test_num, repeat_idx)
            cache_entry = max_bw_cache.get(cache_key)
            if cache_entry is None:
                raise ValueError(
                    f"max_bw cache missing test_num={test_num}, repeat_idx={repeat_idx}; "
                    "please re-run max_bw_offline collection."
                )
            actual_signature = _build_avail_signature(avail_gpu)
            expected_signature = cache_entry.get("avail_signature", "")
            if expected_signature and actual_signature != expected_signature:
                raise ValueError(
                    "max_bw cache inconsistent with current sample; random_seed/if_dynamic may differ. "
                    "Please re-run max_bw_offline collection."
                )
            max_bw = float(cache_entry["max_bw"])
            if max_bw <= 0:
                logger.warning("max_bw<=0, skip this configuration")
                continue

            probe_job_id = _build_contention_job_id(test_num, repeat_idx, 0)

            def _evaluate_with_real_manager(job_id: int, combo: np.ndarray) -> Tuple[float, float, float]:
                """Real-data evaluation helper: returns (final_bw, standalone_bw, contention_time)."""
                with contention_profiling_session() as eval_profiler:
                    try:
                        real_manager.set_job_context(job_id)
                        final_bw = float(real_manager.predict_with_contention(combo))
                    finally:
                        real_manager.clear_job_context()
                    contention_time = eval_profiler.total_time if eval_profiler is not None else 0.0
                standalone_bw = float(real_manager.bandwidth_predictor(combo))
                return final_bw, standalone_bw, contention_time

            base_record = _build_base_record(
                test_num=test_num,
                total_gpu=total_gpu,
                repeat_idx=repeat_idx,
                bw_type=bw_type,
                cluster_type=cluster_type,
                if_dynamic=if_dynamic,
                seed=current_seed,
                avail_gpu=avail_gpu,
                max_bw=max_bw,
                extra={
                    "contention_mode": contention_mode,
                    "search_if_real_data_global": search_if_real_data,
                    "background_gpu_count": len(mask_gpu),
                    "background_signature": _build_combo_signature(background_combo),
                    "occupancy_seed": occupancy_seed,
                    "probe_job_id": probe_job_id,
                    "max_bw_cache_file": str(max_bw_cache_file),
                    "max_bw_cache_avail_signature": expected_signature,
                },
            )

            # Helper: unify validation, real evaluation, and record writing for algorithms
            def _append_algo_record(
                *,
                algorithm: str,
                combo,
                elapsed: float,
                predict_time: float,
                contention_time: float,
                search_if_real_data_effective: Optional[bool],
                job_id: int,
            ) -> None:
                if combo is None:
                    return
                combo_arr = np.asarray(combo, dtype=int)
                if combo_arr.size != total_gpu:
                    logger.warning("%s produced an invalid-dimension combo; skipped", algorithm)
                    return
                if np.any(combo_arr == 1) and mask_gpu:
                    overlap = [idx for idx in mask_gpu if combo_arr[idx] == 1]
                    if overlap:
                        logger.warning("%s selected GPUs occupied by background tasks (sample %s); ignored", algorithm, overlap[:4])
                        return
                try:
                    final_bw, standalone_bw, eval_contention_time = _evaluate_with_real_manager(job_id, combo_arr)
                except ValueError as exc:
                    logger.error("%s combo evaluation failed: %s", algorithm, exc)
                    return
                total_contention_time = contention_time + eval_contention_time

                record = dict(base_record)
                final_util = float(final_bw) / max_bw * 100 if max_bw > 0 else 0.0
                standalone_util = float(standalone_bw) / max_bw * 100 if max_bw > 0 else 0.0
                record.update(
                    {
                        "algorithm": algorithm,
                        "search_if_real_data_effective": search_if_real_data_effective,
                        "final_bw": float(final_bw),
                        "standalone_bw": float(standalone_bw),
                        "final_utilization": final_util,
                        "standalone_utilization": standalone_util,
                        "elapsed_time": elapsed,
                        "predict_time": predict_time,
                        "contention_time": total_contention_time,
                        "selected_gpu_count": int(np.sum(combo_arr)),
                        "combo_signature": _build_combo_signature(combo_arr),
                        "contention_job_id": job_id,
                    }
                )
                results.append(record)
                logger.info(
                    "[single contention] Algo=%s, test_num=%s, repeat=%s, final_bw=%.2f, standalone=%.2f, utilization=%.2f%%",
                    algorithm,
                    test_num,
                    repeat_idx,
                    final_bw,
                    standalone_bw,
                    final_util,
                )

            search_data_path = _select_data_path(search_if_real_data, training_data_path, evaluation_data_path)

            for algo_idx, name in enumerate(algo_order):
                job_id = probe_job_id
                combo = None

                if name == "BandPilot":
                    search_manager.set_job_context(job_id)
                    try:
                        combo, elapsed, predict_time, contention_time = _run_and_record(
                            lambda *a, **kw: improved_searching_algo(*a, **kw),
                            total_gpu,
                            avail_gpu,
                            model,
                            test_num,
                            total_gpu,
                            gpu_bw_dict_list,
                            switch_config,
                            search_data_path,
                            device,
                            artifact_dir,
                            search_if_real_data,
                            cluster_manager=search_manager,
                            evaluation_data_path=evaluation_data_path,
                        )
                    finally:
                        search_manager.clear_job_context()
                    _append_algo_record(
                        algorithm=name,
                        combo=combo,
                        elapsed=elapsed,
                        predict_time=predict_time,
                        contention_time=contention_time,
                        search_if_real_data_effective=search_if_real_data,
                        job_id=job_id,
                    )

                elif name == "Default":
                    combo, elapsed, predict_time, contention_time = _run_and_record(
                        default_algo,
                        total_gpu,
                        avail_gpu,
                        test_num,
                    )
                    _append_algo_record(
                        algorithm=name,
                        combo=combo,
                        elapsed=elapsed,
                        predict_time=predict_time,
                        contention_time=contention_time,
                        search_if_real_data_effective=None,
                        job_id=job_id,
                    )

                elif name == "Tree":
                    search_manager.set_job_context(job_id)
                    try:
                        combo, elapsed, predict_time, contention_time = _run_and_record(
                            lambda *a, **kw: tree_search_only(*a, **kw),
                            total_gpu,
                            avail_gpu,
                            model,
                            test_num,
                            total_gpu,
                            gpu_bw_dict_list,
                            switch_config,
                            search_data_path,
                            device,
                            artifact_dir,
                            search_if_real_data,
                            cluster_manager=search_manager,
                            evaluation_data_path=evaluation_data_path,
                        )
                    finally:
                        search_manager.clear_job_context()
                    _append_algo_record(
                        algorithm=name,
                        combo=combo,
                        elapsed=elapsed,
                        predict_time=predict_time,
                        contention_time=contention_time,
                        search_if_real_data_effective=search_if_real_data,
                        job_id=job_id,
                    )

                elif name == "EHA":
                    search_manager.set_job_context(job_id)
                    try:
                        combo, elapsed, predict_time, contention_time = _run_and_record(
                            lambda *a, **kw: eha_search(*a, **kw),
                            total_gpu,
                            avail_gpu,
                            model,
                            test_num,
                            total_gpu,
                            gpu_bw_dict_list,
                            switch_config,
                            search_data_path,
                            device,
                            artifact_dir,
                            search_if_real_data,
                            cluster_manager=search_manager,
                            evaluation_data_path=evaluation_data_path,
                        )
                    finally:
                        search_manager.clear_job_context()
                    _append_algo_record(
                        algorithm=name,
                        combo=combo,
                        elapsed=elapsed,
                        predict_time=predict_time,
                        contention_time=contention_time,
                        search_if_real_data_effective=search_if_real_data,
                        job_id=job_id,
                    )

                elif name == "Topo":
                    combo, elapsed, predict_time, contention_time = _run_and_record(
                        lambda *a, **kw: slurm_best_fit_algo(*a, **kw),
                        total_gpu,
                        avail_gpu,
                        test_num,
                        topo_matrix,
                        gpu_to_node_map,
                    )
                    _append_algo_record(
                        algorithm=name,
                        combo=combo,
                        elapsed=elapsed,
                        predict_time=predict_time,
                        contention_time=contention_time,
                        search_if_real_data_effective=None,
                        job_id=job_id,
                    )

                elif name == "Random":
                    combo, elapsed, predict_time, contention_time = _run_and_record(
                        random_algo,
                        total_gpu,
                        avail_gpu,
                        test_num,
                    )
                    _append_algo_record(
                        algorithm=name,
                        combo=combo,
                        elapsed=elapsed,
                        predict_time=predict_time,
                        contention_time=contention_time,
                        search_if_real_data_effective=None,
                        job_id=job_id,
                    )

                elif name == "UpperBandPilot":
                    real_manager.set_job_context(job_id)
                    try:
                        combo, elapsed, predict_time, contention_time = _run_and_record(
                            lambda *a, **kw: improved_searching_algo(*a, **kw),
                            total_gpu,
                            avail_gpu,
                            model,
                            test_num,
                            total_gpu,
                            gpu_bw_dict_list,
                            switch_config,
                            training_data_path,
                            device,
                            artifact_dir,
                            True,
                            cluster_manager=real_manager,
                            evaluation_data_path=evaluation_data_path,
                        )
                    finally:
                        real_manager.clear_job_context()
                    _append_algo_record(
                        algorithm=name,
                        combo=combo,
                        elapsed=elapsed,
                        predict_time=predict_time,
                        contention_time=contention_time,
                        search_if_real_data_effective=True,
                        job_id=job_id,
                    )

                else:
                    logger.warning("Unknown algorithm: %s (idx=%s), skipping", name, algo_idx)

    return pd.DataFrame(results)



def build_single_experiment_filename(
    metric_type: str,
    random_seed: int,
    contention_mode: Optional[str],
    num_train_samples: int,
    if_dynamic: bool,
    total_gpu: int,
    repeat_num: int,
) -> str:
    """Build filename for single GPU allocation experiment (utilization/accumulation).

    Naming:
        Single_{utilization/accumulation}_{random_seed}RS_{num_train_samples}TD_{if_dynamic}Dy_{total_gpu}GPU_{repeat_num}RN.csv

    Args:
        metric_type: 'utilization' or 'accumulation'
        random_seed: top-level random_seed
        num_train_samples: training.num_train_samples
        if_dynamic: whether to sample dynamically (evaluation.if_dynamic)
        total_gpu: cluster.total_gpu
        repeat_num: repeats per experiment (evaluation.repeat_num)
    """
    prefix = ""
    if contention_mode:
        # If contention_mode is provided, normalize to lowercase to keep filenames consistent
        prefix = f"{normalize_contention_mode(contention_mode)}CM_"
    return (
        f"Single_{metric_type}_"
        f"{prefix}"
        f"{random_seed}RS_"
        f"{num_train_samples}TD_"
        f"{if_dynamic}Dy_"
        f"{total_gpu}GPU_"
        f"{repeat_num}RN"
    )
