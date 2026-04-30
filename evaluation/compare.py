"""Single-contention evaluation and algorithm-comparison utilities.

The module generates offline `MaxBW_*` caches and `Single_contention_*.csv`
comparison artifacts. It keeps background contention, occupancy seeds, probe
job IDs, and domain-isolated lookup timing aligned across BandPilot and the
reviewer-facing baselines.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

from algorithms.adaptive_knn import (
    AdaptiveKNNConfig,
    build_adaptive_knn_feature_rows,
    build_compare_records_from_replay,
    run_adaptive_knn_replay,
)
from algorithms.adaptive_policy import resolve_adaptive_thresholds
from algorithms.baseline import default_algo, random_algo
from algorithms.eha import eha_search
from algorithms.hu_unit_gate import resolve_hu_unit_gate_config
from algorithms.linear_bw import load_linear_bw_model, resolve_linear_bw_model_path
from algorithms.network_baselines import (
    CASCORE_NAME,
    LEGACY_NETWORK_LOCALITY_NAME,
    bw_greedy_algo,
    cascore_algo,
    normalize_network_baseline_name,
)
from algorithms.runtime_adaptive import RuntimeAdaptiveKNNState
from algorithms.search import (
    hu_pts_only_search,
    improved_searching_algo,
    tree_search_only,
    legacy_improved_searching_algo,
    threshold_legacy_exact_improved_searching_algo,
    threshold_legacy_improved_searching_algo,
)
from algorithms.slurm import slurm_best_fit_algo
from core.bandwidth import BandwidthLookupCache, SwitchBandwidthConfig, calculate_bandwidth_values
from core.cluster_state import (
    ClusterStateManager,
    SharedResourceCompatibilityScorer,
    create_bandwidth_predictor,
    create_bandwidth_predictor_batch,
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
from utils.helpers import read_active_num_train_samples

logger = logging.getLogger(__name__)

BACKGROUND_JOB_ID = -1  # Shared job_id for background tasks to avoid probe-job collision
BANDPILOT_LABEL = "BandPilot"
LEGACY_BANDPILOT_LABEL = "legacy-BandPilot"
PTS_LABEL = "PTS"
LEGACY_PTS_LABEL = "legacy-PTS"
TREE_LABEL = "tree"
LEGACY_BANDPILOT_KNN_LABEL = "legacy-BandPilot-KNN"
# Backward-compatible aliases for historical result files. Public outputs use
# the canonical labels above.
ALGORITHM_NAME_ALIASES = {
    "HU-" + "BandPilot": BANDPILOT_LABEL,
    "HU-" + "Adaptive": BANDPILOT_LABEL,
    "HU-" + "PTS": PTS_LABEL,
    "HU-" + "PTS-only": PTS_LABEL,
    "BandPilot": BANDPILOT_LABEL,
    "Adaptive": LEGACY_BANDPILOT_KNN_LABEL,
    "AdaptiveThresholdLegacy": LEGACY_BANDPILOT_KNN_LABEL,
    "Tree": TREE_LABEL,
    "PTS-only": LEGACY_PTS_LABEL,
}
MODEL_DOMAIN_ALGORITHMS = {
    BANDPILOT_LABEL,
    LEGACY_BANDPILOT_LABEL,
    PTS_LABEL,
    LEGACY_PTS_LABEL,
    "EHA",
    TREE_LABEL,
    LEGACY_BANDPILOT_KNN_LABEL,
    "LinearBW",
}
REAL_DOMAIN_ALGORITHMS = {"UpperBandPilot"}


def normalize_algorithm_label(name: object) -> str:
    """Return the canonical public algorithm label while accepting legacy names."""

    raw_name = str(name).strip()
    return ALGORITHM_NAME_ALIASES.get(raw_name, raw_name)


@dataclass(frozen=True)
class SingleContentionRuntimeContext:
    """Runtime resources shared by a complete `single_contention` stream.

    The context owns expensive or stateful objects that must stay aligned across
    all `(k, repeat)` cases: the predictor, topology maps, max-bandwidth cache,
    adaptive thresholds, and experiment paths. Both the local compare runner and
    replay-style kNN validation consume this object so they operate on the same
    sample stream and cache state.
    """

    repeat_num: int
    total_gpu: int
    gpu_bw_dict_list: object
    switch_config: SwitchBandwidthConfig
    model_path: Path
    model_cfg: Dict[str, object]
    cluster_type: str
    training_data_path: str
    evaluation_data_path: str
    bw_type: str
    artifact_dir: Path
    if_dynamic: bool
    random_seed: Optional[int]
    contention_mode: str
    search_if_real_data: bool
    max_bw_cache_file: Path
    model: BandwidthPredictor
    device: torch.device
    topo_matrix: object
    gpu_to_node_map: object
    adaptive_thresholds: object
    max_bw_cache: Dict[Tuple[int, int], Dict[str, object]]


@dataclass(frozen=True)
class SingleContentionCaseContext:
    """Per-case state for one `single_contention` `(k, repeat)` dispatch.

    The case captures the sampled available GPUs, background occupancy mask,
    occupancy seed, probe job id, cached max bandwidth, and shared CSV fields.
    Search algorithms should only add algorithm-specific outputs on top of this
    common case state.
    """

    test_num: int
    repeat_idx: int
    seed_used: Optional[int]
    avail_gpu: np.ndarray
    mask_gpu: List[int]
    background_combo: np.ndarray
    occupancy_seed: int
    probe_job_id: int
    max_bw: float
    base_record: Dict[str, object]


def _metadata_to_csv_safe_fields(metadata: Mapping[str, object]) -> Dict[str, object]:
    """Flatten algorithm metadata into compare-CSV-safe scalar fields.

    `search.py` can return nested metadata for runtime adaptation, PTS, and EHA
    confidence diagnostics. The compare CSV stores scalar values directly and
    serializes structured values as deterministic JSON strings so downstream
    report builders can parse them without losing traceability.
    """

    csv_safe_fields: Dict[str, object] = {}
    for key, value in dict(metadata).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            csv_safe_fields[str(key)] = value
        else:
            csv_safe_fields[str(key)] = json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
            )
    return csv_safe_fields


def _resolve_hu_bandpilot_runtime_adaptive_enabled(
    adaptive_runtime_policy: Optional[Mapping[str, object]],
) -> bool:
    """Return whether compare should run `BandPilot` with runtime kNN adaptation."""

    raw_policy = dict(adaptive_runtime_policy or {})
    return str(raw_policy.get("mode", "")).strip().lower() == "adaptive_knn"


def _build_compare_runtime_adaptive_state(
    *,
    adaptive_runtime_policy: Optional[Mapping[str, object]],
    cluster_type: str,
    contention_mode: str,
    bank_scope: str,
) -> Optional[RuntimeAdaptiveKNNState]:
    """Build the per-stream runtime adaptive state for `BandPilot`."""

    if not _resolve_hu_bandpilot_runtime_adaptive_enabled(adaptive_runtime_policy):
        return None
    normalized_mode = normalize_contention_mode(contention_mode)
    return RuntimeAdaptiveKNNState.from_mapping(
        adaptive_runtime_policy,
        bank_id=(
            f"compare:{bank_scope}:BandPilot:"
            f"{cluster_type}:{normalized_mode}"
        ),
    )


def _log_runtime_bank_summary(
    *,
    summary: Mapping[str, object],
    cluster_type: str,
    contention_mode: str,
) -> None:
    """Log the final state of one compare-stream runtime-adaptive bank."""

    logger.info(
        "BandPilot runtime bank finished | cluster=%s | mode=%s | bank=%s | version=%s | "
        "active_next=%s | labeled=%s | unlabeled=%s | unsafe_skip=%.2f | over_trigger=%.2f",
        cluster_type,
        contention_mode,
        summary.get("bank_id", ""),
        summary.get("bank_version", -1),
        bool(summary.get("bank_active_next", False)),
        int(summary.get("bank_size_labeled", 0)),
        int(summary.get("bank_size_unlabeled_skips", 0)),
        float(summary.get("shadow_unsafe_skip_rate_pct", 0.0)),
        float(summary.get("shadow_over_trigger_rate_pct", 0.0)),
    )


def _build_hu_bandpilot_algo_fn(
    runtime_state: Optional[RuntimeAdaptiveKNNState],
    *,
    aggressive: bool = False,
):
    """Build the compare wrapper for the public `BandPilot` search path.

    Without runtime state the wrapper runs the main `search.py` path and returns
    metadata. With runtime state it enables adaptive PTS so the same public label
    can be evaluated with online kNN-triggered refinement.
    """

    if runtime_state is None:
        return lambda *args, **kwargs: improved_searching_algo(
            *args,
            return_metadata=True,
            aggressive=bool(aggressive),
            **kwargs,
        )
    return lambda *args, **kwargs: improved_searching_algo(
        *args,
        adaptive_pts=True,
        adaptive_runtime_state=runtime_state,
        return_metadata=True,
        aggressive=bool(aggressive),
        **kwargs,
    )


def _build_hu_pts_only_algo_fn(*, aggressive: bool = False):
    """Build the compare wrapper for the public `PTS` baseline.

    The wrapper calls `hu_pts_only_search(...)` with metadata enabled so compare
    CSVs can report the PTS primitive with the same timing and diagnostic schema
    used by BandPilot.
    """

    return lambda *args, **kwargs: hu_pts_only_search(
        *args,
        return_metadata=True,
        aggressive=bool(aggressive),
        **kwargs,
    )


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


def _resolve_compare_baseline_config(
    baseline_config: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    """Resolve optional compare-baseline settings with stable defaults."""

    raw_config = dict(baseline_config or {})
    return {
        "network_locality_penalty_weight": float(raw_config.get("network_locality_penalty_weight", 0.35)),
        "cascore_shortlist_limit": int(raw_config.get("cascore_shortlist_limit", 12)),
        "cascore_extra_node_slack": int(raw_config.get("cascore_extra_node_slack", 1)),
        "bw_greedy_penalty_weight": float(raw_config.get("bw_greedy_penalty_weight", 0.15)),
        "linear_bw_model_root": raw_config.get(
            "linear_bw_model_root",
            "./evaluation/baselines/artifacts/models",
        ),
        "linear_bw_model_path": raw_config.get("linear_bw_model_path"),
    }


def _resolve_compare_linear_bw_model_path(
    *,
    baseline_config: Mapping[str, object],
    cluster_type: str,
    artifact_dir: Path,
) -> Path:
    """Resolve the `LinearBW` checkpoint path for the compare baseline."""

    explicit_path = baseline_config.get("linear_bw_model_path")
    if explicit_path:
        return Path(str(explicit_path))

    model_root = baseline_config.get("linear_bw_model_root")
    if not model_root:
        raise FileNotFoundError(
            "LinearBW requires `evaluation.single_contention.baselines.linear_bw_model_root` "
            "or an explicit `linear_bw_model_path`."
        )

    inferred_num_train_samples: Optional[int]
    try:
        inferred_num_train_samples = int(read_active_num_train_samples(Path(artifact_dir)))
    except (FileNotFoundError, ValueError):
        inferred_num_train_samples = None

    return resolve_linear_bw_model_path(
        cluster_type=cluster_type,
        model_root=Path(str(model_root)),
        num_train_samples=inferred_num_train_samples,
    )


def _build_linear_bw_runtime_context(
    runtime_context: SingleContentionRuntimeContext,
    *,
    baseline_config: Mapping[str, object],
) -> SingleContentionRuntimeContext:
    """Return a runtime context whose predictor is the `LinearBW` baseline."""

    model_path = _resolve_compare_linear_bw_model_path(
        baseline_config=baseline_config,
        cluster_type=runtime_context.cluster_type,
        artifact_dir=runtime_context.artifact_dir,
    )
    linear_model, linear_artifact_dir = load_linear_bw_model(
        model_path=model_path,
        device=runtime_context.device,
    )
    return replace(
        runtime_context,
        model=linear_model,
        model_path=model_path,
        artifact_dir=linear_artifact_dir,
    )


def _run_plain_compare_baseline(
    *,
    runtime_context: SingleContentionRuntimeContext,
    case_context: SingleContentionCaseContext,
    real_manager: ClusterStateManager,
    algorithm: str,
    algo_fn,
    algo_args: Sequence[object],
    algo_kwargs: Mapping[str, object],
) -> Optional[Dict[str, object]]:
    """Run a plain heuristic baseline and build a compare-compatible record."""

    combo, elapsed, predict_time, contention_time = _run_and_record(
        algo_fn,
        *algo_args,
        **dict(algo_kwargs),
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


def _prepare_bandwidth_lookup_for_path(data_path: str) -> None:
    """Prewarm the lookup cache for a target data domain outside the timed region."""
    BandwidthLookupCache.ensure_loaded(Path(data_path))


def _resolve_search_if_real_data(algorithm: str, default_search_if_real_data: bool) -> Optional[bool]:
    """Return the effective search data mode for an algorithm.

    Search algorithms may use model-domain (`training_data_path`) or real-domain
    (`evaluation_data_path`) lookups. Baselines that do not depend on lookup tables
    return None so the caller can skip domain preparation.
    """
    canonical_algorithm = normalize_algorithm_label(algorithm)
    if canonical_algorithm in MODEL_DOMAIN_ALGORITHMS:
        return bool(default_search_if_real_data)
    if canonical_algorithm in REAL_DOMAIN_ALGORITHMS:
        return True
    return None


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
    _pred_kwargs = dict(
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
    predictor = create_bandwidth_predictor(**_pred_kwargs)
    predictor_batch = create_bandwidth_predictor_batch(**_pred_kwargs)
    manager = ClusterStateManager(
        total_gpu=total_gpu,
        bandwidth_predictor=predictor,
        bandwidth_predictor_batch=predictor_batch,
        contention_mode=contention_mode,
        occupancy_seed=occupancy_seed,
    )
    if np.any(background_combo):
        manager.allocate_job(job_id=BACKGROUND_JOB_ID, combo=background_combo)
    return manager


def _build_timed_manager(
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
    """Build a fresh manager for a single timed algorithm invocation.

    Even though the current latency drift is dominated by the process-level lookup
    cache, using a fresh manager keeps timing isolated from any hidden per-manager
    state that future search changes may introduce.
    """
    return _create_cluster_manager(
        use_real_data=use_real_data,
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


def _run_timed_cluster_search(
    *,
    algo_fn,
    use_real_data: bool,
    job_id: int,
    total_gpu: int,
    avail_gpu: np.ndarray,
    model,
    test_num: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    evaluation_data_path: str,
    device,
    artifact_dir: Path,
    contention_mode: str,
    background_combo: np.ndarray,
    occupancy_seed: int,
) -> Tuple[np.ndarray, float, float, float]:
    """Run one search algorithm with domain-isolated timing.

    The lookup table prewarm and fresh manager construction intentionally happen
    before `_run_and_record(...)`, so `elapsed_time` only reflects the search body.
    Final real-data evaluation is handled separately by the caller.
    """
    search_data_path = _select_data_path(use_real_data, training_data_path, evaluation_data_path)
    _prepare_bandwidth_lookup_for_path(search_data_path)
    timed_manager = _build_timed_manager(
        use_real_data=use_real_data,
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

    # `set_job_context()` must happen on the fresh manager used by this timed run;
    # otherwise adaptive/common contention ratios could inherit another algorithm's state.
    timed_manager.set_job_context(job_id)
    try:
        return _run_and_record(
            algo_fn,
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
            use_real_data,
            cluster_manager=timed_manager,
            evaluation_data_path=evaluation_data_path,
        )
    finally:
        timed_manager.clear_job_context()


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


def prepare_single_contention_runtime_context(
    *,
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
    adaptive_threshold_policy: Optional[dict] = None,
) -> SingleContentionRuntimeContext:
    """Prepare shared runtime state for a full `single_contention` stream.

    This loader validates the offline max-bandwidth cache, loads the predictor,
    builds topology maps, and resolves adaptive thresholds before any timed
    algorithm call. Keeping this setup outside the timed region makes direct
    compare runs and replay-based validation use the same sample stream.
    """

    normalized_mode = normalize_contention_mode(contention_mode)
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
    adaptive_thresholds = resolve_adaptive_thresholds(
        policy_cfg=adaptive_threshold_policy,
        switch_config=switch_config,
        cluster_manager=None,
    )
    logger.info(
        "Prepared single_contention runtime | cluster=%s | mode=%s | bucket=%s | cv=%.3f | gap=%.3f | min_candidates=%s",
        cluster_type,
        adaptive_thresholds.policy_mode,
        adaptive_thresholds.policy_bucket,
        adaptive_thresholds.cv_threshold,
        adaptive_thresholds.gap_threshold,
        adaptive_thresholds.min_candidates_for_cv,
    )

    return SingleContentionRuntimeContext(
        repeat_num=repeat_num,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        model_path=model_path,
        model_cfg=dict(model_cfg),
        cluster_type=cluster_type,
        training_data_path=training_data_path,
        evaluation_data_path=evaluation_data_path,
        bw_type=bw_type,
        artifact_dir=artifact_dir,
        if_dynamic=if_dynamic,
        random_seed=random_seed,
        contention_mode=normalized_mode,
        search_if_real_data=search_if_real_data,
        max_bw_cache_file=max_bw_cache_file,
        model=model,
        device=device,
        topo_matrix=topo_matrix,
        gpu_to_node_map=gpu_to_node_map,
        adaptive_thresholds=adaptive_thresholds,
        max_bw_cache=max_bw_cache,
    )


def iter_single_contention_case_contexts(
    runtime_context: SingleContentionRuntimeContext,
    *,
    test_num_values: Optional[Sequence[int]] = None,
    repeat_indices: Optional[Sequence[int]] = None,
) -> Iterator[SingleContentionCaseContext]:
    """Yield compare-ready `single_contention` cases in deterministic order.

    The default stream covers `k in [2, total_gpu)` for every repeat and yields
    cases in `repeat_idx -> test_num` order so runtime-adaptive banks advance by
    repeat. Optional filters are intended for smoke tests while preserving the
    same sampling, seeding, and max-bandwidth-cache validation logic.
    """

    total_gpu = int(runtime_context.total_gpu)
    repeat_num = int(runtime_context.repeat_num)
    if test_num_values is None:
        ordered_test_nums = list(range(2, total_gpu))
    else:
        ordered_test_nums = [int(value) for value in test_num_values]
    if repeat_indices is None:
        ordered_repeats = list(range(repeat_num))
    else:
        ordered_repeats = [int(value) for value in repeat_indices]

    def _build_case_context(
        *,
        test_num: int,
        repeat_idx: int,
        current_seed: Optional[int],
        avail_gpu: np.ndarray,
        occupancy_seed: int,
    ) -> Optional[SingleContentionCaseContext]:
        """Build and validate one `(repeat_idx, test_num)` case context."""

        if len(avail_gpu) < test_num:
            logger.warning(
                "Available GPUs insufficient (test_num=%s, avail=%s); skip this repeat",
                test_num,
                len(avail_gpu),
            )
            return None

        mask_gpu = sorted(set(range(total_gpu)) - set(int(idx) for idx in avail_gpu))
        background_combo = np.zeros(total_gpu, dtype=int)
        if mask_gpu:
            background_combo[mask_gpu] = 1

        probe_job_id = _build_contention_job_id(test_num, repeat_idx, 0)
        cache_key = (test_num, repeat_idx)
        cache_entry = runtime_context.max_bw_cache.get(cache_key)
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
            return None

        base_record = _build_base_record(
            test_num=test_num,
            total_gpu=total_gpu,
            repeat_idx=repeat_idx,
            bw_type=runtime_context.bw_type,
            cluster_type=runtime_context.cluster_type,
            if_dynamic=runtime_context.if_dynamic,
            seed=current_seed,
            avail_gpu=avail_gpu,
            max_bw=max_bw,
            extra={
                "contention_mode": runtime_context.contention_mode,
                "search_if_real_data_global": runtime_context.search_if_real_data,
                "background_gpu_count": len(mask_gpu),
                "background_signature": _build_combo_signature(background_combo),
                "occupancy_seed": occupancy_seed,
                "probe_job_id": probe_job_id,
                "max_bw_cache_file": str(runtime_context.max_bw_cache_file),
                "max_bw_cache_avail_signature": expected_signature,
            },
        )
        return SingleContentionCaseContext(
            test_num=test_num,
            repeat_idx=repeat_idx,
            seed_used=current_seed,
            avail_gpu=np.asarray(avail_gpu, dtype=int),
            mask_gpu=mask_gpu,
            background_combo=background_combo,
            occupancy_seed=occupancy_seed,
            probe_job_id=probe_job_id,
            max_bw=max_bw,
            base_record=base_record,
        )

    # With `random_seed=None`, sample once in the historical `test_num -> repeat`
    # order and then replay those cases in `repeat -> test_num` order. This keeps
    # old cache files compatible while still feeding runtime-adaptive banks by
    # repeat.
    precomputed_unseeded_cases: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}
    if runtime_context.random_seed is None:
        for test_num in ordered_test_nums:
            for repeat_idx in ordered_repeats:
                avail_gpu = np.asarray(
                    _sample_available_gpu(
                        total_gpu,
                        test_num,
                        runtime_context.if_dynamic,
                        random_seed=None,
                    ),
                    dtype=int,
                )
                occupancy_seed = int(np.random.randint(0, 2**31 - 1))
                precomputed_unseeded_cases[(repeat_idx, test_num)] = (
                    avail_gpu,
                    occupancy_seed,
                )

    for repeat_idx in ordered_repeats:
        print(f"===== Single-contention experiment: current repeat_idx {repeat_idx} =====")
        for test_num in ordered_test_nums:
            if runtime_context.random_seed is not None:
                current_seed = int(runtime_context.random_seed + repeat_idx)
                np.random.seed(current_seed)
                import random

                random.seed(current_seed)
                avail_gpu = np.asarray(
                    _sample_available_gpu(
                        total_gpu,
                        test_num,
                        runtime_context.if_dynamic,
                        random_seed=current_seed,
                    ),
                    dtype=int,
                )
                occupancy_seed = int(current_seed + test_num * 97)
            else:
                current_seed = None
                avail_gpu, occupancy_seed = precomputed_unseeded_cases[(repeat_idx, test_num)]

            case_context = _build_case_context(
                test_num=test_num,
                repeat_idx=repeat_idx,
                current_seed=current_seed,
                avail_gpu=np.asarray(avail_gpu, dtype=int),
                occupancy_seed=int(occupancy_seed),
            )
            if case_context is not None:
                yield case_context


def build_single_contention_real_manager(
    runtime_context: SingleContentionRuntimeContext,
    case_context: SingleContentionCaseContext,
) -> ClusterStateManager:
    """Build the untimed real-data manager used to evaluate one case.

    The helper mirrors the timed search manager's background occupancy and
    contention mode so all algorithms are re-scored against the same real-data
    contention model.
    """

    return _create_cluster_manager(
        use_real_data=True,
        total_gpu=runtime_context.total_gpu,
        gpu_bw_dict_list=runtime_context.gpu_bw_dict_list,
        switch_config=runtime_context.switch_config,
        training_data_path=runtime_context.training_data_path,
        evaluation_data_path=runtime_context.evaluation_data_path,
        model=runtime_context.model,
        device=runtime_context.device,
        artifact_dir=runtime_context.artifact_dir,
        contention_mode=runtime_context.contention_mode,
        background_combo=case_context.background_combo,
        occupancy_seed=case_context.occupancy_seed,
    )


def evaluate_single_contention_combo(
    *,
    real_manager: ClusterStateManager,
    job_id: int,
    combo: np.ndarray,
) -> Tuple[float, float, float]:
    """Evaluate one selected combo with the real-data contention manager.

    Returns the contention-aware bandwidth, standalone bandwidth, and evaluation
    contention-model time. The real manager job context is always cleared before
    returning.
    """

    combo_arr = np.asarray(combo, dtype=int)
    with contention_profiling_session() as eval_profiler:
        try:
            real_manager.set_job_context(job_id)
            final_bw = float(real_manager.predict_with_contention(combo_arr))
        finally:
            real_manager.clear_job_context()
        contention_time = eval_profiler.total_time if eval_profiler is not None else 0.0
    standalone_bw = float(real_manager.bandwidth_predictor(combo_arr))
    return final_bw, standalone_bw, contention_time


def build_single_contention_record(
    *,
    runtime_context: SingleContentionRuntimeContext,
    case_context: SingleContentionCaseContext,
    algorithm: str,
    combo,
    elapsed: float,
    predict_time: float,
    contention_time: float,
    search_if_real_data_effective: Optional[bool],
    job_id: int,
    real_manager: ClusterStateManager,
) -> Optional[Dict[str, object]]:
    """Convert one algorithm output into a compare-compatible result record.

    Invalid selections are skipped: the combo must match `total_gpu`, must not
    overlap background-occupied GPUs, and must be evaluable by the real-data
    manager.
    """

    if combo is None:
        return None

    combo_arr = np.asarray(combo, dtype=int)
    if combo_arr.size != int(runtime_context.total_gpu):
        logger.warning("%s produced an invalid-dimension combo; skipped", algorithm)
        return None

    if np.any(combo_arr == 1) and case_context.mask_gpu:
        overlap = [idx for idx in case_context.mask_gpu if combo_arr[idx] == 1]
        if overlap:
            logger.warning(
                "%s selected GPUs occupied by background tasks (sample %s); ignored",
                algorithm,
                overlap[:4],
            )
            return None

    try:
        final_bw, standalone_bw, eval_contention_time = evaluate_single_contention_combo(
            real_manager=real_manager,
            job_id=job_id,
            combo=combo_arr,
        )
    except ValueError as exc:
        logger.error("%s combo evaluation failed: %s", algorithm, exc)
        return None

    record = dict(case_context.base_record)
    final_util = float(final_bw) / case_context.max_bw * 100 if case_context.max_bw > 0 else 0.0
    standalone_util = float(standalone_bw) / case_context.max_bw * 100 if case_context.max_bw > 0 else 0.0
    record.update(
        {
            "algorithm": algorithm,
            "search_if_real_data_effective": search_if_real_data_effective,
            "final_bw": float(final_bw),
            "standalone_bw": float(standalone_bw),
            "final_utilization": final_util,
            "standalone_utilization": standalone_util,
            "elapsed_time": float(elapsed),
            "predict_time": float(predict_time),
            "contention_time": float(contention_time + eval_contention_time),
            "selected_gpu_count": int(np.sum(combo_arr)),
            "combo_signature": _build_combo_signature(combo_arr),
            "contention_job_id": job_id,
        }
    )
    return record


def run_single_contention_search_algorithm(
    *,
    runtime_context: SingleContentionRuntimeContext,
    case_context: SingleContentionCaseContext,
    algorithm: str,
    algo_fn,
    use_real_data: bool,
    real_manager: ClusterStateManager,
    job_id: Optional[int] = None,
) -> Dict[str, object]:
    """Run one compare-style search algorithm and return output plus records.

    The helper standardizes timing, contention-aware evaluation, metadata
    flattening, and record construction for direct compare runs, targeted
    legacy-BandPilot-KNN validation, and replay-style adaptive-kNN experiments.
    """

    effective_job_id = case_context.probe_job_id if job_id is None else int(job_id)
    raw_output, elapsed, predict_time, contention_time = _run_timed_cluster_search(
        algo_fn=algo_fn,
        use_real_data=bool(use_real_data),
        job_id=effective_job_id,
        total_gpu=runtime_context.total_gpu,
        avail_gpu=case_context.avail_gpu,
        model=runtime_context.model,
        test_num=case_context.test_num,
        gpu_bw_dict_list=runtime_context.gpu_bw_dict_list,
        switch_config=runtime_context.switch_config,
        training_data_path=runtime_context.training_data_path,
        evaluation_data_path=runtime_context.evaluation_data_path,
        device=runtime_context.device,
        artifact_dir=runtime_context.artifact_dir,
        contention_mode=runtime_context.contention_mode,
        background_combo=case_context.background_combo,
        occupancy_seed=case_context.occupancy_seed,
    )

    combo = raw_output
    metadata: Dict[str, object] = {}
    if (
        isinstance(raw_output, tuple)
        and len(raw_output) == 2
        and isinstance(raw_output[1], dict)
    ):
        combo = raw_output[0]
        metadata = dict(raw_output[1])

    record = build_single_contention_record(
        runtime_context=runtime_context,
        case_context=case_context,
        algorithm=algorithm,
        combo=combo,
        elapsed=elapsed,
        predict_time=predict_time,
        contention_time=contention_time,
        search_if_real_data_effective=use_real_data,
        job_id=effective_job_id,
        real_manager=real_manager,
    )
    if record is not None and metadata:
        record.update(_metadata_to_csv_safe_fields(metadata))
    return {
        "raw_output": raw_output,
        "combo": combo,
        "metadata": metadata,
        "record": record,
        "elapsed_time_s": float(elapsed),
        "predict_time_s": float(predict_time),
        "contention_time_s": float(contention_time),
    }


def _extract_record_field(
    result: Mapping[str, object],
    field_name: str,
    *,
    default: float = 0.0,
) -> float:
    """Extract a numeric field from a compare helper result record."""

    record = result.get("record")
    if isinstance(record, dict) and field_name in record:
        return float(record[field_name])
    return float(default)


def _extract_record_signature(
    result: Mapping[str, object],
    *,
    default: str = "",
) -> str:
    """Extract a combo signature from a compare helper result record."""

    record = result.get("record")
    if isinstance(record, dict):
        return str(record.get("combo_signature", default))
    return str(default)


def _extract_selected_gpu_count(
    result: Mapping[str, object],
) -> int:
    """Extract the selected-GPU count from a compare helper result record."""

    record = result.get("record")
    if isinstance(record, dict):
        return int(record.get("selected_gpu_count", 0))
    return 0


def _build_adaptive_knn_runtime_sample(
    *,
    runtime_context: SingleContentionRuntimeContext,
    case_context: SingleContentionCaseContext,
    eha_result: Mapping[str, object],
    pts_result: Mapping[str, object],
    bandpilot_result: Mapping[str, object],
) -> Dict[str, object]:
    """Build one compare-style runtime sample for `adaptive_knn` replay.

    The sample combines shared case context, EHA confidence metadata, primitive
    timings, and BandPilot outputs into the schema consumed by the stateful
    replay and later converted back into compare CSV rows.
    """

    eha_meta = dict(eha_result.get("metadata", {}))
    bw_list = [float(value) for value in list(eha_meta.get("bw_list", []))]

    return {
        "cluster_type": str(runtime_context.cluster_type),
        "policy_bucket": str(runtime_context.adaptive_thresholds.policy_bucket),
        "policy_mode": "adaptive_knn",
        "contention_mode": str(runtime_context.contention_mode),
        "total_gpu": int(runtime_context.total_gpu),
        "if_dynamic": bool(runtime_context.if_dynamic),
        "search_if_real_data": bool(runtime_context.search_if_real_data),
        "seed_used": case_context.seed_used,
        "test_num": int(case_context.test_num),
        "repeat_idx": int(case_context.repeat_idx),
        "avail_gpu_count": int(len(case_context.avail_gpu)),
        "avail_signature": str(case_context.base_record.get("avail_signature", "")),
        "background_gpu_count": int(case_context.base_record.get("background_gpu_count", 0)),
        "background_signature": str(case_context.base_record.get("background_signature", "")),
        "occupancy_seed": int(case_context.occupancy_seed),
        "probe_job_id": int(case_context.probe_job_id),
        "max_bw": float(case_context.max_bw),
        "bw_type": str(runtime_context.bw_type),
        "reference_cv_threshold": float(runtime_context.adaptive_thresholds.cv_threshold),
        "reference_gap_threshold": float(runtime_context.adaptive_thresholds.gap_threshold),
        "reference_min_candidates_for_cv": int(
            runtime_context.adaptive_thresholds.min_candidates_for_cv
        ),
        "eha_feasible": eha_result.get("record") is not None,
        "eha_node_count": int(eha_meta.get("node_count", 0)),
        "eha_min_node_density": int(eha_meta.get("min_node_density", 0)),
        "eha_num_candidates": int(eha_meta.get("num_candidates", 0)),
        "eha_bw_cv": float(eha_meta.get("bw_cv", 0.0)),
        "eha_top5_gap": float(eha_meta.get("top5_gap", 0.0)),
        "eha_best_pred_bw": float(eha_meta.get("best_bw", 0.0)),
        "eha_second_pred_bw": float(bw_list[1]) if len(bw_list) >= 2 else 0.0,
        "eha_topk_pred_bws_json": str(bw_list[:5]).replace("'", ""),
        "eha_phase2_mode": str(eha_meta.get("phase2_mode", "")),
        "eha_hierarchical_path": bool(eha_meta.get("hierarchical_path", False)),
        "eha_candidate_plan_count": int(eha_meta.get("candidate_plan_count", 0)),
        "eha_estimated_subset_calls": int(eha_meta.get("estimated_subset_calls", 0)),
        "eha_kplus1_probe_count": int(eha_meta.get("kplus1_probe_count", 0)),
        "eha_k_values_json": str(
            [int(value) for value in list(eha_meta.get("k_values", []))]
        ).replace("'", ""),
        "eha_search_latency_s": _extract_record_field(eha_result, "elapsed_time"),
        "eha_predict_time_s": _extract_record_field(eha_result, "predict_time"),
        "eha_contention_time_s": _extract_record_field(eha_result, "contention_time"),
        "eha_final_bw": _extract_record_field(eha_result, "final_bw"),
        "eha_standalone_bw": _extract_record_field(eha_result, "standalone_bw"),
        "selected_gpu_count_eha": _extract_selected_gpu_count(eha_result),
        "eha_combo_signature": _extract_record_signature(eha_result),
        "pts_search_latency_s": _extract_record_field(pts_result, "elapsed_time"),
        "pts_predict_time_s": _extract_record_field(pts_result, "predict_time"),
        "pts_contention_time_s": _extract_record_field(pts_result, "contention_time"),
        "pts_final_bw": _extract_record_field(pts_result, "final_bw"),
        "pts_standalone_bw": _extract_record_field(pts_result, "standalone_bw"),
        "selected_gpu_count_pts": _extract_selected_gpu_count(pts_result),
        "pts_combo_signature": _extract_record_signature(pts_result),
        "bandpilot_search_latency_s": _extract_record_field(
            bandpilot_result,
            "elapsed_time",
        ),
        "bandpilot_predict_time_s": _extract_record_field(
            bandpilot_result,
            "predict_time",
        ),
        "bandpilot_contention_time_s": _extract_record_field(
            bandpilot_result,
            "contention_time",
        ),
        "bandpilot_final_bw": _extract_record_field(bandpilot_result, "final_bw"),
        "bandpilot_standalone_bw": _extract_record_field(
            bandpilot_result,
            "standalone_bw",
        ),
        "selected_gpu_count_bandpilot": _extract_selected_gpu_count(bandpilot_result),
        "bandpilot_combo_signature": _extract_record_signature(bandpilot_result),
    }


def get_multi_mode_single_dispatch_with_adaptive_knn_data(
    *,
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
    contention_modes: Sequence[str],
    max_bw_cache_files: Mapping[str, Path],
    if_dynamic: bool = True,
    random_seed: Optional[int] = None,
    search_if_real_data: bool = False,
    adaptive_threshold_policy: Optional[dict] = None,
    adaptive_knn_policy: Optional[Mapping[str, object]] = None,
    hu_unit_gate: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Run multi-mode compare and emit adaptive-kNN replay rows.

    The multi-mode path mixes idle/common/intensive cases into one replay bank,
    while each direct compare branch still uses the per-mode contention manager.
    It also emits the direct current `BandPilot` result row so the public
    output remains tied to the search.py mainline path.
    """

    config = AdaptiveKNNConfig.from_mapping(adaptive_knn_policy)
    config = replace(config, algorithm_label=normalize_algorithm_label(config.algorithm_label))
    hu_aggressive = resolve_hu_unit_gate_config(hu_unit_gate).aggressive
    results: List[Dict[str, object]] = []
    samples: List[Dict[str, object]] = []

    # Run direct compare branches first, then append samples to the replay bank.
    for contention_mode in contention_modes:
        normalized_mode = normalize_contention_mode(contention_mode)
        max_bw_cache_file = max_bw_cache_files.get(normalized_mode)
        if max_bw_cache_file is None:
            raise FileNotFoundError(
                f"adaptive_knn missing max_bw cache for mode={normalized_mode}"
            )

        runtime_context = prepare_single_contention_runtime_context(
            repeat_num=repeat_num,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            model_path=model_path,
            model_cfg=model_cfg,
            cluster_type=cluster_type,
            training_data_path=training_data_path,
            evaluation_data_path=evaluation_data_path,
            bw_type=normalized_mode,
            artifact_dir=artifact_dir,
            if_dynamic=if_dynamic,
            random_seed=random_seed,
            contention_mode=normalized_mode,
            search_if_real_data=search_if_real_data,
            max_bw_cache_file=max_bw_cache_file,
            adaptive_threshold_policy=adaptive_threshold_policy,
        )
        hu_runtime_state = _build_compare_runtime_adaptive_state(
            adaptive_runtime_policy=adaptive_knn_policy,
            cluster_type=cluster_type,
            contention_mode=normalized_mode,
            bank_scope="multi_mode",
        )
        current_repeat_idx: Optional[int] = None

        for case_context in iter_single_contention_case_contexts(runtime_context):
            if hu_runtime_state is not None:
                if current_repeat_idx is None:
                    current_repeat_idx = int(case_context.repeat_idx)
                elif int(case_context.repeat_idx) != int(current_repeat_idx):
                    _log_runtime_bank_summary(
                        summary=hu_runtime_state.finish_bank(),
                        cluster_type=cluster_type,
                        contention_mode=normalized_mode,
                    )
                    current_repeat_idx = int(case_context.repeat_idx)

            real_manager = build_single_contention_real_manager(runtime_context, case_context)
            job_id = case_context.probe_job_id

            # `legacy-BandPilot` is the exact-PTS legacy path. `BandPilot` is
            # the public mainline search path backed by EHA, runtime kNN, and PTS.
            bandpilot_result = run_single_contention_search_algorithm(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm=LEGACY_BANDPILOT_LABEL,
                algo_fn=lambda *args, **kwargs: legacy_improved_searching_algo(
                    *args,
                    return_metadata=True,
                    **kwargs,
                ),
                use_real_data=bool(search_if_real_data),
                real_manager=real_manager,
                job_id=job_id,
            )
            hu_bandpilot_result = run_single_contention_search_algorithm(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm=BANDPILOT_LABEL,
                algo_fn=_build_hu_bandpilot_algo_fn(runtime_state=hu_runtime_state, aggressive=hu_aggressive),
                use_real_data=bool(search_if_real_data),
                real_manager=real_manager,
                job_id=job_id,
            )
            hu_pts_result = run_single_contention_search_algorithm(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm=PTS_LABEL,
                algo_fn=_build_hu_pts_only_algo_fn(aggressive=hu_aggressive),
                use_real_data=bool(search_if_real_data),
                real_manager=real_manager,
                job_id=job_id,
            )
            eha_result = run_single_contention_search_algorithm(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm="EHA",
                algo_fn=lambda *args, **kwargs: eha_search(
                    *args,
                    return_confidence=True,
                    **kwargs,
                ),
                use_real_data=bool(search_if_real_data),
                real_manager=real_manager,
                job_id=job_id,
            )
            tree_result = run_single_contention_search_algorithm(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm=TREE_LABEL,
                algo_fn=tree_search_only,
                use_real_data=bool(search_if_real_data),
                real_manager=real_manager,
                job_id=job_id,
            )

            for result in (
                bandpilot_result,
                hu_bandpilot_result,
                hu_pts_result,
                eha_result,
                tree_result,
            ):
                if result["record"] is not None:
                    results.append(dict(result["record"]))

            upper_result = run_single_contention_search_algorithm(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm="UpperBandPilot",
                algo_fn=legacy_improved_searching_algo,
                use_real_data=True,
                real_manager=real_manager,
                job_id=job_id,
            )
            if upper_result["record"] is not None:
                results.append(dict(upper_result["record"]))

            if config.include_legacy_adaptive_baseline:
                legacy_result = run_single_contention_search_algorithm(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=config.legacy_algorithm_label,
                    algo_fn=lambda *args, **kwargs: threshold_legacy_exact_improved_searching_algo(
                        *args,
                        cv_threshold=runtime_context.adaptive_thresholds.cv_threshold,
                        gap_threshold=runtime_context.adaptive_thresholds.gap_threshold,
                        min_candidates_for_cv=runtime_context.adaptive_thresholds.min_candidates_for_cv,
                        **kwargs,
                    ),
                    use_real_data=bool(search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if legacy_result["record"] is not None:
                    legacy_record = dict(legacy_result["record"])
                    legacy_record["algorithm"] = config.legacy_algorithm_label
                    legacy_record["adaptive_policy_name"] = "adaptive_threshold_legacy"
                    results.append(legacy_record)

            default_combo, default_elapsed, default_predict, default_contention = _run_and_record(
                default_algo,
                runtime_context.total_gpu,
                case_context.avail_gpu,
                case_context.test_num,
            )
            default_record = build_single_contention_record(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm="Default",
                combo=default_combo,
                elapsed=default_elapsed,
                predict_time=default_predict,
                contention_time=default_contention,
                search_if_real_data_effective=None,
                job_id=job_id,
                real_manager=real_manager,
            )
            if default_record is not None:
                results.append(default_record)

            topo_combo, topo_elapsed, topo_predict, topo_contention = _run_and_record(
                lambda *args, **kwargs: slurm_best_fit_algo(*args, **kwargs),
                runtime_context.total_gpu,
                case_context.avail_gpu,
                case_context.test_num,
                runtime_context.topo_matrix,
                runtime_context.gpu_to_node_map,
            )
            topo_record = build_single_contention_record(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm="Topo",
                combo=topo_combo,
                elapsed=topo_elapsed,
                predict_time=topo_predict,
                contention_time=topo_contention,
                search_if_real_data_effective=None,
                job_id=job_id,
                real_manager=real_manager,
            )
            if topo_record is not None:
                results.append(topo_record)

            random_combo, random_elapsed, random_predict, random_contention = _run_and_record(
                random_algo,
                runtime_context.total_gpu,
                case_context.avail_gpu,
                case_context.test_num,
            )
            random_record = build_single_contention_record(
                runtime_context=runtime_context,
                case_context=case_context,
                algorithm="Random",
                combo=random_combo,
                elapsed=random_elapsed,
                predict_time=random_predict,
                contention_time=random_contention,
                search_if_real_data_effective=None,
                job_id=job_id,
                real_manager=real_manager,
            )
            if random_record is not None:
                results.append(random_record)

            replay_is_mainline_bandpilot = (
                normalize_algorithm_label(config.algorithm_label) == BANDPILOT_LABEL
            )
            replay_bandpilot_result = (
                hu_bandpilot_result if replay_is_mainline_bandpilot else bandpilot_result
            )
            replay_pts_result = hu_pts_result if replay_is_mainline_bandpilot else tree_result

            # Only complete tuples can be admitted to the replay bank. For the
            # public BandPilot label, the replay sample must be based on the
            # current search.py mainline result and current PTS primitive.
            if (
                replay_bandpilot_result.get("record") is None
                or replay_pts_result.get("record") is None
                or eha_result.get("record") is None
            ):
                logger.warning(
                    "Skip adaptive_knn sample due to missing baseline record | cluster=%s | mode=%s | k=%s | repeat=%s",
                    cluster_type,
                    normalized_mode,
                    case_context.test_num,
                    case_context.repeat_idx,
                )
                continue

            samples.append(
                _build_adaptive_knn_runtime_sample(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    eha_result=eha_result,
                    pts_result=replay_pts_result,
                    bandpilot_result=replay_bandpilot_result,
                )
            )

        if hu_runtime_state is not None and current_repeat_idx is not None:
            _log_runtime_bank_summary(
                summary=hu_runtime_state.finish_bank(),
                cluster_type=cluster_type,
                contention_mode=normalized_mode,
            )

    feature_rows = build_adaptive_knn_feature_rows(samples=samples, config=config)
    replay = run_adaptive_knn_replay(
        samples=samples,
        feature_rows=feature_rows,
        config=config,
    )
    results.extend(
        build_compare_records_from_replay(
            policy_rows=replay["policy_rows"],
            config=config,
        )
    )
    activation_summary = replay.get("activation_summary", {})
    logger.info(
        "adaptive_knn replay finished | cluster=%s | activated=%s | activation_bank=%s | warmup_cases=%s | post_latency_ms=%.3f | post_unsafe_skip_pct=%.3f",
        cluster_type,
        bool(activation_summary.get("activated", False)),
        int(activation_summary.get("activation_bank_version", -1)),
        int(activation_summary.get("warmup_case_count", 0)),
        float(activation_summary.get("post_activation_mean_search_latency_ms", 0.0)),
        float(activation_summary.get("post_activation_unsafe_skip_rate_pct", 0.0)),
    )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    # Keep legacy, current, and diagnostic rows in a stable algorithm-family order.
    algo_order = {
        str(config.algorithm_label): 0,
        BANDPILOT_LABEL: 1,
        PTS_LABEL: 2,
        # LEGACY_BANDPILOT_LABEL: 3,
        "EHA": 4,
        TREE_LABEL: 5,
        "UpperBandPilot": 6,
        "Default": 8,
        "Topo": 9,
        "Random": 10,
    }
    if str(config.legacy_algorithm_label) not in algo_order:
        algo_order[str(config.legacy_algorithm_label)] = 7
    result_df["__algo_order"] = result_df["algorithm"].map(
        lambda value: algo_order.get(str(value), 99)
    )
    result_df = result_df.sort_values(
        by=["contention_mode", "test_num", "repeat_idx", "__algo_order"],
        kind="stable",
    ).drop(columns=["__algo_order"])
    return result_df.reset_index(drop=True)


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
    adaptive_threshold_policy: Optional[dict] = None,
    adaptive_runtime_policy: Optional[Mapping[str, object]] = None,
    hu_unit_gate: Optional[Mapping[str, object]] = None,
    baseline_config: Optional[Mapping[str, object]] = None,
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
        adaptive_threshold_policy: Optional adaptive trigger policy config; when unset,
            falls back to the legacy global threshold tuple.
        adaptive_runtime_policy: Optional runtime adaptive policy config. When
            `mode = adaptive_knn`, compare runs `BandPilot` through the
            `search.py` runtime adaptive path and finishes the bank per repeat.
        baseline_config: Optional config for newly promoted compare baselines.
            Supported keys:
            - `network_locality_penalty_weight`
            - `bw_greedy_penalty_weight`
            - `linear_bw_model_root`
            - `linear_bw_model_path`

    Returns:
        pandas.DataFrame of per-run results with final_bw/standalone_bw/utilization and timing fields.
    """
    hu_aggressive = resolve_hu_unit_gate_config(hu_unit_gate).aggressive
    runtime_context = prepare_single_contention_runtime_context(
        repeat_num=repeat_num,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        model_path=model_path,
        model_cfg=model_cfg,
        cluster_type=cluster_type,
        training_data_path=training_data_path,
        evaluation_data_path=evaluation_data_path,
        bw_type=bw_type,
        artifact_dir=artifact_dir,
        if_dynamic=if_dynamic,
        random_seed=random_seed,
        contention_mode=contention_mode,
        search_if_real_data=search_if_real_data,
        max_bw_cache_file=max_bw_cache_file,
        adaptive_threshold_policy=adaptive_threshold_policy,
    )
    # Direct single-mode compare labels:
    # `legacy-BandPilot` is the exact-PTS legacy path;
    # `BandPilot` is the mainline EHA + runtime kNN + PTS path;
    # `PTS` is the PTS primitive baseline;
    # `UpperBandPilot` is the real-data exact upper baseline;
    # `CasCore / BWGreedy / LinearBW` are reviewer-facing network-aware challengers.
    # Threshold and replay-adaptive variants are excluded from the public direct
    # compare order unless the caller explicitly enables those policies.
    algo_order = [
        # LEGACY_BANDPILOT_LABEL,
        BANDPILOT_LABEL,
        PTS_LABEL,
        "EHA",
        # TREE_LABEL,
        "UpperBandPilot",
        "Default",
        "Topo",
        "Random",
        CASCORE_NAME,
        "BWGreedy",
        "LinearBW",
    ]  # Timing-sensitive search algorithms.
    results = []
    resolved_baseline_config = _resolve_compare_baseline_config(baseline_config)
    pair_bw_cache: MutableMapping[Tuple[int, int], float] = {}
    linear_runtime_context = _build_linear_bw_runtime_context(
        runtime_context,
        baseline_config=resolved_baseline_config,
    )
    hu_runtime_state = _build_compare_runtime_adaptive_state(
        adaptive_runtime_policy=adaptive_runtime_policy,
        cluster_type=cluster_type,
        contention_mode=contention_mode,
        bank_scope="single_mode",
    )
    current_repeat_idx: Optional[int] = None

    for case_context in iter_single_contention_case_contexts(runtime_context):
        if hu_runtime_state is not None:
            if current_repeat_idx is None:
                current_repeat_idx = int(case_context.repeat_idx)
            elif int(case_context.repeat_idx) != int(current_repeat_idx):
                _log_runtime_bank_summary(
                    summary=hu_runtime_state.finish_bank(),
                    cluster_type=cluster_type,
                    contention_mode=contention_mode,
                )
                current_repeat_idx = int(case_context.repeat_idx)

        real_manager = build_single_contention_real_manager(runtime_context, case_context)
        for algo_idx, name in enumerate(algo_order):
            job_id = case_context.probe_job_id
            effective_search_if_real_data = _resolve_search_if_real_data(
                name,
                runtime_context.search_if_real_data,
            )

            # `legacy-BandPilot` is the old exact-PTS path.
            # `BandPilot` is the current mainline search.py path.
            if name == LEGACY_BANDPILOT_LABEL:
                result = run_single_contention_search_algorithm(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    algo_fn=legacy_improved_searching_algo,
                    use_real_data=bool(effective_search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if result["record"] is not None:
                    results.append(result["record"])

            elif name == BANDPILOT_LABEL:
                result = run_single_contention_search_algorithm(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    algo_fn=_build_hu_bandpilot_algo_fn(runtime_state=hu_runtime_state, aggressive=hu_aggressive),
                    use_real_data=bool(effective_search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if result["record"] is not None:
                    results.append(result["record"])

            elif name == PTS_LABEL:
                result = run_single_contention_search_algorithm(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    algo_fn=_build_hu_pts_only_algo_fn(aggressive=hu_aggressive),
                    use_real_data=bool(effective_search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if result["record"] is not None:
                    results.append(result["record"])

            elif name == TREE_LABEL:
                result = run_single_contention_search_algorithm(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    algo_fn=tree_search_only,
                    use_real_data=bool(effective_search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if result["record"] is not None:
                    results.append(result["record"])

            elif name == "EHA":
                result = run_single_contention_search_algorithm(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    algo_fn=eha_search,
                    use_real_data=bool(effective_search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if result["record"] is not None:
                    results.append(result["record"])

            elif name == "UpperBandPilot":
                result = run_single_contention_search_algorithm(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    algo_fn= _build_hu_bandpilot_algo_fn(runtime_state=None, aggressive=hu_aggressive),#legacy_improved_searching_algo,
                    use_real_data=bool(effective_search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if result["record"] is not None:
                    results.append(result["record"])

            elif name == "Default":
                combo, elapsed, predict_time, contention_time = _run_and_record(
                    default_algo,
                    runtime_context.total_gpu,
                    case_context.avail_gpu,
                    case_context.test_num,
                )
                record = build_single_contention_record(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    combo=combo,
                    elapsed=elapsed,
                    predict_time=predict_time,
                    contention_time=contention_time,
                    search_if_real_data_effective=None,
                    job_id=job_id,
                    real_manager=real_manager,
                )
                if record is not None:
                    results.append(record)

            elif name == "Topo":
                combo, elapsed, predict_time, contention_time = _run_and_record(
                    lambda *a, **kw: slurm_best_fit_algo(*a, **kw),
                    runtime_context.total_gpu,
                    case_context.avail_gpu,
                    case_context.test_num,
                    runtime_context.topo_matrix,
                    runtime_context.gpu_to_node_map,
                )
                record = build_single_contention_record(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    combo=combo,
                    elapsed=elapsed,
                    predict_time=predict_time,
                    contention_time=contention_time,
                    search_if_real_data_effective=None,
                    job_id=job_id,
                    real_manager=real_manager,
                )
                if record is not None:
                    results.append(record)

            elif name == "Random":
                combo, elapsed, predict_time, contention_time = _run_and_record(
                    random_algo,
                    runtime_context.total_gpu,
                    case_context.avail_gpu,
                    case_context.test_num,
                )
                record = build_single_contention_record(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    combo=combo,
                    elapsed=elapsed,
                    predict_time=predict_time,
                    contention_time=contention_time,
                    search_if_real_data_effective=None,
                    job_id=job_id,
                    real_manager=real_manager,
                )
                if record is not None:
                    results.append(record)

            elif normalize_network_baseline_name(name) == CASCORE_NAME:
                record = _run_plain_compare_baseline(
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
                        "shortlist_limit": int(resolved_baseline_config["cascore_shortlist_limit"]),
                        "extra_node_slack": int(resolved_baseline_config["cascore_extra_node_slack"]),
                        "penalty_weight": float(resolved_baseline_config["network_locality_penalty_weight"]),
                    },
                )
                if record is not None:
                    results.append(record)

            elif name == "BWGreedy":
                record = _run_plain_compare_baseline(
                    runtime_context=runtime_context,
                    case_context=case_context,
                    real_manager=real_manager,
                    algorithm=name,
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
                        "penalty_weight": float(
                            resolved_baseline_config["bw_greedy_penalty_weight"]
                        ),
                    },
                )
                if record is not None:
                    results.append(record)

            elif name == "LinearBW":
                result = run_single_contention_search_algorithm(
                    runtime_context=linear_runtime_context,
                    case_context=case_context,
                    algorithm=name,
                    algo_fn=improved_searching_algo,
                    use_real_data=bool(effective_search_if_real_data),
                    real_manager=real_manager,
                    job_id=job_id,
                )
                if result["record"] is not None:
                    results.append(result["record"])

            else:
                logger.warning("Unknown algorithm: %s (idx=%s), skipping", name, algo_idx)

    if hu_runtime_state is not None and current_repeat_idx is not None:
        _log_runtime_bank_summary(
            summary=hu_runtime_state.finish_bank(),
            cluster_type=cluster_type,
            contention_mode=contention_mode,
        )

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
