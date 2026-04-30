"""Reviewer-facing scalability-latency benchmark suite.

The suite combines measured 32-GPU dispatch latency, simulated 64-1024 GPU
scaled traces, and synthesized 2048-4096 GPU control-plane bounds. Raw CSVs
retain the configured algorithm set, while public summaries focus on `EHA`,
`PTS`, and `BandPilot`.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from algorithms.eha import eha_search
from algorithms.hu_unit_gate import normalize_hu_unit_gate_config
from algorithms.runtime_adaptive import RuntimeAdaptiveKNNConfig, RuntimeAdaptiveKNNState
from algorithms.search import hu_pts_only_search, improved_searching_algo
from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values
from core.cluster_state import (
    ClusterStateManager,
    create_bandwidth_predictor,
    create_bandwidth_predictor_batch,
    contention_profiling_session,
    normalize_contention_mode,
)
from evaluation.compare import _load_predictor
from training.evaluator import (
    prediction_profiling_session,
    predict_with_model,
    preload_prediction_artifacts,
)
from utils.helpers import ensure_directory

logger = logging.getLogger(__name__)

ALGO_ORDER = ["EHA", "PTS", "BandPilot"]
ALGORITHM_DISPLAY_NAMES = {
    "EHA": "EHA",
    "PTS": "PTS",
    "BandPilot": "BandPilot",
}
ALGORITHM_NAME_ALIASES = {
    "EHA": "EHA",
    "EHA-only": "EHA",
    "PTS": "PTS",
    "HU-" + "PTS": "PTS",
    "HU-" + "PTS-only": "PTS",
    "BandPilot": "BandPilot",
    "HU-" + "BandPilot": "BandPilot",
    "HU-" + "Adaptive": "BandPilot",
    "legacy-PTS": "legacy-PTS",
    "PTS-only": "legacy-PTS",
    "tree": "tree",
    "Tree": "tree",
}
BENCHMARK_LOG_FILENAME = "search_overhead.log"
PUBLIC_REAL_TRACE_PLOT_FILENAME = "real_latency_adaptive_32gpu.pdf"
PUBLIC_SCALED_TRACE_PLOT_FILENAME = "scaled_search_trace_adaptive.pdf"
PUBLIC_PREDICTOR_PROFILE_PLOT_FILENAME = "predictor_latency_vs_nodes.pdf"
PUBLIC_SYNTH_LATENCY_PLOT_FILENAME = "dispatch_latency_bound_adaptive.pdf"
PUBLIC_TRIGGER_RATE_PLOT_FILENAME = "adaptive_trigger_rate_vs_scale.pdf"
PUBLIC_SUMMARY_FILENAME = "scalability_latency_summary.csv"
PUBLIC_SUMMARY_TEX_FILENAME = "scalability_latency_summary.tex"
REAL_SCENARIO_KEYS = ["cluster_type", "contention_mode", "k", "repeat_idx", "seed"]
REAL_CACHE_KEY_COLUMNS = REAL_SCENARIO_KEYS + ["algorithm"]
REAL_CACHE_VALUE_COLUMNS = [
    "avail_gpu_count",
    "total_gpu",
    "elapsed_time",
    "measured_wall_time_s",
    "predict_time",
    "predictor_time_s",
    "predict_count",
    "predictor_calls",
    "contention_time",
    "contention_time_s",
    "eha_time",
    "eha_phase_time_s",
    "pts_time",
    "pts_phase_time_s",
    "non_predictor_search_time_s",
    "latency_evidence_kind",
    "bandwidth_evidence_kind",
    "final_bw",
    "pts_triggered",
    "trigger_reason",
    "bw_cv",
    "top5_gap",
    "num_candidates",
    "node_count",
    "min_node_density",
]
REAL_CACHE_COLUMNS = REAL_CACHE_KEY_COLUMNS + REAL_CACHE_VALUE_COLUMNS
_ANNOTATION_ALGORITHM_NAMES = set(ALGO_ORDER) | set(ALGORITHM_DISPLAY_NAMES.values())
TIER1_ANNOTATION_BASE_COLUMNS = {
    *_ANNOTATION_ALGORITHM_NAMES,
    *(f"{name}_ref" for name in _ANNOTATION_ALGORITHM_NAMES),
    "hu_pts_only_bw",
    "eha_only_bw",
    "bw_loss_pct",
    "hu_pts_improvement_pct",
    "trigger_correctness",
}
SCALED_SCENARIO_KEYS = [
    "total_gpu",
    "k",
    "avail_ratio",
    "contention_mode",
    "inter_pod_factor",
    "repeat_idx",
    "seed",
]
SCALED_CACHE_KEY_COLUMNS = SCALED_SCENARIO_KEYS + ["algorithm"]
MANGLED_DUPLICATE_SUFFIX_RE = re.compile(r"\.\d+$")
CLUSTER_TAG_RE = re.compile(r"[^A-Za-z0-9_-]+")
DEFAULT_PUBLIC_VIEW = {
    "public_algorithms": ["EHA", "PTS", "BandPilot"],
    "upper_bound_algorithm": None,
    "representative_contention_mode": "common",
    "representative_avail_ratio": 0.7,
    "representative_inter_pod_factor": 0.7,
    "representative_k_values": [16, 64],
    "envelope_contention_modes": ["common", "intensive"],
}


@dataclass
class ProfilingPredictor:
    """Callable wrapper that records predictor call count and time."""

    fn: Callable[[np.ndarray], float]
    call_count: int = 0
    total_time: float = 0.0

    def __call__(self, combo: np.ndarray) -> float:
        start = time.perf_counter()
        self.call_count += 1
        value = float(self.fn(np.asarray(combo, dtype=int)))
        self.total_time += time.perf_counter() - start
        return value

    def reset(self) -> None:
        self.call_count = 0
        self.total_time = 0.0


def _resolve_algorithm_names(algorithm_names: Optional[Sequence[str]]) -> List[str]:
    """Validate and normalize the algorithm subset for a benchmark tier."""
    if algorithm_names is None:
        return list(ALGO_ORDER)
    normalized_requested: List[str] = []
    for name in algorithm_names:
        canonical = _canonicalize_algorithm_name(name)
        if canonical is not None and canonical not in normalized_requested:
            normalized_requested.append(canonical)
    selected = [name for name in ALGO_ORDER if name in set(normalized_requested)]
    if not selected:
        raise ValueError(
            "algorithm_names must contain at least one of "
            f"{list(ALGORITHM_NAME_ALIASES.keys())}"
        )
    return selected


def _canonicalize_algorithm_name(name: Optional[Any]) -> Optional[str]:
    """Map legacy/public algorithm labels to the benchmark's canonical internal names."""
    if name is None:
        return None
    normalized = str(name).strip()
    if normalized == "":
        return None
    return ALGORITHM_NAME_ALIASES.get(normalized, normalized)


def _display_algorithm_name(name: Optional[Any]) -> str:
    """Return the reviewer-facing display label for one canonical algorithm name."""
    canonical = _canonicalize_algorithm_name(name)
    if canonical is None:
        return ""
    return str(ALGORITHM_DISPLAY_NAMES.get(canonical, canonical))


def _normalize_benchmark_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cached/raw benchmark frames to canonical internal labels."""
    if df.empty:
        return df.copy()
    normalized = df.copy()
    if "algorithm" in normalized.columns:
        normalized["algorithm"] = normalized["algorithm"].apply(_canonicalize_algorithm_name)
        normalized["algorithm_display_name"] = normalized["algorithm"].apply(_display_algorithm_name)
    elif "algorithm_display_name" in normalized.columns:
        normalized["algorithm_display_name"] = normalized["algorithm_display_name"].astype(str)
    if "selected_backend" in normalized.columns:
        normalized["selected_backend"] = normalized["selected_backend"].apply(_canonicalize_algorithm_name)
    if "upper_bound_algorithm" in normalized.columns:
        normalized["upper_bound_algorithm"] = normalized["upper_bound_algorithm"].apply(
            _canonicalize_algorithm_name
        )
    return normalized


def _stable_int_from_text(text: str) -> int:
    """Build a deterministic integer fingerprint for group-level seed offsets."""
    value = 0
    for char in str(text):
        value = (value * 131 + ord(char)) % 1_000_000_007
    return int(value)


def _sanitize_cluster_tag(cluster_type: str) -> str:
    """Convert cluster_type to a filesystem-friendly tag."""
    return CLUSTER_TAG_RE.sub("_", str(cluster_type)).strip("_")


def _resolve_public_view_cfg(benchmark_cfg: dict) -> Dict[str, Any]:
    """Merge user config with the fixed reviewer-facing public-view defaults."""
    public_cfg = dict(DEFAULT_PUBLIC_VIEW)
    public_cfg.update(dict(benchmark_cfg.get("public_view", {})))
    public_cfg["public_algorithms"] = _resolve_algorithm_names(public_cfg.get("public_algorithms"))
    upper_bound_algorithm = public_cfg.get("upper_bound_algorithm", DEFAULT_PUBLIC_VIEW.get("upper_bound_algorithm"))
    upper_bound_algorithm = _canonicalize_algorithm_name(upper_bound_algorithm)
    if upper_bound_algorithm is not None and upper_bound_algorithm not in ALGO_ORDER:
        raise ValueError(f"upper_bound_algorithm must be one of {ALGO_ORDER}, got {upper_bound_algorithm!r}")
    public_cfg["upper_bound_algorithm"] = upper_bound_algorithm
    public_cfg["representative_contention_mode"] = normalize_contention_mode(
        public_cfg.get("representative_contention_mode", DEFAULT_PUBLIC_VIEW["representative_contention_mode"])
    )
    public_cfg["representative_avail_ratio"] = float(
        public_cfg.get("representative_avail_ratio", DEFAULT_PUBLIC_VIEW["representative_avail_ratio"])
    )
    public_cfg["representative_inter_pod_factor"] = float(
        public_cfg.get(
            "representative_inter_pod_factor",
            DEFAULT_PUBLIC_VIEW["representative_inter_pod_factor"],
        )
    )
    public_cfg["representative_k_values"] = [
        int(value) for value in public_cfg.get("representative_k_values", DEFAULT_PUBLIC_VIEW["representative_k_values"])
    ]
    public_cfg["envelope_contention_modes"] = [
        normalize_contention_mode(value)
        for value in public_cfg.get(
            "envelope_contention_modes",
            DEFAULT_PUBLIC_VIEW["envelope_contention_modes"],
        )
    ]
    return public_cfg


def _filter_algorithm_rows(df: pd.DataFrame, algorithm_names: Optional[Sequence[str]]) -> pd.DataFrame:
    """Filter a dataframe down to the requested algorithm subset."""
    if df.empty or "algorithm" not in df.columns or algorithm_names is None:
        return df.copy()
    allowed = {
        _canonicalize_algorithm_name(name)
        for name in algorithm_names
        if _canonicalize_algorithm_name(name) is not None
    }
    if not allowed:
        return df.iloc[0:0].copy()
    return df[df["algorithm"].isin(allowed)].copy()


def _match_float_series(series: pd.Series, target: float) -> pd.Series:
    """Robust equality check for float-valued config columns."""
    return np.isclose(series.astype(float), float(target), atol=1e-8)


def _build_cluster_output_path(output_dir: Path, prefix: str, cluster_type: str) -> Path:
    """Return a per-cluster artifact path under the benchmark output directory."""
    return output_dir / f"{prefix}_{_sanitize_cluster_tag(cluster_type)}.csv"


def _configure_benchmark_logger(output_dir: Path, level: str = "INFO") -> Path:
    """Create a dedicated stdout + file logger for search-overhead experiments."""
    ensure_directory(output_dir)
    log_path = output_dir / BENCHMARK_LOG_FILENAME
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("Search-overhead benchmark logging initialized: %s", log_path)
    return log_path


def _clear_benchmark_artifacts(output_dir: Path) -> List[Path]:
    """Remove cached CSV/figure/log artifacts for a clean rerun."""
    removed: List[Path] = []
    # Only clear generated benchmark artifacts under the configured output
    # directory; source files and configs are never touched.
    for pattern in ("*.csv", "*.pdf", "*.tex", "*.log"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def _load_cached_dataframe(path: Path) -> pd.DataFrame:
    """Load a cached CSV if it exists, otherwise return an empty frame."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return _normalize_benchmark_dataframe(pd.read_csv(path))


def _canonicalize_mangled_column(column: Any) -> Any:
    """Map pandas-mangled duplicate names like `foo.1` back to `foo`."""
    if not isinstance(column, str):
        return column
    return MANGLED_DUPLICATE_SUFFIX_RE.sub("", column)


def _ensure_unique_columns(df: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    """Drop duplicate column labels while keeping the newest copy."""
    if df.empty or df.columns.is_unique:
        return df.copy()
    duplicate_columns = df.columns[df.columns.duplicated(keep=False)].tolist()
    logger.warning(
        "%s contains duplicate columns; keeping last occurrence | duplicates=%s",
        frame_name,
        duplicate_columns,
    )
    return df.loc[:, ~df.columns.duplicated(keep="last")].copy()


def _prepare_tier1_cache_dataframe(df: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    """Normalize Tier 1 cache frames to raw benchmark records only."""
    if df.empty:
        return df.copy()
    prepared = _ensure_unique_columns(df, frame_name)
    derived_columns = [
        column
        for column in prepared.columns
        if _canonicalize_mangled_column(column) in TIER1_ANNOTATION_BASE_COLUMNS
    ]
    if derived_columns:
        logger.info(
            "%s contains derived Tier 1 columns; dropping before cache merge | columns=%s",
            frame_name,
            derived_columns,
        )
        prepared = prepared.drop(columns=derived_columns)
    existing_columns = [column for column in REAL_CACHE_COLUMNS if column in prepared.columns]
    missing_key_columns = [column for column in REAL_CACHE_KEY_COLUMNS if column not in prepared.columns]
    if missing_key_columns:
        logger.warning(
            "%s is missing Tier 1 key columns after normalization | missing=%s",
            frame_name,
            missing_key_columns,
        )
    extra_columns = [column for column in prepared.columns if column not in existing_columns]
    if extra_columns:
        logger.info(
            "%s contains extra Tier 1 columns; preserving them after raw schema columns | columns=%s",
            frame_name,
            extra_columns,
        )
    return prepared.loc[:, existing_columns + extra_columns].copy()


def _build_completed_key_set(df: pd.DataFrame, key_columns: Sequence[str]) -> set[Tuple[Any, ...]]:
    """Build the set of finished scenario keys from a cached dataframe."""
    if df.empty:
        return set()
    available_columns = [column for column in key_columns if column in df.columns]
    if len(available_columns) != len(key_columns):
        return set()
    keys_df = df[available_columns].drop_duplicates()
    return set(tuple(row) for row in keys_df.itertuples(index=False, name=None))


def _upsert_dataframe(
    cached_df: pd.DataFrame,
    new_records: Sequence[Dict[str, Any]],
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """Merge cached + new records, keeping the newest row per scenario key."""
    if not new_records:
        return cached_df.copy() if not cached_df.empty else pd.DataFrame()
    cached_df = _ensure_unique_columns(cached_df, "cache dataframe")
    new_df = _ensure_unique_columns(pd.DataFrame(new_records), "new records dataframe")
    combined = pd.concat([cached_df, new_df], ignore_index=True, sort=False)
    if key_columns:
        available_columns = [column for column in key_columns if column in combined.columns]
        if len(available_columns) == len(key_columns):
            combined = combined.drop_duplicates(subset=available_columns, keep="last")
    return combined


def _persist_tier1_cache(path: Path, cached_df: pd.DataFrame, new_records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Persist Tier 1 cache as canonical raw records only."""
    prepared_cached_df = _prepare_tier1_cache_dataframe(cached_df, f"Tier 1 cache [{path.name}]")
    merged = _upsert_dataframe(prepared_cached_df, new_records, REAL_CACHE_KEY_COLUMNS)
    if merged.empty:
        return merged
    merged = merged.sort_values(REAL_CACHE_KEY_COLUMNS).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def _persist_tier2_cache(
    path: Path,
    cached_df: pd.DataFrame,
    new_records: Sequence[Dict[str, Any]],
    inference_profile: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Persist Tier 2 cache and refresh derived latency estimates."""
    merged = _upsert_dataframe(cached_df, new_records, SCALED_CACHE_KEY_COLUMNS)
    if merged.empty:
        return merged
    if inference_profile is not None and not inference_profile.empty:
        merged = _attach_scaled_latency_estimate(merged, inference_profile)
    merged = merged.sort_values(SCALED_CACHE_KEY_COLUMNS).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def _persist_inference_profile_cache(
    path: Path,
    cached_df: pd.DataFrame,
    new_records: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """Persist inference-scaling cache keyed by node_count/device."""
    merged = _upsert_dataframe(cached_df, new_records, ["node_count", "device"])
    if merged.empty:
        return merged
    merged = merged.sort_values(["node_count"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def _sample_available_gpu(
    total_gpu: int,
    gpu_need: int,
    if_dynamic: bool,
    seed: Optional[int],
) -> np.ndarray:
    """Sample the currently available GPU indices."""
    rng = np.random.default_rng(seed)
    if not if_dynamic:
        return np.arange(total_gpu, dtype=int)
    avail_gpu_num = int(rng.integers(gpu_need, total_gpu + 1))
    return np.sort(rng.choice(total_gpu, avail_gpu_num, replace=False).astype(int))


def _build_background_combo(total_gpu: int, avail_gpu: Sequence[int]) -> np.ndarray:
    """Build a background occupancy mask from unavailable GPUs."""
    combo = np.zeros(total_gpu, dtype=int)
    occupied = sorted(set(range(total_gpu)) - set(int(idx) for idx in avail_gpu))
    if occupied:
        combo[occupied] = 1
    return combo


def _combo_signature(combo: Optional[np.ndarray]) -> str:
    """Serialize one selected combo into a stable GPU-index signature."""

    if combo is None:
        return ""
    combo_arr = np.asarray(combo, dtype=int)
    return ",".join(str(int(idx)) for idx in np.where(combo_arr == 1)[0].tolist())


def _index_signature(indices: Sequence[int]) -> str:
    """Serialize an index list so caches / logs can trace one concrete case."""

    return ",".join(str(int(idx)) for idx in list(indices))


def _build_bank_round_seed(base_seed: int, scenario_group_id: str, bank_round_idx: int) -> int:
    """Derive a deterministic per-group, per-round sampling seed."""
    return int(base_seed + _stable_int_from_text(scenario_group_id) % 100_000 + int(bank_round_idx))


def _build_tier1_group_specs(
    *,
    cluster_type: str,
    contention_modes: Sequence[str],
    repeat_num: int,
) -> List[Dict[str, Any]]:
    """Build Tier 1 bank groups keyed by contention mode."""

    group_specs: List[Dict[str, Any]] = []
    for contention_mode in contention_modes:
        normalized_mode = normalize_contention_mode(contention_mode)
        group_specs.append(
            {
                "scenario_group_id": f"{cluster_type}:tier1:mode_{normalized_mode}",
                "bank_scope": "tier1_contention_mode",
                "contention_mode": normalized_mode,
                "target_round_num": int(repeat_num),
            }
        )
    return group_specs


def _build_tier2_group_specs(
    *,
    cluster_type: str,
    gpu_counts: Sequence[int],
    k_values: Sequence[int],
    avail_ratios: Sequence[float],
    contention_modes: Sequence[str],
    inter_pod_factors: Sequence[float],
    repeat_num: int,
    public_view_cfg: Mapping[str, Any],
    public_repeat_num: Optional[int],
) -> List[Dict[str, Any]]:
    """Build Tier 2 bank groups keyed by scale, availability, factor, and contention mode."""

    group_specs: List[Dict[str, Any]] = []
    representative_ks = {int(value) for value in public_view_cfg["representative_k_values"]}
    max_k = max(int(value) for value in k_values)

    for total_gpu in gpu_counts:
        for inter_pod_factor in inter_pod_factors:
            for avail_ratio in avail_ratios:
                target_avail = max(max_k, int(round(int(total_gpu) * float(avail_ratio))))
                for contention_mode in contention_modes:
                    normalized_mode = normalize_contention_mode(contention_mode)
                    # Tier 2 public scenarios require one exact-fit
                    # `k == target_avail` case such as 128 GPUs at
                    # avail_ratio=0.5 with k=64.
                    feasible_ks = [int(value) for value in k_values if int(value) <= int(target_avail)]
                    if not feasible_ks:
                        continue
                    has_public_slice = any(
                        normalized_mode == public_view_cfg["representative_contention_mode"]
                        and abs(float(avail_ratio) - float(public_view_cfg["representative_avail_ratio"])) < 1e-8
                        and abs(float(inter_pod_factor) - float(public_view_cfg["representative_inter_pod_factor"])) < 1e-8
                        and int(k) in representative_ks
                        for k in feasible_ks
                    )
                    target_round_num = int(repeat_num)
                    if public_repeat_num is not None and has_public_slice:
                        target_round_num = max(int(repeat_num), int(public_repeat_num))
                    group_specs.append(
                        {
                            "scenario_group_id": (
                                f"{cluster_type}:tier2:g{int(total_gpu)}:"
                                f"a{float(avail_ratio):.2f}:f{float(inter_pod_factor):.2f}:m{normalized_mode}"
                            ),
                            "bank_scope": "tier2_scale_avail_factor_mode",
                            "total_gpu": int(total_gpu),
                            "inter_pod_factor": float(inter_pod_factor),
                            "avail_ratio": float(avail_ratio),
                            "target_avail": int(target_avail),
                            "contention_mode": normalized_mode,
                            "k_values": feasible_ks,
                            "target_round_num": int(target_round_num),
                            "has_public_slice": bool(has_public_slice),
                        }
                    )
    return group_specs


def _build_job_id(k: int, repeat_idx: int, algo_offset: int = 0) -> int:
    """Build a stable job id so common-mode occupancy stays repeatable."""
    return int(k * 10_000 + repeat_idx * 100 + algo_offset)


def _evaluate_combo_with_manager(
    manager: ClusterStateManager,
    combo: Optional[np.ndarray],
    job_id: int,
) -> float:
    """Evaluate final bandwidth under a real/scaled contention manager."""
    if combo is None:
        return 0.0
    try:
        manager.set_job_context(job_id)
        return float(manager.predict_with_contention(np.asarray(combo, dtype=int)))
    finally:
        manager.clear_job_context()


def _create_cluster_manager(
    *,
    total_gpu: int,
    predictor: Callable[[np.ndarray], float],
    contention_mode: str,
    background_combo: np.ndarray,
    occupancy_seed: int,
    predictor_batch: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> ClusterStateManager:
    """Create a contention-aware cluster manager with a fixed background load."""
    manager = ClusterStateManager(
        total_gpu=total_gpu,
        bandwidth_predictor=predictor,
        bandwidth_predictor_batch=predictor_batch,
        contention_mode=normalize_contention_mode(contention_mode),
        occupancy_seed=occupancy_seed,
    )
    if np.any(background_combo):
        manager.allocate_job(job_id=-1, combo=background_combo)
    return manager


def _run_profiled_search(run_fn: Callable[[], Any]) -> Tuple[Any, float, float, int, float]:
    """Execute a search function with latency profiling."""
    with prediction_profiling_session() as pred_profiler, contention_profiling_session() as contention_profiler:
        start = time.perf_counter()
        result = run_fn()
        elapsed = time.perf_counter() - start
        predict_time = pred_profiler.total_time if pred_profiler is not None else 0.0
        predict_count = pred_profiler.call_count if pred_profiler is not None else 0
        contention_time = contention_profiler.total_time if contention_profiler is not None else 0.0
    return result, elapsed, predict_time, predict_count, contention_time


def _build_latency_fields(
    *,
    measured_wall_time_s: float,
    predictor_time_s: float,
    predictor_calls: int,
    contention_time_s: float,
    eha_phase_time_s: float,
    pts_phase_time_s: float,
    latency_evidence_kind: str,
    bandwidth_evidence_kind: str,
    evidence_type: str,
) -> Dict[str, Any]:
    """Build a normalized latency payload while preserving legacy column names."""
    non_predictor_search_time_s = max(
        0.0,
        float(measured_wall_time_s) - float(predictor_time_s),
    )
    return {
        "elapsed_time": float(measured_wall_time_s),
        "measured_wall_time_s": float(measured_wall_time_s),
        "predict_time": float(predictor_time_s),
        "predictor_time_s": float(predictor_time_s),
        "predict_count": int(predictor_calls),
        "predictor_calls": int(predictor_calls),
        "contention_time": float(contention_time_s),
        "contention_time_s": float(contention_time_s),
        "eha_time": float(eha_phase_time_s),
        "eha_phase_time_s": float(eha_phase_time_s),
        "pts_time": float(pts_phase_time_s),
        "pts_phase_time_s": float(pts_phase_time_s),
        "non_predictor_search_time_s": float(non_predictor_search_time_s),
        "latency_evidence_kind": latency_evidence_kind,
        "bandwidth_evidence_kind": bandwidth_evidence_kind,
        "evidence_type": str(evidence_type),
    }


def _safe_pct_loss(reference: float, current: float) -> float:
    """Return percentage loss against a reference bandwidth."""
    if reference <= 0:
        return 0.0
    return max(0.0, (reference - current) / reference * 100.0)


def _compute_pod_stats(combo: Optional[np.ndarray], total_gpu: int, pod_size: int) -> Tuple[int, float]:
    """Compute active pod count and cross-pod ratio for a selected combo."""
    if combo is None:
        return 0, 0.0
    config = np.asarray(combo, dtype=int)
    gpus_per_pod = pod_size * 8
    total_selected = int(config.sum())
    if total_selected <= 0:
        return 0, 0.0
    pod_counts = np.array(
        [
            int(config[start : start + gpus_per_pod].sum())
            for start in range(0, total_gpu, gpus_per_pod)
        ],
        dtype=int,
    )
    active_pods = pod_counts[pod_counts > 0]
    if active_pods.size == 0:
        return 0, 0.0
    densest_pod = int(active_pods.max())
    cross_pod_ratio = max(0.0, (total_selected - densest_pod) / total_selected)
    return int(active_pods.size), float(cross_pod_ratio)


def _extract_eha_meta_record_fields(eha_meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract EHA confidence metadata into benchmark-row fields.

    BandPilot uses these confidence features for runtime adaptation. Persisting
    them in raw CSV rows keeps activation and trigger decisions auditable.
    """

    meta = dict(eha_meta or {})
    return {
        "eha_phase2_mode": str(meta.get("phase2_mode", "")),
        "eha_hierarchical_path": bool(meta.get("hierarchical_path", False)),
        "eha_candidate_plan_count": int(meta.get("candidate_plan_count", 0)),
        "eha_kplus1_probe_count": int(meta.get("kplus1_probe_count", 0)),
        "eha_k_values_json": meta.get("k_values_json", "[]"),
        "eha_topk_pred_bws_json": meta.get("topk_pred_bws_json", "[]"),
        "eha_best_pred_bw": float(meta.get("best_pred_bw", 0.0)),
        "eha_second_pred_bw": float(meta.get("second_pred_bw", 0.0)),
    }


def _normalize_runtime_adaptive_policy(
    adaptive_runtime_policy: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize scalability config for `search.py` runtime adaptation.

    Older benchmark-local configs used `bw_improvement_threshold_pct_of_hu_pts`;
    the shared runtime state expects `bw_improvement_threshold_pct_of_bandpilot`.
    """

    if adaptive_runtime_policy is None:
        return None
    normalized = dict(adaptive_runtime_policy)
    if (
        "bw_improvement_threshold_pct_of_bandpilot" not in normalized
        and "bw_improvement_threshold_pct_of_hu_pts" in normalized
    ):
        normalized["bw_improvement_threshold_pct_of_bandpilot"] = float(
            normalized["bw_improvement_threshold_pct_of_hu_pts"]
        )
    return normalized


def _log_runtime_bank_summary(
    *,
    stage_name: str,
    cluster_type: str,
    repeat_idx: int,
    summary: Mapping[str, Any],
) -> None:
    """Log the final state of one runtime-adaptive bank."""

    logger.info(
        "%s runtime bank finished | cluster=%s | round=%s | bank=%s | version=%s | "
        "active_before=%s | active_next=%s | labeled=%s | unlabeled=%s | admitted=%s | "
        "unsafe_skip=%.2f | over_trigger=%.2f | support_insufficient=%s",
        stage_name,
        cluster_type,
        int(repeat_idx),
        str(summary.get("bank_id", "")),
        int(summary.get("bank_version", -1)),
        bool(summary.get("bank_active_before", False)),
        bool(summary.get("bank_active_next", False)),
        int(summary.get("bank_size_labeled", 0)),
        int(summary.get("bank_size_unlabeled_skips", 0)),
        int(summary.get("admitted_count", 0)),
        float(summary.get("shadow_unsafe_skip_rate_pct", 0.0)),
        float(summary.get("shadow_over_trigger_rate_pct", 0.0)),
        int(summary.get("shadow_support_insufficient_case_count", 0)),
    )


def _resolve_mainline_selected_backend(
    *,
    pts_triggered: bool,
    final_combo: Optional[np.ndarray],
    eha_combo: Optional[np.ndarray],
    hu_combo: Optional[np.ndarray],
) -> str:
    """Resolve which backend produced the final `BandPilot` combo."""

    if not pts_triggered:
        return "EHA"
    final_signature = _combo_signature(final_combo)
    if final_signature and final_signature == _combo_signature(hu_combo):
        return "PTS"
    if final_signature and final_signature == _combo_signature(eha_combo):
        return "EHA"
    if final_combo is None:
        return ""
    logger.warning(
        "Unable to resolve BandPilot final backend from combo signatures | final=%s | eha=%s | pts=%s",
        final_signature,
        _combo_signature(eha_combo),
        _combo_signature(hu_combo),
    )
    return "unknown_final_selection"


def _build_mainline_adaptive_search_meta(
    *,
    adaptive_meta: Mapping[str, Any],
    eha_meta: Mapping[str, Any],
    selected_backend: str,
) -> Dict[str, Any]:
    """Map `search.py` adaptive metadata into the benchmark row schema."""

    pts_triggered = bool(adaptive_meta.get("pts_triggered", False))
    trigger_reason = str(adaptive_meta.get("trigger_reason", ""))
    return {
        "pts_triggered": pts_triggered,
        "trigger_reason": trigger_reason,
        "eha_meta": dict(eha_meta or {}),
        "eha_time": float(adaptive_meta.get("eha_time", 0.0)),
        "pts_time": float(adaptive_meta.get("pts_time", 0.0)),
        "pts_policy": str(adaptive_meta.get("pts_policy", "")),
        "adaptive_policy_name": str(adaptive_meta.get("adaptive_policy_name", "")),
        "selected_backend": str(_canonicalize_algorithm_name(selected_backend) or selected_backend),
        "online_bank_id": str(adaptive_meta.get("adaptive_bank_id", "")),
        "online_case_index": adaptive_meta.get("adaptive_case_index", np.nan),
        "online_support": adaptive_meta.get("adaptive_online_support", np.nan),
        "online_risk": adaptive_meta.get("adaptive_online_risk", np.nan),
        "adaptive_fallback_reason": trigger_reason if pts_triggered else "",
        "decision_overhead_ms": float(adaptive_meta.get("adaptive_decision_overhead_ms", 0.0)),
        "hu_pts_usage_rate": 1.0 if pts_triggered else 0.0,
        "support_insufficient": adaptive_meta.get("adaptive_support_insufficient", np.nan),
        "online_low_trust": adaptive_meta.get("adaptive_online_low_trust", np.nan),
        "adaptive_bank_phase": str(adaptive_meta.get("adaptive_bank_phase", "")),
        "adaptive_bank_version": adaptive_meta.get("adaptive_bank_version", np.nan),
        "adaptive_bank_active_before": adaptive_meta.get("adaptive_bank_active_before", np.nan),
        "adaptive_train_size_before": adaptive_meta.get("adaptive_train_size_before", np.nan),
        "adaptive_shadow_trigger_reason": str(
            adaptive_meta.get("adaptive_shadow_trigger_reason", "")
        ),
        "adaptive_shadow_trigger_pts": adaptive_meta.get("adaptive_shadow_trigger_pts", np.nan),
    }


def _build_backend_record(
    *,
    case_context: Mapping[str, Any],
    algorithm: str,
    final_bw: float,
    search_meta: Mapping[str, Any],
    combo: Optional[np.ndarray],
    measured_wall_time_s: float,
    predictor_time_s: float,
    predictor_calls: int,
    contention_time_s: float,
    latency_evidence_kind: str,
    bandwidth_evidence_kind: str,
    evidence_type: str,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one benchmark row from a primitive backend result."""

    canonical_algorithm = _canonicalize_algorithm_name(algorithm) or str(algorithm)
    eha_meta = dict(search_meta.get("eha_meta", {}) or {})
    record = {
        **dict(case_context),
        "algorithm": str(canonical_algorithm),
        "algorithm_display_name": _display_algorithm_name(canonical_algorithm),
        "final_bw": float(final_bw),
        "standalone_bw": float(final_bw),
        "combo_signature": _combo_signature(combo),
        "pts_triggered": search_meta.get("pts_triggered"),
        "trigger_reason": str(search_meta.get("trigger_reason", "")),
        "bw_cv": float(eha_meta.get("bw_cv", np.nan)) if eha_meta else np.nan,
        "top5_gap": float(eha_meta.get("top5_gap", np.nan)) if eha_meta else np.nan,
        "num_candidates": float(eha_meta.get("num_candidates", np.nan)) if eha_meta else np.nan,
        "node_count": float(eha_meta.get("node_count", np.nan)) if eha_meta else np.nan,
        "min_node_density": float(eha_meta.get("min_node_density", np.nan)) if eha_meta else np.nan,
        "pts_policy": str(search_meta.get("pts_policy", "")),
        "hu_aggressive": bool(search_meta.get("hu_aggressive", False)),
        "hu_unit_sizes": ",".join(str(int(value)) for value in list(search_meta.get("hu_unit_sizes", []) or [])),
        "adaptive_policy_name": str(search_meta.get("adaptive_policy_name", "")),
        "selected_backend": str(
            _canonicalize_algorithm_name(search_meta.get("selected_backend", canonical_algorithm))
            or canonical_algorithm
        ),
        "selected_backend_display_name": _display_algorithm_name(
            search_meta.get("selected_backend", canonical_algorithm)
        ),
        "online_bank_id": str(search_meta.get("online_bank_id", "")),
        "online_case_index": search_meta.get("online_case_index", np.nan),
        "online_support": search_meta.get("online_support", np.nan),
        "online_risk": search_meta.get("online_risk", np.nan),
        "adaptive_bank_phase": str(search_meta.get("adaptive_bank_phase", "")),
        "adaptive_bank_version": search_meta.get("adaptive_bank_version", np.nan),
        "adaptive_bank_active_before": search_meta.get("adaptive_bank_active_before", np.nan),
        "adaptive_train_size_before": search_meta.get("adaptive_train_size_before", np.nan),
        "adaptive_shadow_trigger_reason": str(
            search_meta.get("adaptive_shadow_trigger_reason", "")
        ),
        "adaptive_shadow_trigger_pts": search_meta.get("adaptive_shadow_trigger_pts", np.nan),
        "adaptive_fallback_reason": str(search_meta.get("adaptive_fallback_reason", "")),
        "decision_overhead_ms": float(search_meta.get("decision_overhead_ms", 0.0)),
        "hu_pts_usage_rate": float(search_meta.get("hu_pts_usage_rate", np.nan)),
        "support_insufficient": search_meta.get("support_insufficient", np.nan),
        "online_low_trust": search_meta.get("online_low_trust", np.nan),
        "avail_signature": str(case_context.get("avail_signature", "")),
        "background_signature": str(case_context.get("background_signature", "")),
    }
    record.update(_extract_eha_meta_record_fields(eha_meta))
    record.update(
        _build_latency_fields(
            measured_wall_time_s=float(measured_wall_time_s),
            predictor_time_s=float(predictor_time_s),
            predictor_calls=int(predictor_calls),
            contention_time_s=float(contention_time_s),
            eha_phase_time_s=float(search_meta.get("eha_time", 0.0)),
            pts_phase_time_s=float(search_meta.get("pts_time", 0.0)),
            latency_evidence_kind=latency_evidence_kind,
            bandwidth_evidence_kind=bandwidth_evidence_kind,
            evidence_type=evidence_type,
        )
    )
    if extra_fields:
        record.update(dict(extra_fields))
    return record


def make_real_cluster_config(
    *,
    cluster_type: str,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig,
    model_path: Path,
    model_cfg: dict,
    training_data_path: str,
    evaluation_data_path: str,
    artifact_dir: Path,
    device: torch.device,
    adaptive_runtime_policy: Optional[dict] = None,
    hu_unit_gate: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize the cluster information required by Tier 1 / Tier 2.

    Notes:
    - scalability benchmark      compare    threshold policy;
    -    cluster config           `RuntimeAdaptiveKNNState`   runtime policy.
    """
    return {
        "cluster_type": cluster_type,
        "total_gpu": total_gpu,
        "gpu_bw_dict_list": gpu_bw_dict_list,
        "switch_config": switch_config,
        "model_path": Path(model_path),
        "model_cfg": dict(model_cfg),
        "training_data_path": training_data_path,
        "evaluation_data_path": evaluation_data_path,
        "artifact_dir": Path(artifact_dir),
        "device": device,
        "adaptive_runtime_policy": _normalize_runtime_adaptive_policy(adaptive_runtime_policy)
        if adaptive_runtime_policy is not None
        else None,
        "hu_unit_gate": normalize_hu_unit_gate_config(hu_unit_gate),
    }


def _annotate_tier1_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate `BandPilot` quality relative to the `PTS` and `EHA` baselines."""
    if df.empty:
        return df

    pivot = (
        df.pivot_table(
            index=REAL_SCENARIO_KEYS,
            columns="algorithm",
            values="final_bw",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    merged = df.merge(pivot, on=REAL_SCENARIO_KEYS, how="left", suffixes=("", "_ref"))
    if "PTS" in merged.columns:
        merged["hu_pts_only_bw"] = merged["PTS"].astype(float)
    else:
        merged["hu_pts_only_bw"] = np.nan
    if "EHA" in merged.columns:
        merged["eha_only_bw"] = merged["EHA"].astype(float)
    else:
        merged["eha_only_bw"] = np.nan

    adaptive_mask = merged["algorithm"] == "BandPilot"
    merged.loc[adaptive_mask, "bw_loss_pct"] = merged.loc[adaptive_mask].apply(
        lambda row: np.nan
        if pd.isna(row["hu_pts_only_bw"])
        else _safe_pct_loss(float(row["hu_pts_only_bw"]), float(row["final_bw"])),
        axis=1,
    )
    merged.loc[adaptive_mask, "hu_pts_improvement_pct"] = merged.loc[adaptive_mask].apply(
        lambda row: np.nan
        if pd.isna(row["hu_pts_only_bw"]) or pd.isna(row["eha_only_bw"]) or float(row["hu_pts_only_bw"]) <= 0
        else max(
            0.0,
            (float(row["hu_pts_only_bw"]) - float(row["eha_only_bw"])) / float(row["hu_pts_only_bw"]) * 100.0,
        ),
        axis=1,
    )

    def _classify(row: pd.Series) -> str:
        improvement = float(row.get("hu_pts_improvement_pct", 0.0))
        if pd.isna(row.get("hu_pts_improvement_pct", np.nan)):
            return "not_available"
        triggered = bool(row.get("pts_triggered", False))
        if triggered and improvement > 3.0:
            return "true_positive"
        if (not triggered) and improvement <= 3.0:
            return "true_negative"
        if (not triggered) and improvement > 3.0:
            return "false_negative"
        return "false_positive"

    merged.loc[adaptive_mask, "trigger_correctness"] = merged.loc[adaptive_mask].apply(_classify, axis=1)
    return merged


def run_real_data_latency_benchmark(
    cluster_configs: List[dict],
    k_values: List[int],
    contention_modes: List[str],
    repeat_num: int,
    random_seed: int,
    if_dynamic: bool = True,
    algorithm_names: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
    resume: bool = True,
    save_every_n_records: int = 50,
) -> pd.DataFrame:
    """Run the Tier 1 benchmark with mainline runtime-adaptive `BandPilot`.

    Execution model:
    - each `repeat_idx` owns a deterministic `contention_mode -> k` case stream;
    - each `repeat_idx` owns an isolated `RuntimeAdaptiveKNNState`;
    - `BandPilot` calls
      `improved_searching_algo(..., adaptive_pts=True, adaptive_runtime_state=...)`,
      and primitive latency is replayed through the runtime state.
    """
    records: List[Dict[str, Any]] = []
    selected_algorithms = _resolve_algorithm_names(algorithm_names)
    logger.info(
        "Tier 1 start | clusters=%s | algorithms=%s | k_values=%s | contention_modes=%s | repeat_num=%s",
        [cfg["cluster_type"] for cfg in cluster_configs],
        selected_algorithms,
        k_values,
        contention_modes,
        repeat_num,
    )

    for cluster_cfg in cluster_configs:
        cluster_type = str(cluster_cfg["cluster_type"])
        cluster_cache_path = None if output_dir is None else output_dir / f"tier1_{cluster_type}.csv"
        effective_resume = bool(resume)
        if resume and "BandPilot" in selected_algorithms:
            logger.warning(
                "Tier 1 cluster=%s | BandPilot uses mainline runtime state; "
                "existing cache cannot restore bank state, disabling resume for this cluster.",
                cluster_type,
            )
            effective_resume = False
        cached_cluster_df = (
            _load_cached_dataframe(cluster_cache_path)
            if (effective_resume and cluster_cache_path is not None)
            else pd.DataFrame()
        )
        cached_cluster_df = _prepare_tier1_cache_dataframe(
            cached_cluster_df,
            f"Tier 1 loaded cache [{cluster_type}]",
        )
        completed_keys = _build_completed_key_set(cached_cluster_df, REAL_CACHE_KEY_COLUMNS)
        pending_records: List[Dict[str, Any]] = []
        logger.info(
            "Tier 1 cluster=%s | resume=%s | cached_rows=%s | completed_keys=%s",
            cluster_type,
            effective_resume,
            len(cached_cluster_df),
            len(completed_keys),
        )

        model = _load_predictor(
            cluster_cfg["model_path"],
            cluster_cfg["device"],
            cluster_cfg["model_cfg"],
        )
        total_gpu = int(cluster_cfg["total_gpu"])
        adaptive_policy = RuntimeAdaptiveKNNConfig.from_mapping(
            cluster_cfg.get("adaptive_runtime_policy")
        )
        logger.info(
            "Tier 1 cluster=%s | adaptive_policy=%s | k_neighbors=%s | min_support=%s | risk=%.3f",
            cluster_type,
            adaptive_policy.policy_name,
            adaptive_policy.k_neighbors,
            adaptive_policy.min_support,
            adaptive_policy.risk_threshold,
        )

        real_predictor = create_bandwidth_predictor(
            if_real_data=True,
            total_gpu=total_gpu,
            gpu_bw_dict_list=cluster_cfg["gpu_bw_dict_list"],
            switch_config=cluster_cfg["switch_config"],
            training_data_path=cluster_cfg["training_data_path"],
            evaluation_data_path=cluster_cfg["evaluation_data_path"],
        )
        real_predictor_batch = create_bandwidth_predictor_batch(
            if_real_data=True,
            total_gpu=total_gpu,
            gpu_bw_dict_list=cluster_cfg["gpu_bw_dict_list"],
            switch_config=cluster_cfg["switch_config"],
            training_data_path=cluster_cfg["training_data_path"],
            evaluation_data_path=cluster_cfg["evaluation_data_path"],
        )
        search_predictor = create_bandwidth_predictor(
            if_real_data=False,
            total_gpu=total_gpu,
            gpu_bw_dict_list=cluster_cfg["gpu_bw_dict_list"],
            switch_config=cluster_cfg["switch_config"],
            training_data_path=cluster_cfg["training_data_path"],
            evaluation_data_path=cluster_cfg["evaluation_data_path"],
            model=model,
            device=cluster_cfg["device"],
            artifact_dir=cluster_cfg["artifact_dir"],
        )
        search_predictor_batch = create_bandwidth_predictor_batch(
            if_real_data=False,
            total_gpu=total_gpu,
            gpu_bw_dict_list=cluster_cfg["gpu_bw_dict_list"],
            switch_config=cluster_cfg["switch_config"],
            training_data_path=cluster_cfg["training_data_path"],
            evaluation_data_path=cluster_cfg["evaluation_data_path"],
            model=model,
            device=cluster_cfg["device"],
            artifact_dir=cluster_cfg["artifact_dir"],
        )

        def _new_managers(
            *,
            normalized_mode: str,
            background_combo: np.ndarray,
            occupancy_seed: int,
        ) -> Tuple[ClusterStateManager, ClusterStateManager]:
            """Create fresh search and real managers for a primitive backend."""

            search_manager = _create_cluster_manager(
                total_gpu=total_gpu,
                predictor=search_predictor,
                predictor_batch=search_predictor_batch,
                contention_mode=normalized_mode,
                background_combo=background_combo,
                occupancy_seed=occupancy_seed,
            )
            real_manager = _create_cluster_manager(
                total_gpu=total_gpu,
                predictor=real_predictor,
                predictor_batch=real_predictor_batch,
                contention_mode=normalized_mode,
                background_combo=background_combo,
                occupancy_seed=occupancy_seed,
            )
            return search_manager, real_manager

        tier1_group_specs = _build_tier1_group_specs(
            cluster_type=cluster_type,
            contention_modes=contention_modes,
            repeat_num=repeat_num,
        )
        for group_spec in tier1_group_specs:
            normalized_mode = str(group_spec["contention_mode"])
            runtime_state = RuntimeAdaptiveKNNState(
                config=adaptive_policy,
                bank_id=str(group_spec["scenario_group_id"]),
            )
            logger.info(
                "Tier 1 cluster=%s | group=%s | bank_scope=%s | rounds=%s | k_count=%s",
                cluster_type,
                group_spec["scenario_group_id"],
                group_spec["bank_scope"],
                group_spec["target_round_num"],
                len(k_values),
            )
            for bank_round_idx in range(int(group_spec["target_round_num"])):
                seed = _build_bank_round_seed(
                    int(random_seed),
                    str(group_spec["scenario_group_id"]),
                    int(bank_round_idx),
                )
                for k in k_values:
                    case_completed = all(
                        (
                            cluster_type,
                            normalized_mode,
                            int(k),
                            int(bank_round_idx),
                            int(seed),
                            algo_name,
                        )
                        in completed_keys
                        for algo_name in selected_algorithms
                    )
                    if case_completed:
                        continue

                    avail_gpu = _sample_available_gpu(
                        total_gpu=total_gpu,
                        gpu_need=k,
                        if_dynamic=if_dynamic,
                        seed=seed,
                    )
                    if len(avail_gpu) < k:
                        continue

                    background_combo = _build_background_combo(total_gpu, avail_gpu)
                    background_gpu = np.where(background_combo == 1)[0].astype(int).tolist()
                    occupancy_seed = int(seed + k * 97)
                    probe_job_id = _build_job_id(k, int(bank_round_idx))
                    case_context = {
                        "cluster_type": cluster_type,
                        "contention_mode": normalized_mode,
                        "k": int(k),
                        "repeat_idx": int(bank_round_idx),
                        "bank_round_idx": int(bank_round_idx),
                        "seed": int(seed),
                        "avail_gpu_count": int(len(avail_gpu)),
                        "total_gpu": int(total_gpu),
                        "avail_gpu": [int(value) for value in avail_gpu.tolist()],
                        "background_gpu": background_gpu,
                        "avail_signature": _index_signature(avail_gpu.tolist()),
                        "background_signature": _index_signature(background_gpu),
                        "if_dynamic": bool(if_dynamic),
                        "search_if_real_data": False,
                        "occupancy_seed": int(occupancy_seed),
                        "probe_job_id": int(probe_job_id),
                        "scenario_group_id": str(group_spec["scenario_group_id"]),
                        "bank_scope": str(group_spec["bank_scope"]),
                    }

                    eha_search_manager, eha_real_manager = _new_managers(
                        normalized_mode=normalized_mode,
                        background_combo=background_combo,
                        occupancy_seed=occupancy_seed,
                    )

                    def _run_eha() -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
                        eha_search_manager.set_job_context(probe_job_id)
                        try:
                            combo, eha_meta = eha_search(
                                total_gpu,
                                avail_gpu,
                                model,
                                k,
                                total_gpu,
                                cluster_cfg["gpu_bw_dict_list"],
                                cluster_cfg["switch_config"],
                                cluster_cfg["training_data_path"],
                                cluster_cfg["device"],
                                cluster_cfg["artifact_dir"],
                                if_real_data=False,
                                cluster_manager=eha_search_manager,
                                evaluation_data_path=cluster_cfg["evaluation_data_path"],
                                return_confidence=True,
                            )
                        finally:
                            eha_search_manager.clear_job_context()
                        return combo, {
                            "pts_triggered": False,
                            "trigger_reason": "eha_only",
                            "eha_meta": eha_meta,
                            "eha_time": 0.0,
                            "pts_time": 0.0,
                            "pts_policy": "",
                        }

                    (eha_combo, eha_meta), eha_elapsed, eha_predict_time, eha_predict_count, eha_contention_time = (
                        _run_profiled_search(_run_eha)
                    )
                    eha_meta["eha_time"] = eha_elapsed
                    eha_final_bw = _evaluate_combo_with_manager(eha_real_manager, eha_combo, probe_job_id)
                    eha_result = {
                        "combo": eha_combo,
                        "final_bw": float(eha_final_bw),
                        "standalone_bw": float(eha_final_bw),
                        "combo_signature": _combo_signature(eha_combo),
                        "measured_wall_time_s": float(eha_elapsed),
                        "predictor_time_s": float(eha_predict_time),
                        "predictor_calls": int(eha_predict_count),
                        "contention_time_s": float(eha_contention_time),
                        "eha_phase_time_s": float(eha_elapsed),
                        "pts_phase_time_s": 0.0,
                        "eha_meta": dict(eha_meta.get("eha_meta", {}) or {}),
                    }

                    hu_search_manager, hu_real_manager = _new_managers(
                        normalized_mode=normalized_mode,
                        background_combo=background_combo,
                        occupancy_seed=occupancy_seed,
                    )
                    hu_aggressive = bool(cluster_cfg.get("hu_unit_gate", {}).get("aggressive", False))

                    def _run_hu_pts() -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
                        hu_search_manager.set_job_context(probe_job_id)
                        try:
                            return hu_pts_only_search(
                                total_gpu,
                                avail_gpu,
                                model,
                                k,
                                total_gpu,
                                cluster_cfg["gpu_bw_dict_list"],
                                cluster_cfg["switch_config"],
                                cluster_cfg["training_data_path"],
                                cluster_cfg["device"],
                                cluster_cfg["artifact_dir"],
                                if_real_data=False,
                                cluster_manager=hu_search_manager,
                                evaluation_data_path=cluster_cfg["evaluation_data_path"],
                                aggressive=hu_aggressive,
                                return_metadata=True,
                            )
                        finally:
                            hu_search_manager.clear_job_context()

                    (hu_combo, hu_meta), hu_elapsed, hu_predict_time, hu_predict_count, hu_contention_time = (
                        _run_profiled_search(_run_hu_pts)
                    )
                    hu_final_bw = _evaluate_combo_with_manager(hu_real_manager, hu_combo, probe_job_id)
                    hu_result = {
                        "combo": hu_combo,
                        "final_bw": float(hu_final_bw),
                        "standalone_bw": float(hu_final_bw),
                        "combo_signature": _combo_signature(hu_combo),
                        "measured_wall_time_s": float(hu_elapsed),
                        "predictor_time_s": float(hu_predict_time),
                        "predictor_calls": int(hu_predict_count),
                        "contention_time_s": float(hu_contention_time),
                        "eha_phase_time_s": 0.0,
                        "pts_phase_time_s": float(hu_meta.get("pts_time", hu_elapsed)),
                        "eha_meta": {},
                    }

                    adaptive_combo: Optional[np.ndarray] = None
                    adaptive_meta: Dict[str, Any] = {}
                    adaptive_elapsed = 0.0
                    adaptive_predict_time = 0.0
                    adaptive_predict_count = 0
                    adaptive_contention_time = 0.0
                    adaptive_final_bw = 0.0
                    selected_backend = ""
                    if "BandPilot" in selected_algorithms:
                        adaptive_search_manager, adaptive_real_manager = _new_managers(
                            normalized_mode=normalized_mode,
                            background_combo=background_combo,
                            occupancy_seed=occupancy_seed,
                        )

                        def _run_hu_adaptive() -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
                            adaptive_search_manager.set_job_context(probe_job_id)
                            try:
                                return improved_searching_algo(
                                    total_gpu,
                                    avail_gpu,
                                    model,
                                    k,
                                    total_gpu,
                                    cluster_cfg["gpu_bw_dict_list"],
                                    cluster_cfg["switch_config"],
                                    cluster_cfg["training_data_path"],
                                    cluster_cfg["device"],
                                    cluster_cfg["artifact_dir"],
                                if_real_data=False,
                                cluster_manager=adaptive_search_manager,
                                evaluation_data_path=cluster_cfg["evaluation_data_path"],
                                aggressive=hu_aggressive,
                                adaptive_pts=True,
                                return_metadata=True,
                                adaptive_runtime_state=runtime_state,
                                )
                            finally:
                                adaptive_search_manager.clear_job_context()

                        (
                            adaptive_combo,
                            adaptive_meta,
                        ), adaptive_elapsed, adaptive_predict_time, adaptive_predict_count, adaptive_contention_time = (
                            _run_profiled_search(_run_hu_adaptive)
                        )
                        adaptive_final_bw = _evaluate_combo_with_manager(
                            adaptive_real_manager,
                            adaptive_combo,
                            probe_job_id,
                        )
                        selected_backend = _resolve_mainline_selected_backend(
                            pts_triggered=bool(adaptive_meta.get("pts_triggered", False)),
                            final_combo=adaptive_combo,
                            eha_combo=eha_combo,
                            hu_combo=hu_combo,
                        )

                    if "EHA" in selected_algorithms:
                        eha_record = _build_backend_record(
                            case_context=case_context,
                            algorithm="EHA",
                            final_bw=float(eha_final_bw),
                            search_meta=eha_meta,
                            combo=eha_combo,
                            measured_wall_time_s=float(eha_elapsed),
                            predictor_time_s=float(eha_predict_time),
                            predictor_calls=int(eha_predict_count),
                            contention_time_s=float(eha_contention_time),
                            latency_evidence_kind="real_measured",
                            bandwidth_evidence_kind="real_evaluated",
                            evidence_type="measured",
                        )
                        records.append(eha_record)
                        pending_records.append(eha_record)
                        completed_keys.add(
                            (
                                cluster_type,
                                normalized_mode,
                                int(k),
                                int(bank_round_idx),
                                int(seed),
                                "EHA",
                            )
                        )

                    if "PTS" in selected_algorithms:
                        hu_record = _build_backend_record(
                            case_context=case_context,
                            algorithm="PTS",
                            final_bw=float(hu_final_bw),
                            search_meta=hu_meta,
                            combo=hu_combo,
                            measured_wall_time_s=float(hu_elapsed),
                            predictor_time_s=float(hu_predict_time),
                            predictor_calls=int(hu_predict_count),
                            contention_time_s=float(hu_contention_time),
                            latency_evidence_kind="real_measured",
                            bandwidth_evidence_kind="real_evaluated",
                            evidence_type="measured",
                        )
                        records.append(hu_record)
                        pending_records.append(hu_record)
                        completed_keys.add(
                            (
                                cluster_type,
                                normalized_mode,
                                int(k),
                                int(bank_round_idx),
                                int(seed),
                                "PTS",
                            )
                        )

                    if "BandPilot" in selected_algorithms:
                        adaptive_record = _build_backend_record(
                            case_context=case_context,
                            algorithm="BandPilot",
                            final_bw=float(adaptive_final_bw),
                            search_meta=_build_mainline_adaptive_search_meta(
                                adaptive_meta=adaptive_meta,
                                eha_meta=adaptive_meta.get("eha_meta", eha_result["eha_meta"]),
                                selected_backend=selected_backend,
                            ),
                            combo=adaptive_combo,
                            measured_wall_time_s=float(adaptive_elapsed),
                            predictor_time_s=float(adaptive_predict_time),
                            predictor_calls=int(adaptive_predict_count),
                            contention_time_s=float(adaptive_contention_time),
                            latency_evidence_kind="real_measured",
                            bandwidth_evidence_kind="real_evaluated",
                            evidence_type="measured",
                        )
                        records.append(adaptive_record)
                        pending_records.append(adaptive_record)
                        completed_keys.add(
                            (
                                cluster_type,
                                normalized_mode,
                                int(k),
                                int(bank_round_idx),
                                int(seed),
                                "BandPilot",
                            )
                        )

                    if cluster_cache_path is not None and len(pending_records) >= max(1, int(save_every_n_records)):
                        cached_cluster_df = _persist_tier1_cache(
                            cluster_cache_path,
                            cached_cluster_df,
                            pending_records,
                        )
                        logger.info(
                            "Tier 1 cache flush | cluster=%s | path=%s | new_records=%s | total_rows=%s",
                            cluster_type,
                            cluster_cache_path,
                            len(pending_records),
                            len(cached_cluster_df),
                        )
                        pending_records = []

                runtime_summary = runtime_state.finish_bank()
                _log_runtime_bank_summary(
                    stage_name="Tier 1",
                    cluster_type=cluster_type,
                    repeat_idx=int(bank_round_idx),
                    summary=runtime_summary,
                )

        if cluster_cache_path is not None:
            cached_cluster_df = _persist_tier1_cache(cluster_cache_path, cached_cluster_df, pending_records)
            logger.info(
                "Tier 1 cluster cache finalized | cluster=%s | path=%s | total_rows=%s",
                cluster_type,
                cluster_cache_path,
                len(cached_cluster_df),
            )

    if output_dir is not None:
        cluster_frames = []
        for cluster_cfg in cluster_configs:
            cache_path = output_dir / f"tier1_{cluster_cfg['cluster_type']}.csv"
            cache_df = _load_cached_dataframe(cache_path)
            cache_df = _prepare_tier1_cache_dataframe(
                cache_df,
                f"Tier 1 final cache [{cluster_cfg['cluster_type']}]",
            )
            if not cache_df.empty:
                cluster_frames.append(_annotate_tier1_quality(cache_df))
        result_df = pd.concat(cluster_frames, ignore_index=True, sort=False) if cluster_frames else pd.DataFrame()
    else:
        result_df = _annotate_tier1_quality(pd.DataFrame(records))
    logger.info("Tier 1 done | rows=%s", len(result_df))
    return result_df


def build_scaled_cluster_config(
    total_gpu: int,
    cluster_template: dict,
    inter_pod_factor: float,
) -> dict:
    """Build a large virtual cluster by repeating one real 4-node template."""
    if total_gpu % 32 != 0:
        raise ValueError("Scaled benchmark currently expects total_gpu to be a multiple of 32.")
    repeated_nodes = total_gpu // 8
    template_bw_dicts = list(cluster_template["gpu_bw_dict_list"])
    return {
        "cluster_type": cluster_template["cluster_type"],
        "total_gpu": int(total_gpu),
        "pod_size": int(len(template_bw_dicts)),
        "gpu_bw_dict_list": [template_bw_dicts[idx % len(template_bw_dicts)] for idx in range(repeated_nodes)],
        "pod_bw_lookup": {
            "gpu_bw_dict_list": template_bw_dicts,
            "switch_config": cluster_template["switch_config"],
            "data_path": cluster_template["evaluation_data_path"],
        },
        "inter_pod_factor": float(inter_pod_factor),
        # Preserve cluster_type so scaled Tier 2 rows can be grouped by family.
        "switch_config": SwitchBandwidthConfig(
            num_machines=repeated_nodes,
            cluster_type=cluster_template["cluster_type"],
        ),
        "training_data_path": cluster_template["training_data_path"],
        "hu_unit_gate": normalize_hu_unit_gate_config(cluster_template.get("hu_unit_gate")),
    }


def estimate_scaled_bandwidth(
    gpu_config: np.ndarray,
    total_gpu: int,
    pod_size: int,
    gpu_bw_dict_list: list,
    pod_bw_lookup: dict,
    inter_pod_factor: float,
) -> float:
    """Estimate large-scale bandwidth with per-pod real lookup plus inter-pod decay."""
    config = np.asarray(gpu_config, dtype=int)
    if int(config.sum()) <= 1:
        return 0.0

    gpus_per_pod = pod_size * 8
    active_pods: List[int] = []
    pod_bandwidths: List[float] = []

    for pod_idx, start in enumerate(range(0, total_gpu, gpus_per_pod)):
        pod_config = config[start : start + gpus_per_pod]
        if int(pod_config.sum()) <= 0:
            continue
        local_bw, part_bws, _ = calculate_bandwidth_values(
            pod_config,
            gpus_per_pod,
            pod_bw_lookup["gpu_bw_dict_list"],
            pod_bw_lookup["switch_config"],
            pod_bw_lookup["data_path"],
        )
        local_node_bws = []
        for node_offset in range(0, gpus_per_pod, 8):
            node_slice = tuple(int(x) for x in pod_config[node_offset : node_offset + 8])
            if any(node_slice):
                node_dict = pod_bw_lookup["gpu_bw_dict_list"][node_offset // 8]
                local_node_bws.append(float(node_dict.get(node_slice, 0.0)))
        candidates = [float(local_bw)] if float(local_bw) > 0 else []
        candidates.extend([bw for bw in part_bws if bw > 0])
        candidates.extend([bw for bw in local_node_bws if bw > 0])
        pod_bandwidths.append(float(min(candidates)) if candidates else 0.0)
        active_pods.append(pod_idx)

    if not pod_bandwidths:
        return 0.0

    local_bottleneck = min(pod_bandwidths)
    if len(active_pods) <= 1:
        return float(local_bottleneck)

    span = max(active_pods) - min(active_pods)
    hop_distance = max(1, span)
    fanout_penalty = inter_pod_factor ** max(0, len(active_pods) - 2)
    cross_pod_bw = local_bottleneck * (inter_pod_factor ** hop_distance) * fanout_penalty
    return float(min(local_bottleneck, cross_pod_bw))


def generate_realistic_avail_gpu(
    total_gpu: int,
    target_avail: int,
    mode: str = "mixed",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a realistic availability mask with node-aligned and partial occupancy."""
    if target_avail >= total_gpu:
        return np.arange(total_gpu, dtype=int)

    rng = np.random.default_rng(seed)
    total_nodes = total_gpu // 8
    target_busy = total_gpu - target_avail
    busy: set[int] = set()

    def _fill_random(count: int, allowed: Optional[Sequence[int]] = None) -> None:
        if count <= 0:
            return
        population = list(allowed) if allowed is not None else list(range(total_gpu))
        candidates = [gpu for gpu in population if gpu not in busy]
        if not candidates:
            return
        take = min(count, len(candidates))
        busy.update(int(x) for x in rng.choice(candidates, take, replace=False))

    selected_mode = mode
    if mode == "mixed":
        selected_mode = rng.choice(["node_aligned", "partial", "random"], p=[0.4, 0.3, 0.3]).item()

    if selected_mode == "node_aligned":
        full_nodes = min(total_nodes, target_busy // 8)
        if full_nodes > 0:
            node_ids = rng.choice(total_nodes, full_nodes, replace=False)
            for node_id in node_ids:
                busy.update(range(int(node_id) * 8, int(node_id) * 8 + 8))
        _fill_random(target_busy - len(busy))
    elif selected_mode == "partial":
        node_order = rng.permutation(total_nodes)
        remaining = target_busy
        for node_id in node_order:
            if remaining <= 0:
                break
            local_busy = int(min(remaining, rng.integers(1, 7)))
            node_gpus = list(range(int(node_id) * 8, int(node_id) * 8 + 8))
            _fill_random(local_busy, node_gpus)
            remaining = target_busy - len(busy)
        _fill_random(remaining)
    else:
        _fill_random(target_busy)

    if len(busy) < target_busy:
        _fill_random(target_busy - len(busy))
    available = np.array(sorted(set(range(total_gpu)) - busy), dtype=int)
    return available


def _attach_scaled_latency_estimate(df: pd.DataFrame, inference_profile: pd.DataFrame) -> pd.DataFrame:
    """Attach synthesized predictor-cost bounds to scaled search traces."""
    if df.empty:
        return df
    profile = inference_profile.sort_values("node_count")
    x = profile["node_count"].to_numpy(dtype=float)
    p50_col = "e2e_p50_ms" if "e2e_p50_ms" in profile.columns else "p50_ms"
    p95_col = "e2e_p95_ms" if "e2e_p95_ms" in profile.columns else "p95_ms"
    y_p50 = profile[p50_col].to_numpy(dtype=float)
    y_p95 = profile[p95_col].to_numpy(dtype=float)

    def _infer_ms(node_count: int, values: np.ndarray) -> float:
        return float(np.interp(float(node_count), x, values))

    df = df.copy()
    df["predictor_e2e_p50_ms"] = df["total_gpu"].apply(lambda value: _infer_ms(int(value) // 8, y_p50))
    df["predictor_e2e_p95_ms"] = df["total_gpu"].apply(lambda value: _infer_ms(int(value) // 8, y_p95))
    if "non_predictor_search_time_s" not in df.columns:
        df["non_predictor_search_time_s"] = (
            df.get("measured_wall_time_s", df.get("elapsed_time", 0.0))
            - df.get("predictor_time_s", df.get("predict_time", 0.0))
        ).clip(lower=0.0)
    df["synthesized_wall_time_p50_s"] = (
        df["non_predictor_search_time_s"] + df.get("predictor_calls", df.get("predict_count", 0)) * df["predictor_e2e_p50_ms"] / 1000.0
    )
    df["synthesized_wall_time_p95_s"] = (
        df["non_predictor_search_time_s"] + df.get("predictor_calls", df.get("predict_count", 0)) * df["predictor_e2e_p95_ms"] / 1000.0
    )
    return df


def run_scaled_latency_benchmark(
    cluster_cfg: dict,
    gpu_counts: List[int],
    k_values: List[int],
    avail_ratios: List[float],
    contention_modes: List[str],
    inter_pod_factors: List[float],
    repeat_num: int,
    random_seed: int = 0,
    inference_profile: Optional[pd.DataFrame] = None,
    algorithm_names: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
    resume: bool = True,
    save_every_n_records: int = 50,
    public_view_cfg: Optional[Dict[str, Any]] = None,
    public_repeat_num: Optional[int] = None,
) -> pd.DataFrame:
    """Run Tier 2 scaled simulations with mainline runtime-adaptive `BandPilot`.

    Execution model:
    - each `repeat_idx` walks the full scenario grid deterministically;
    - each `repeat_idx` owns a fresh runtime bank, which is reused across
      scale/factor/availability/contention/k cases within that repeat;
    - `BandPilot` uses the `search.py` runtime adaptive path with PTS replay.
    """
    records: List[Dict[str, Any]] = []
    selected_algorithms = _resolve_algorithm_names(algorithm_names)
    resolved_public_cfg = dict(public_view_cfg or DEFAULT_PUBLIC_VIEW)
    cluster_type = str(cluster_cfg["cluster_type"])
    adaptive_policy = RuntimeAdaptiveKNNConfig.from_mapping(
        cluster_cfg.get("adaptive_runtime_policy")
    )
    effective_resume = bool(resume)
    if resume and "BandPilot" in selected_algorithms:
        logger.warning(
            "Tier 2 cluster=%s | BandPilot uses mainline runtime state; "
            "existing cache cannot restore bank state, disabling resume for this cluster.",
            cluster_type,
        )
        effective_resume = False
    cached_df = (
        _load_cached_dataframe(output_path)
        if (effective_resume and output_path is not None)
        else pd.DataFrame()
    )
    completed_keys = _build_completed_key_set(cached_df, SCALED_CACHE_KEY_COLUMNS)
    pending_records: List[Dict[str, Any]] = []

    tier2_group_specs = _build_tier2_group_specs(
        cluster_type=cluster_type,
        gpu_counts=gpu_counts,
        k_values=k_values,
        avail_ratios=avail_ratios,
        contention_modes=contention_modes,
        inter_pod_factors=inter_pod_factors,
        repeat_num=repeat_num,
        public_view_cfg=resolved_public_cfg,
        public_repeat_num=public_repeat_num,
    )
    max_repeat = max((int(group["target_round_num"]) for group in tier2_group_specs), default=0)

    logger.info(
        "Tier 2 start | cluster=%s | algorithms=%s | scenario_groups=%s | max_rounds=%s | cached_rows=%s | completed_keys=%s | adaptive_policy=%s",
        cluster_type,
        selected_algorithms,
        len(tier2_group_specs),
        max_repeat,
        len(cached_df),
        len(completed_keys),
        adaptive_policy.policy_name,
    )

    scaled_cfg_cache: Dict[Tuple[int, float], Dict[str, Any]] = {}
    predictor_fn_cache: Dict[Tuple[int, float], Callable[[np.ndarray], float]] = {}

    def _get_scaled_cfg(total_gpu: int, inter_pod_factor: float) -> Dict[str, Any]:
        key = (int(total_gpu), float(inter_pod_factor))
        if key not in scaled_cfg_cache:
            scaled_cfg_cache[key] = build_scaled_cluster_config(total_gpu, cluster_cfg, inter_pod_factor)
        return scaled_cfg_cache[key]

    def _get_predictor_fn(total_gpu: int, inter_pod_factor: float) -> Callable[[np.ndarray], float]:
        key = (int(total_gpu), float(inter_pod_factor))
        if key not in predictor_fn_cache:
            scaled_cfg = _get_scaled_cfg(total_gpu, inter_pod_factor)

            def _predictor_fn(combo: np.ndarray) -> float:
                return estimate_scaled_bandwidth(
                    gpu_config=combo,
                    total_gpu=scaled_cfg["total_gpu"],
                    pod_size=scaled_cfg["pod_size"],
                    gpu_bw_dict_list=scaled_cfg["gpu_bw_dict_list"],
                    pod_bw_lookup=scaled_cfg["pod_bw_lookup"],
                    inter_pod_factor=scaled_cfg["inter_pod_factor"],
                )

            predictor_fn_cache[key] = _predictor_fn
        return predictor_fn_cache[key]

    for group_spec in tier2_group_specs:
        total_gpu = int(group_spec["total_gpu"])
        inter_pod_factor = float(group_spec["inter_pod_factor"])
        avail_ratio = float(group_spec["avail_ratio"])
        normalized_mode = str(group_spec["contention_mode"])
        target_avail = int(group_spec["target_avail"])
        runtime_state = RuntimeAdaptiveKNNState(
            config=adaptive_policy,
            bank_id=str(group_spec["scenario_group_id"]),
        )
        logger.info(
            "Tier 2 cluster=%s | group=%s | bank_scope=%s | rounds=%s | k_values=%s",
            cluster_type,
            group_spec["scenario_group_id"],
            group_spec["bank_scope"],
            group_spec["target_round_num"],
            group_spec["k_values"],
        )

        for bank_round_idx in range(int(group_spec["target_round_num"])):
            seed = _build_bank_round_seed(
                int(random_seed + total_gpu * 13 + int(inter_pod_factor * 100)),
                str(group_spec["scenario_group_id"]),
                int(bank_round_idx),
            )
            for k in list(group_spec["k_values"]):
                case_completed = all(
                    (
                        int(total_gpu),
                        int(k),
                        float(avail_ratio),
                        normalized_mode,
                        float(inter_pod_factor),
                        int(bank_round_idx),
                        int(seed),
                        algo_name,
                    )
                    in completed_keys
                    for algo_name in selected_algorithms
                )
                if case_completed:
                    continue

                avail_gpu = generate_realistic_avail_gpu(
                    total_gpu=total_gpu,
                    target_avail=target_avail,
                    mode="mixed",
                    seed=seed,
                )
                if len(avail_gpu) < k:
                    continue

                background_combo = _build_background_combo(total_gpu, avail_gpu)
                background_gpu = np.where(background_combo == 1)[0].astype(int).tolist()
                occupancy_seed = int(seed + k * 97)
                probe_job_id = _build_job_id(k, int(bank_round_idx))
                scaled_cfg = _get_scaled_cfg(total_gpu, inter_pod_factor)
                predictor_fn = _get_predictor_fn(total_gpu, inter_pod_factor)

                def _new_managers() -> Tuple[ClusterStateManager, ClusterStateManager, ProfilingPredictor]:
                    """Create fresh managers for a scaled primitive backend."""

                    search_predictor = ProfilingPredictor(predictor_fn)
                    real_predictor = ProfilingPredictor(predictor_fn)
                    search_manager = _create_cluster_manager(
                        total_gpu=total_gpu,
                        predictor=search_predictor,
                        contention_mode=normalized_mode,
                        background_combo=background_combo,
                        occupancy_seed=occupancy_seed,
                    )
                    real_manager = _create_cluster_manager(
                        total_gpu=total_gpu,
                        predictor=real_predictor,
                        contention_mode=normalized_mode,
                        background_combo=background_combo,
                        occupancy_seed=occupancy_seed,
                    )
                    return search_manager, real_manager, search_predictor

                case_context = {
                    "cluster_type": cluster_type,
                    "total_gpu": int(total_gpu),
                    "k": int(k),
                    "avail_ratio": float(avail_ratio),
                    "contention_mode": normalized_mode,
                    "inter_pod_factor": float(inter_pod_factor),
                    "repeat_idx": int(bank_round_idx),
                    "bank_round_idx": int(bank_round_idx),
                    "seed": int(seed),
                    "avail_gpu_count": int(len(avail_gpu)),
                    "avail_gpu": [int(value) for value in avail_gpu.tolist()],
                    "background_gpu": background_gpu,
                    "avail_signature": _index_signature(avail_gpu.tolist()),
                    "background_signature": _index_signature(background_gpu),
                    "if_dynamic": True,
                    "search_if_real_data": False,
                    "occupancy_seed": int(occupancy_seed),
                    "probe_job_id": int(probe_job_id),
                    "scenario_group_id": str(group_spec["scenario_group_id"]),
                    "bank_scope": str(group_spec["bank_scope"]),
                }

                eha_search_manager, eha_real_manager, eha_predictor = _new_managers()

                def _run_eha() -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
                    eha_search_manager.set_job_context(probe_job_id)
                    try:
                        combo, eha_meta = eha_search(
                            total_gpu,
                            avail_gpu,
                            model=None,
                            gpu_need=k,
                            total_gpu=total_gpu,
                            gpu_bw_dict_list=scaled_cfg["gpu_bw_dict_list"],
                            switch_config=scaled_cfg["switch_config"],
                            training_data_path=scaled_cfg["training_data_path"],
                            device=torch.device("cpu"),
                            artifact_dir=Path("."),
                            if_real_data=False,
                            cluster_manager=eha_search_manager,
                            return_confidence=True,
                        )
                    finally:
                        eha_search_manager.clear_job_context()
                    return combo, {
                        "pts_triggered": False,
                        "trigger_reason": "eha_only",
                        "eha_meta": eha_meta,
                        "eha_time": 0.0,
                        "pts_time": 0.0,
                        "pts_policy": "",
                    }

                (eha_combo, eha_meta), eha_elapsed, _, _, eha_contention_time = _run_profiled_search(_run_eha)
                eha_meta["eha_time"] = eha_elapsed
                eha_final_bw = _evaluate_combo_with_manager(eha_real_manager, eha_combo, probe_job_id)
                eha_result = {
                    "combo": eha_combo,
                    "final_bw": float(eha_final_bw),
                    "standalone_bw": float(eha_final_bw),
                    "combo_signature": _combo_signature(eha_combo),
                    "measured_wall_time_s": float(eha_elapsed),
                    "predictor_time_s": float(eha_predictor.total_time),
                    "predictor_calls": int(eha_predictor.call_count),
                    "contention_time_s": float(eha_contention_time),
                    "eha_phase_time_s": float(eha_elapsed),
                    "pts_phase_time_s": 0.0,
                    "eha_meta": dict(eha_meta.get("eha_meta", {}) or {}),
                }

                hu_search_manager, hu_real_manager, hu_predictor = _new_managers()
                hu_aggressive = bool(scaled_cfg.get("hu_unit_gate", {}).get("aggressive", False))

                def _run_hu_pts() -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
                    hu_search_manager.set_job_context(probe_job_id)
                    try:
                        return hu_pts_only_search(
                            total_gpu,
                            avail_gpu,
                            model=None,
                            gpu_need=k,
                            total_gpu=total_gpu,
                            gpu_bw_dict_list=scaled_cfg["gpu_bw_dict_list"],
                            switch_config=scaled_cfg["switch_config"],
                            training_data_path=scaled_cfg["training_data_path"],
                            device=torch.device("cpu"),
                            artifact_dir=Path("."),
                            if_real_data=False,
                            cluster_manager=hu_search_manager,
                            aggressive=hu_aggressive,
                            return_metadata=True,
                        )
                    finally:
                        hu_search_manager.clear_job_context()

                (hu_combo, hu_meta), hu_elapsed, _, _, hu_contention_time = _run_profiled_search(_run_hu_pts)
                hu_final_bw = _evaluate_combo_with_manager(hu_real_manager, hu_combo, probe_job_id)
                hu_result = {
                    "combo": hu_combo,
                    "final_bw": float(hu_final_bw),
                    "standalone_bw": float(hu_final_bw),
                    "combo_signature": _combo_signature(hu_combo),
                    "measured_wall_time_s": float(hu_elapsed),
                    "predictor_time_s": float(hu_predictor.total_time),
                    "predictor_calls": int(hu_predictor.call_count),
                    "contention_time_s": float(hu_contention_time),
                    "eha_phase_time_s": 0.0,
                    "pts_phase_time_s": float(hu_meta.get("pts_time", hu_elapsed)),
                    "eha_meta": {},
                }

                adaptive_combo: Optional[np.ndarray] = None
                adaptive_meta: Dict[str, Any] = {}
                adaptive_elapsed = 0.0
                adaptive_predict_time = 0.0
                adaptive_predict_count = 0
                adaptive_contention_time = 0.0
                adaptive_final_bw = 0.0
                selected_backend = ""
                if "BandPilot" in selected_algorithms:
                    adaptive_search_manager, adaptive_real_manager, adaptive_predictor = _new_managers()

                    def _run_hu_adaptive() -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
                        adaptive_search_manager.set_job_context(probe_job_id)
                        try:
                            return improved_searching_algo(
                                total_gpu,
                                avail_gpu,
                                model=None,
                                gpu_need=k,
                                total_gpu=total_gpu,
                                gpu_bw_dict_list=scaled_cfg["gpu_bw_dict_list"],
                                switch_config=scaled_cfg["switch_config"],
                                training_data_path=scaled_cfg["training_data_path"],
                                device=torch.device("cpu"),
                                artifact_dir=Path("."),
                                if_real_data=False,
                                cluster_manager=adaptive_search_manager,
                                aggressive=hu_aggressive,
                                adaptive_pts=True,
                                return_metadata=True,
                                adaptive_runtime_state=runtime_state,
                            )
                        finally:
                            adaptive_search_manager.clear_job_context()

                    (
                        adaptive_combo,
                        adaptive_meta,
                    ), adaptive_elapsed, _, _, adaptive_contention_time = _run_profiled_search(_run_hu_adaptive)
                    adaptive_predict_time = float(adaptive_predictor.total_time)
                    adaptive_predict_count = int(adaptive_predictor.call_count)
                    adaptive_final_bw = _evaluate_combo_with_manager(
                        adaptive_real_manager,
                        adaptive_combo,
                        probe_job_id,
                    )
                    selected_backend = _resolve_mainline_selected_backend(
                        pts_triggered=bool(adaptive_meta.get("pts_triggered", False)),
                        final_combo=adaptive_combo,
                        eha_combo=eha_combo,
                        hu_combo=hu_combo,
                    )

                if "EHA" in selected_algorithms:
                    num_active_pods, cross_pod_ratio = _compute_pod_stats(
                        eha_combo,
                        total_gpu=total_gpu,
                        pod_size=scaled_cfg["pod_size"],
                    )
                    eha_record = _build_backend_record(
                        case_context=case_context,
                        algorithm="EHA",
                        final_bw=float(eha_final_bw),
                        search_meta=eha_meta,
                        combo=eha_combo,
                        measured_wall_time_s=float(eha_elapsed),
                        predictor_time_s=float(eha_predictor.total_time),
                        predictor_calls=int(eha_predictor.call_count),
                        contention_time_s=float(eha_contention_time),
                        latency_evidence_kind="scaled_trace",
                        bandwidth_evidence_kind="scaled_estimated",
                        evidence_type="simulated",
                        extra_fields={
                            "num_active_pods": int(num_active_pods),
                            "cross_pod_ratio": float(cross_pod_ratio),
                        },
                    )
                    records.append(eha_record)
                    pending_records.append(eha_record)
                    completed_keys.add(
                        (
                            int(total_gpu),
                            int(k),
                            float(avail_ratio),
                            normalized_mode,
                            float(inter_pod_factor),
                            int(bank_round_idx),
                            int(seed),
                            "EHA",
                        )
                    )

                if "PTS" in selected_algorithms:
                    num_active_pods, cross_pod_ratio = _compute_pod_stats(
                        hu_combo,
                        total_gpu=total_gpu,
                        pod_size=scaled_cfg["pod_size"],
                    )
                    hu_record = _build_backend_record(
                        case_context=case_context,
                        algorithm="PTS",
                        final_bw=float(hu_final_bw),
                        search_meta=hu_meta,
                        combo=hu_combo,
                        measured_wall_time_s=float(hu_elapsed),
                        predictor_time_s=float(hu_predictor.total_time),
                        predictor_calls=int(hu_predictor.call_count),
                        contention_time_s=float(hu_contention_time),
                        latency_evidence_kind="scaled_trace",
                        bandwidth_evidence_kind="scaled_estimated",
                        evidence_type="simulated",
                        extra_fields={
                            "num_active_pods": int(num_active_pods),
                            "cross_pod_ratio": float(cross_pod_ratio),
                        },
                    )
                    records.append(hu_record)
                    pending_records.append(hu_record)
                    completed_keys.add(
                        (
                            int(total_gpu),
                            int(k),
                            float(avail_ratio),
                            normalized_mode,
                            float(inter_pod_factor),
                            int(bank_round_idx),
                            int(seed),
                            "PTS",
                        )
                    )

                if "BandPilot" in selected_algorithms:
                    num_active_pods, cross_pod_ratio = _compute_pod_stats(
                        adaptive_combo,
                        total_gpu=total_gpu,
                        pod_size=scaled_cfg["pod_size"],
                    )
                    adaptive_record = _build_backend_record(
                        case_context=case_context,
                        algorithm="BandPilot",
                        final_bw=float(adaptive_final_bw),
                        search_meta=_build_mainline_adaptive_search_meta(
                            adaptive_meta=adaptive_meta,
                            eha_meta=adaptive_meta.get("eha_meta", eha_result["eha_meta"]),
                            selected_backend=selected_backend,
                        ),
                        combo=adaptive_combo,
                        measured_wall_time_s=float(adaptive_elapsed),
                        predictor_time_s=float(adaptive_predict_time),
                        predictor_calls=int(adaptive_predict_count),
                        contention_time_s=float(adaptive_contention_time),
                        latency_evidence_kind="scaled_trace",
                        bandwidth_evidence_kind="scaled_estimated",
                        evidence_type="simulated",
                        extra_fields={
                            "num_active_pods": int(num_active_pods),
                            "cross_pod_ratio": float(cross_pod_ratio),
                        },
                    )
                    records.append(adaptive_record)
                    pending_records.append(adaptive_record)
                    completed_keys.add(
                        (
                            int(total_gpu),
                            int(k),
                            float(avail_ratio),
                            normalized_mode,
                            float(inter_pod_factor),
                            int(bank_round_idx),
                            int(seed),
                            "BandPilot",
                        )
                    )

                if output_path is not None and len(pending_records) >= max(1, int(save_every_n_records)):
                    cached_df = _persist_tier2_cache(output_path, cached_df, pending_records, inference_profile)
                    logger.info(
                        "Tier 2 cache flush | path=%s | new_records=%s | total_rows=%s",
                        output_path,
                        len(pending_records),
                        len(cached_df),
                    )
                    pending_records = []

            runtime_summary = runtime_state.finish_bank()
            _log_runtime_bank_summary(
                stage_name="Tier 2",
                cluster_type=cluster_type,
                repeat_idx=int(bank_round_idx),
                summary=runtime_summary,
            )

    if output_path is not None:
        cached_df = _persist_tier2_cache(output_path, cached_df, pending_records, inference_profile)
        logger.info("Tier 2 cache finalized | path=%s | total_rows=%s", output_path, len(cached_df))
        df = cached_df
    else:
        df = pd.DataFrame(records)
        if inference_profile is not None:
            df = _attach_scaled_latency_estimate(df, inference_profile)
    logger.info("Tier 2 done | rows=%s", len(df))
    return df


def profile_model_inference_scaling(
    model,
    node_counts: List[int],
    device: str | torch.device,
    artifact_dir: Path,
    cluster_type: str,
    repeats: int = 100,
    num_train_samples: Optional[int] = None,
    output_path: Optional[Path] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Profile single-query predictor latency as a function of node count."""
    resolved_device = torch.device(device)
    records: List[Dict[str, float]] = []
    cached_df = _load_cached_dataframe(output_path) if (resume and output_path is not None) else pd.DataFrame()
    cached_device_df = (
        cached_df[cached_df["device"] == str(resolved_device)].copy()
        if (not cached_df.empty and "device" in cached_df.columns)
        else pd.DataFrame()
    )
    completed_node_counts = set()
    if not cached_device_df.empty and "node_count" in cached_device_df.columns:
        completed_node_counts = set(int(value) for value in cached_device_df["node_count"].tolist())
    warmup = min(10, max(2, repeats // 10))
    preloaded_artifacts = preload_prediction_artifacts(
        artifact_dir,
        num_train_samples=num_train_samples,
    )
    logger.info(
        "Tier 3 inference profiling start | cluster=%s | node_counts=%s | repeats=%s | device=%s | resume=%s | cached_rows=%s",
        cluster_type,
        node_counts,
        repeats,
        resolved_device,
        resume,
        len(cached_device_df),
    )

    for node_count in node_counts:
        if int(node_count) in completed_node_counts:
            continue
        part_bws = np.full((1, node_count), 120.0, dtype=float)
        per_node_counts = np.full((1, node_count), 4, dtype=int)
        total_counts = np.array([[node_count * 4]], dtype=float)

        for _ in range(warmup):
            _ = predict_with_model(
                model,
                part_bws,
                per_node_counts,
                total_counts,
                resolved_device,
                artifact_dir,
                num_train_samples=num_train_samples,
            )
            _ = predict_with_model(
                model,
                part_bws,
                per_node_counts,
                total_counts,
                resolved_device,
                artifact_dir,
                num_train_samples=num_train_samples,
                preloaded_artifacts=preloaded_artifacts,
            )

        e2e_latencies_ms: List[float] = []
        preloaded_latencies_ms: List[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = predict_with_model(
                model,
                part_bws,
                per_node_counts,
                total_counts,
                resolved_device,
                artifact_dir,
                num_train_samples=num_train_samples,
            )
            e2e_latencies_ms.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            _ = predict_with_model(
                model,
                part_bws,
                per_node_counts,
                total_counts,
                resolved_device,
                artifact_dir,
                num_train_samples=num_train_samples,
                preloaded_artifacts=preloaded_artifacts,
            )
            preloaded_latencies_ms.append((time.perf_counter() - start) * 1000.0)

        e2e_arr = np.asarray(e2e_latencies_ms, dtype=float)
        preloaded_arr = np.asarray(preloaded_latencies_ms, dtype=float)
        records.append(
            {
                "node_count": int(node_count),
                "e2e_p50_ms": float(np.percentile(e2e_arr, 50)),
                "e2e_p95_ms": float(np.percentile(e2e_arr, 95)),
                "e2e_mean_ms": float(np.mean(e2e_arr)),
                "e2e_std_ms": float(np.std(e2e_arr)),
                "preloaded_p50_ms": float(np.percentile(preloaded_arr, 50)),
                "preloaded_p95_ms": float(np.percentile(preloaded_arr, 95)),
                "preloaded_mean_ms": float(np.mean(preloaded_arr)),
                "preloaded_std_ms": float(np.std(preloaded_arr)),
                "cluster_type": str(cluster_type),
                "device": str(resolved_device),
                "repeats_used": int(repeats),
            }
        )
        if output_path is not None:
            cached_df = _persist_inference_profile_cache(output_path, cached_df, [records[-1]])
            logger.info(
                "Tier 3 inference cache flush | path=%s | node_count=%s | total_rows=%s",
                output_path,
                node_count,
                len(cached_df),
            )

    if output_path is not None:
        cached_df = _persist_inference_profile_cache(output_path, cached_df, [])
        result_df = cached_df
        if "device" in result_df.columns:
            result_df = result_df[result_df["device"] == str(resolved_device)].copy()
    else:
        result_df = pd.DataFrame(records)
    logger.info("Tier 3 inference profiling done | rows=%s", len(result_df))
    return result_df


def extrapolate_search_overhead(
    tier2_results: pd.DataFrame,
    inference_profile: pd.DataFrame,
    target_gpu_counts: List[int],
    cluster_type: str,
) -> pd.DataFrame:
    """Synthesize deployment-level latency bounds from one cluster's scaled traces."""
    if tier2_results.empty or inference_profile.empty:
        return pd.DataFrame()
    logger.info(
        "Tier 4 latency synthesis start | cluster=%s | target_gpu_counts=%s | observed_scales=%s",
        cluster_type,
        target_gpu_counts,
        sorted(tier2_results["total_gpu"].unique().tolist()),
    )

    infer_profile = inference_profile.sort_values("node_count")
    infer_x = infer_profile["node_count"].to_numpy(dtype=float)
    p50_col = "e2e_p50_ms" if "e2e_p50_ms" in infer_profile.columns else "p50_ms"
    p95_col = "e2e_p95_ms" if "e2e_p95_ms" in infer_profile.columns else "p95_ms"
    infer_y_p50 = infer_profile[p50_col].to_numpy(dtype=float)
    infer_y_p95 = infer_profile[p95_col].to_numpy(dtype=float)

    observed_scales = sorted(int(v) for v in tier2_results["total_gpu"].unique().tolist())

    def _nearest_scale(scale: int) -> int:
        return min(observed_scales, key=lambda value: abs(value - scale))

    def _infer_ms(total_gpu: int, values: np.ndarray) -> float:
        return float(np.interp(total_gpu / 8.0, infer_x, values))

    records: List[Dict[str, Any]] = []
    for total_gpu in target_gpu_counts:
        nearest_scale = _nearest_scale(total_gpu)
        scale_factor = float(total_gpu) / float(nearest_scale)
        target_subset = tier2_results[tier2_results["total_gpu"] == nearest_scale].copy()
        if target_subset.empty:
            continue
        target_subset["reference_total_gpu"] = int(nearest_scale)
        target_subset["total_gpu"] = int(total_gpu)
        target_subset["scale_factor"] = float(scale_factor)
        if "measured_wall_time_s" not in target_subset.columns:
            target_subset["measured_wall_time_s"] = target_subset.get("elapsed_time", 0.0)
        if "non_predictor_search_time_s" not in target_subset.columns:
            target_subset["non_predictor_search_time_s"] = (
                target_subset["measured_wall_time_s"] - target_subset.get("predict_time", 0.0)
            ).clip(lower=0.0)
        if "predictor_calls" not in target_subset.columns:
            target_subset["predictor_calls"] = target_subset.get("predict_count", 0)
        target_subset["trace_measured_wall_time_s"] = target_subset["measured_wall_time_s"]
        target_subset["trace_non_predictor_search_time_s"] = target_subset["non_predictor_search_time_s"]
        target_subset["scaled_predictor_calls"] = target_subset["predictor_calls"] * scale_factor
        target_subset["scaled_non_predictor_search_time_s"] = (
            target_subset["non_predictor_search_time_s"] * scale_factor
        )
        target_subset["predictor_e2e_p50_ms"] = _infer_ms(total_gpu, infer_y_p50)
        target_subset["predictor_e2e_p95_ms"] = _infer_ms(total_gpu, infer_y_p95)
        target_subset["synthesized_wall_time_p50_s"] = (
            target_subset["scaled_non_predictor_search_time_s"]
            + target_subset["scaled_predictor_calls"] * target_subset["predictor_e2e_p50_ms"] / 1000.0
        )
        target_subset["synthesized_wall_time_p95_s"] = (
            target_subset["scaled_non_predictor_search_time_s"]
            + target_subset["scaled_predictor_calls"] * target_subset["predictor_e2e_p95_ms"] / 1000.0
        )
        target_subset["latency_evidence_kind"] = "synthesized"
        target_subset["evidence_type"] = "synthesized"
        target_subset["cluster_type"] = str(cluster_type)
        records.extend(target_subset.to_dict(orient="records"))

    result_df = pd.DataFrame(records)
    logger.info("Tier 4 latency synthesis done | cluster=%s | rows=%s", cluster_type, len(result_df))
    return result_df


def _filter_public_algorithm_rows(df: pd.DataFrame, public_view_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Filter one dataframe down to the reviewer-facing public algorithms."""
    return _filter_algorithm_rows(df, public_view_cfg["public_algorithms"])


def _select_public_diagnostic_algorithm(public_view_cfg: Dict[str, Any]) -> Optional[str]:
    """Select the adaptive diagnostics algorithm for public plots."""

    preferred = _canonicalize_algorithm_name("BandPilot")
    public_algorithms = [
        _canonicalize_algorithm_name(name)
        for name in public_view_cfg.get("public_algorithms", [])
    ]
    if preferred in set(public_algorithms):
        return preferred
    for candidate in public_algorithms + list(ALGO_ORDER):
        if candidate in set(ALGO_ORDER):
            return str(candidate)
    return None


def _filter_representative_slice(df: pd.DataFrame, public_view_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Select the fixed representative slice used by reviewer-facing Tier2/Tier3 plots."""
    if df.empty:
        return df.copy()
    subset = _filter_public_algorithm_rows(df, public_view_cfg)
    if "contention_mode" in subset.columns:
        subset = subset[
            subset["contention_mode"] == public_view_cfg["representative_contention_mode"]
        ]
    if "avail_ratio" in subset.columns:
        subset = subset[_match_float_series(subset["avail_ratio"], public_view_cfg["representative_avail_ratio"])]
    if "inter_pod_factor" in subset.columns:
        subset = subset[
            _match_float_series(
                subset["inter_pod_factor"],
                public_view_cfg["representative_inter_pod_factor"],
            )
        ]
    if "k" in subset.columns:
        subset = subset[subset["k"].astype(int).isin(public_view_cfg["representative_k_values"])]
    return subset.copy()


def _percentile_or_nan(series: pd.Series, percentile: float) -> float:
    """Return a percentile for a numeric series, or NaN when unavailable."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean.to_numpy(dtype=float), percentile))


def _mean_percent_or_nan(series: pd.Series) -> float:
    """Return the mean percentage for a boolean-like series, or NaN when unavailable."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean() * 100.0)


def _build_envelope_p95(
    df: pd.DataFrame,
    value_column: str,
    cluster_type: str,
    total_gpu: int,
    k: int,
    public_view_cfg: Dict[str, Any],
    algorithm_names: Optional[Sequence[str]] = None,
) -> float:
    """Compute the worst p95 over the allowed envelope contention modes."""
    if df.empty or value_column not in df.columns:
        return float("nan")
    subset = _filter_algorithm_rows(
        df,
        public_view_cfg["public_algorithms"] if algorithm_names is None else algorithm_names,
    )
    subset = subset[
        (subset["cluster_type"] == cluster_type)
        & (subset["total_gpu"].astype(int) == int(total_gpu))
        & (subset["k"].astype(int) == int(k))
        & (subset["contention_mode"].isin(public_view_cfg["envelope_contention_modes"]))
    ]
    if subset.empty:
        return float("nan")
    grouped = subset.groupby("contention_mode")[value_column].apply(lambda s: _percentile_or_nan(s, 95))
    grouped = grouped.dropna()
    if grouped.empty:
        return float("nan")
    return float(grouped.max())


def _build_upper_bound_fields(
    subset: pd.DataFrame,
    *,
    algorithm_name: Optional[str],
    p50_value_column: str,
    p95_value_column: Optional[str] = None,
    envelope_df: pd.DataFrame,
    cluster_type: str,
    total_gpu: int,
    k: int,
    public_view_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize the configured upper-bound algorithm on the same reviewer-facing slice."""
    fields: Dict[str, Any] = {
        "Upper-Bound Algorithm": algorithm_name or "",
        "Upper-Bound Latency p50 (s)": float("nan"),
        "Upper-Bound Latency p95 (s)": float("nan"),
        "Upper-Bound Envelope p95 (s)": float("nan"),
    }
    resolved_p95_column = p95_value_column or p50_value_column
    if (
        algorithm_name is None
        or subset.empty
        or p50_value_column not in subset.columns
        or resolved_p95_column not in subset.columns
    ):
        return fields
    fields["Upper-Bound Latency p50 (s)"] = _percentile_or_nan(subset[p50_value_column], 50)
    fields["Upper-Bound Latency p95 (s)"] = _percentile_or_nan(subset[resolved_p95_column], 95)
    fields["Upper-Bound Envelope p95 (s)"] = _build_envelope_p95(
        envelope_df,
        resolved_p95_column,
        cluster_type,
        total_gpu,
        k,
        public_view_cfg,
        algorithm_names=[algorithm_name],
    )
    return fields


def _save_latency_real_plot(
    df: pd.DataFrame,
    output_path: Path,
    public_view_cfg: Dict[str, Any],
) -> None:
    """Figure 1: 32-GPU real measured latency for the reviewer-facing primary algorithm."""
    selected = _filter_public_algorithm_rows(df, public_view_cfg)
    if selected.empty:
        return
    clusters = sorted(selected["cluster_type"].unique().tolist())
    contention_modes = ["idle", "common", "intensive"]
    fig, axes = plt.subplots(
        len(clusters),
        len(contention_modes),
        figsize=(5.1 * len(contention_modes), 3.9 * max(1, len(clusters))),
        sharey="row",
        squeeze=False,
    )
    for row_idx, cluster_type in enumerate(clusters):
        cluster_subset = selected[selected["cluster_type"] == cluster_type]
        available_algorithms = [
            name for name in public_view_cfg["public_algorithms"]
            if name in set(cluster_subset["algorithm"].unique())
        ]
        for col_idx, mode in enumerate(contention_modes):
            ax = axes[row_idx][col_idx]
            subset = cluster_subset[cluster_subset["contention_mode"] == mode]
            for algo_name in available_algorithms:
                stats = (
                    subset[subset["algorithm"] == algo_name]
                    .groupby("k")["measured_wall_time_s"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                if stats.empty:
                    continue
                ax.errorbar(
                    stats["k"],
                    stats["mean"],
                    yerr=stats["std"].fillna(0.0),
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    label=_display_algorithm_name(algo_name),
                )
            ax.set_title(f"{cluster_type} | {mode}")
            ax.set_xlabel("Requested GPUs (k)")
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("Measured wall-clock latency (s)")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_trigger_analysis_plot(df: pd.DataFrame, output_path: Path, cv_threshold: float) -> None:
    """Figure 2: Trigger reason breakdown and CV-vs-improvement scatter."""
    adaptive = df[df["algorithm"] == "BandPilot"].copy()
    if adaptive.empty:
        return
    adaptive = adaptive.dropna(subset=["bw_cv"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[1.1, 1.3])

    reason_stats = (
        adaptive.groupby(["k", "trigger_reason"]).size().unstack(fill_value=0).sort_index()
    )
    proportions = reason_stats.div(reason_stats.sum(axis=1), axis=0).fillna(0.0)
    bottom = np.zeros(len(proportions))
    for reason in proportions.columns:
        values = proportions[reason].to_numpy(dtype=float)
        axes[0].bar(proportions.index.astype(int), values, bottom=bottom, label=reason)
        bottom += values
    axes[0].set_ylabel("Fraction")
    axes[0].set_xlabel("Requested GPUs (k)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    colors = adaptive["pts_triggered"].fillna(False).map({True: "tab:red", False: "tab:blue"})
    axes[1].scatter(
        adaptive["bw_cv"],
        adaptive["pts_improvement_pct"].fillna(0.0),
        c=colors,
        alpha=0.75,
        s=28,
    )
    axes[1].axvline(cv_threshold, color="tab:gray", linestyle="--", linewidth=1.2)
    axes[1].axhline(3.0, color="tab:green", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("EHA candidate CV")
    axes[1].set_ylabel("PTS improvement (%)")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_bw_quality_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Figure 3: `BandPilot` bandwidth-loss quality."""
    adaptive = df[df["algorithm"] == "BandPilot"].copy()
    if adaptive.empty:
        return
    adaptive = adaptive.dropna(subset=["bw_loss_pct"])
    if adaptive.empty:
        return
    stats = adaptive.groupby("k")["bw_loss_pct"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(stats["k"], stats["mean"], marker="o", linewidth=2.0)
    ax.fill_between(
        stats["k"],
        (stats["mean"] - stats["std"].fillna(0.0)).clip(lower=0.0),
        stats["mean"] + stats["std"].fillna(0.0),
        alpha=0.2,
    )
    ax.axhline(3.0, linestyle="--", color="tab:red", linewidth=1.2)
    ax.set_xlabel("Requested GPUs (k)")
    ax.set_ylabel("Bandwidth loss (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_scaled_latency_plot(
    df: pd.DataFrame,
    output_path: Path,
    public_view_cfg: Dict[str, Any],
) -> None:
    """Figure 2: representative scaled-trace latency for the reviewer-facing public algorithms."""
    selected = _filter_representative_slice(df, public_view_cfg)
    if selected.empty:
        return
    clusters = sorted(selected["cluster_type"].unique().tolist())
    fig, axes = plt.subplots(
        1,
        len(clusters),
        figsize=(6.2 * max(1, len(clusters)), 4.8),
        squeeze=False,
        sharey=True,
    )
    representative_ks = [int(value) for value in public_view_cfg["representative_k_values"]]
    for col_idx, cluster_type in enumerate(clusters):
        ax = axes[0][col_idx]
        cluster_subset = selected[selected["cluster_type"] == cluster_type]
        for k in representative_ks:
            k_subset = cluster_subset[cluster_subset["k"].astype(int) == int(k)]
            if k_subset.empty:
                continue
            available_algorithms = [
                algo_name
                for algo_name in public_view_cfg["public_algorithms"]
                if algo_name in set(k_subset["algorithm"].unique())
            ]
            for algo_name in available_algorithms:
                algo_subset = k_subset[k_subset["algorithm"] == algo_name]
                stats = algo_subset.groupby("total_gpu").agg(
                    trace_p50=("measured_wall_time_s", lambda s: _percentile_or_nan(s, 50)),
                    trace_p95=("measured_wall_time_s", lambda s: _percentile_or_nan(s, 95)),
                ).reset_index()
                if stats.empty:
                    continue
                ax.plot(
                    stats["total_gpu"],
                    stats["trace_p50"],
                    marker="o",
                    linewidth=2.0,
                    label=f"{_display_algorithm_name(algo_name)} | k={k} p50",
                )
                ax.plot(
                    stats["total_gpu"],
                    stats["trace_p95"],
                    linestyle="--",
                    linewidth=1.3,
                    label=f"{_display_algorithm_name(algo_name)} | k={k} p95",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(cluster_type)
        ax.set_xlabel("Total GPUs")
        ax.grid(True, which="both", alpha=0.25)
        if col_idx == 0:
            ax.set_ylabel("Scaled-trace wall-clock (s)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_trigger_rate_plot(
    df: pd.DataFrame,
    output_path: Path,
    public_view_cfg: Dict[str, Any],
) -> None:
    """Figure 4: PTS usage rate vs scale for the reviewer-facing adaptive policy."""

    diagnostic_algorithm = _select_public_diagnostic_algorithm(public_view_cfg)
    if diagnostic_algorithm is None:
        return
    adaptive = df[df["algorithm"] == str(diagnostic_algorithm)].copy()
    if adaptive.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    grouped = (
        adaptive.groupby(["total_gpu", "avail_ratio", "inter_pod_factor"])["hu_pts_usage_rate"]
        .mean()
        .reset_index()
    )
    for (avail_ratio, inter_pod_factor), subset in grouped.groupby(["avail_ratio", "inter_pod_factor"]):
        label = f"avail={avail_ratio:.1f}, factor={inter_pod_factor:.1f}"
        ax.plot(subset["total_gpu"], subset["hu_pts_usage_rate"] * 100.0, marker="o", linewidth=1.8, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total GPUs")
    ax.set_ylabel("PTS usage rate (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_inference_scaling_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Auxiliary plot: predictor latency scaling vs node count for each cluster model."""
    if df.empty:
        return
    e2e_p50_col = "e2e_p50_ms" if "e2e_p50_ms" in df.columns else "p50_ms"
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    grouped = df.groupby("cluster_type") if "cluster_type" in df.columns else [("default", df)]
    for cluster_type, subset in grouped:
        ordered = subset.sort_values("node_count")
        ax.plot(
            ordered["node_count"],
            ordered[e2e_p50_col],
            marker="o",
            linewidth=2.0,
            label=f"{cluster_type} e2e p50",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Node count")
    ax.set_ylabel("Predictor latency (ms)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_extrapolation_plot(
    df: pd.DataFrame,
    output_path: Path,
    public_view_cfg: Dict[str, Any],
) -> None:
    """Figure 3: representative synthesized latency bound for the reviewer-facing public algorithms."""
    selected = _filter_representative_slice(df, public_view_cfg)
    if selected.empty:
        return
    clusters = sorted(selected["cluster_type"].unique().tolist())
    fig, axes = plt.subplots(
        1,
        len(clusters),
        figsize=(6.2 * max(1, len(clusters)), 4.8),
        squeeze=False,
        sharey=True,
    )
    representative_ks = [int(value) for value in public_view_cfg["representative_k_values"]]
    for col_idx, cluster_type in enumerate(clusters):
        ax = axes[0][col_idx]
        cluster_subset = selected[selected["cluster_type"] == cluster_type]
        for k in representative_ks:
            k_subset = cluster_subset[cluster_subset["k"].astype(int) == int(k)]
            if k_subset.empty:
                continue
            available_algorithms = [
                algo_name
                for algo_name in public_view_cfg["public_algorithms"]
                if algo_name in set(k_subset["algorithm"].unique())
            ]
            for algo_name in available_algorithms:
                algo_subset = k_subset[k_subset["algorithm"] == algo_name]
                stats = algo_subset.groupby("total_gpu").agg(
                    synth_p50=("synthesized_wall_time_p50_s", lambda s: _percentile_or_nan(s, 50)),
                    synth_p95=("synthesized_wall_time_p95_s", lambda s: _percentile_or_nan(s, 95)),
                ).reset_index()
                if stats.empty:
                    continue
                ax.plot(
                    stats["total_gpu"],
                    stats["synth_p50"],
                    marker="o",
                    linewidth=2.0,
                    label=f"{_display_algorithm_name(algo_name)} | k={k} p50",
                )
                ax.plot(
                    stats["total_gpu"],
                    stats["synth_p95"],
                    linestyle="--",
                    linewidth=1.3,
                    label=f"{_display_algorithm_name(algo_name)} | k={k} p95",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(cluster_type)
        ax.set_xlabel("Total GPUs")
        ax.grid(True, which="both", alpha=0.25)
        if col_idx == 0:
            ax.set_ylabel("Synthesized latency bound (s)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_quality_latency_pareto(df: pd.DataFrame, output_path: Path) -> None:
    """Show latency-vs-quality tradeoff for the reviewer-facing narrative."""
    adaptive = df[df["algorithm"] == "BandPilot"].copy()
    if adaptive.empty or "bw_loss_pct" not in adaptive.columns:
        return
    adaptive = adaptive.dropna(subset=["bw_loss_pct"])
    if adaptive.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(
        adaptive["measured_wall_time_s"],
        adaptive["bw_loss_pct"],
        c=adaptive["pts_triggered"].fillna(False).map({True: "tab:red", False: "tab:blue"}),
        alpha=0.75,
        s=26,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Measured wall-clock latency (s)")
    ax.set_ylabel("Bandwidth loss (%)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_latency_summary_table(
    tier1_results: pd.DataFrame,
    tier2_results: pd.DataFrame,
    tier3_results: pd.DataFrame,
    public_view_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Build a reviewer-facing summary table for the primary public algorithm."""
    rows: List[Dict[str, Any]] = []
    representative_mode = public_view_cfg["representative_contention_mode"]
    representative_avail = float(public_view_cfg["representative_avail_ratio"])
    representative_factor = float(public_view_cfg["representative_inter_pod_factor"])
    representative_ks = [int(value) for value in public_view_cfg["representative_k_values"]]
    primary_algorithms = list(public_view_cfg["public_algorithms"])
    upper_bound_algorithm = public_view_cfg.get("upper_bound_algorithm")
    if upper_bound_algorithm in set(primary_algorithms):
        upper_bound_algorithm = None

    public_tier1 = _filter_algorithm_rows(tier1_results, primary_algorithms)
    public_tier2 = _filter_algorithm_rows(tier2_results, primary_algorithms)
    public_tier3 = _filter_algorithm_rows(tier3_results, primary_algorithms)
    cluster_types = sorted(
        set(public_tier1.get("cluster_type", pd.Series(dtype=str)).dropna().tolist())
        | set(public_tier2.get("cluster_type", pd.Series(dtype=str)).dropna().tolist())
        | set(public_tier3.get("cluster_type", pd.Series(dtype=str)).dropna().tolist())
    )
    if not cluster_types:
        return pd.DataFrame()

    observed_scaled_gpu = (
        int(max(public_tier2["total_gpu"].tolist()))
        if (not public_tier2.empty and "total_gpu" in public_tier2.columns)
        else None
    )
    observed_synth_gpu = (
        int(max(public_tier3["total_gpu"].tolist()))
        if (not public_tier3.empty and "total_gpu" in public_tier3.columns)
        else None
    )

    for cluster_type in cluster_types:
        for algorithm_name in primary_algorithms:
            public_tier1_algo = _filter_algorithm_rows(public_tier1, [algorithm_name])
            public_tier2_algo = _filter_algorithm_rows(public_tier2, [algorithm_name])
            public_tier3_algo = _filter_algorithm_rows(public_tier3, [algorithm_name])

            upper_tier1_algo = _filter_algorithm_rows(tier1_results, [upper_bound_algorithm])
            upper_tier2_algo = _filter_algorithm_rows(tier2_results, [upper_bound_algorithm])
            upper_tier3_algo = _filter_algorithm_rows(tier3_results, [upper_bound_algorithm])

            for k in representative_ks:
                real_subset = public_tier1_algo[
                    (public_tier1_algo["cluster_type"] == cluster_type)
                    & (public_tier1_algo["k"].astype(int) == int(k))
                    & (public_tier1_algo["contention_mode"] == representative_mode)
                ]
                if not real_subset.empty:
                    upper_real_subset = upper_tier1_algo[
                        (upper_tier1_algo["cluster_type"] == cluster_type)
                        & (upper_tier1_algo["k"].astype(int) == int(k))
                        & (upper_tier1_algo["contention_mode"] == representative_mode)
                    ]
                    row = {
                        "Cluster": cluster_type,
                        "Algorithm": _display_algorithm_name(algorithm_name),
                        "Evidence": str(real_subset["evidence_type"].dropna().iloc[0]) if "evidence_type" in real_subset.columns else "measured",
                        "Scale (GPUs)": 32,
                        "k": int(k),
                        "Representative Slice": f"mode={representative_mode}",
                        "Latency p50 (s)": _percentile_or_nan(real_subset["measured_wall_time_s"], 50),
                        "Latency p95 (s)": _percentile_or_nan(real_subset["measured_wall_time_s"], 95),
                        "Envelope p95 (s)": _build_envelope_p95(
                            public_tier1_algo,
                            "measured_wall_time_s",
                            cluster_type,
                            32,
                            k,
                            public_view_cfg,
                            algorithm_names=[algorithm_name],
                        ),
                        "Predictor Calls p50": _percentile_or_nan(real_subset["predictor_calls"], 50),
                        "Predictor Time p50 (s)": _percentile_or_nan(real_subset["predictor_time_s"], 50),
                        "Non-Predictor Time p50 (s)": _percentile_or_nan(
                            real_subset["non_predictor_search_time_s"],
                            50,
                        ),
                        "PTS Usage Rate (%)": _mean_percent_or_nan(real_subset["hu_pts_usage_rate"]),
                    }
                    row.update(
                        _build_upper_bound_fields(
                            upper_real_subset,
                            algorithm_name=upper_bound_algorithm,
                            p50_value_column="measured_wall_time_s",
                            envelope_df=upper_tier1_algo,
                            cluster_type=cluster_type,
                            total_gpu=32,
                            k=k,
                            public_view_cfg=public_view_cfg,
                        )
                    )
                    rows.append(row)

                if observed_scaled_gpu is not None:
                    scaled_subset = public_tier2_algo[
                        (public_tier2_algo["cluster_type"] == cluster_type)
                        & (public_tier2_algo["total_gpu"].astype(int) == int(observed_scaled_gpu))
                        & (public_tier2_algo["k"].astype(int) == int(k))
                        & (public_tier2_algo["contention_mode"] == representative_mode)
                        & _match_float_series(public_tier2_algo["avail_ratio"], representative_avail)
                        & _match_float_series(public_tier2_algo["inter_pod_factor"], representative_factor)
                    ]
                    if not scaled_subset.empty:
                        upper_scaled_subset = upper_tier2_algo[
                            (upper_tier2_algo["cluster_type"] == cluster_type)
                            & (upper_tier2_algo["total_gpu"].astype(int) == int(observed_scaled_gpu))
                            & (upper_tier2_algo["k"].astype(int) == int(k))
                            & (upper_tier2_algo["contention_mode"] == representative_mode)
                            & _match_float_series(upper_tier2_algo["avail_ratio"], representative_avail)
                            & _match_float_series(upper_tier2_algo["inter_pod_factor"], representative_factor)
                        ]
                        row = {
                            "Cluster": cluster_type,
                            "Algorithm": _display_algorithm_name(algorithm_name),
                            "Evidence": str(scaled_subset["evidence_type"].dropna().iloc[0]) if "evidence_type" in scaled_subset.columns else "simulated",
                            "Scale (GPUs)": int(observed_scaled_gpu),
                            "k": int(k),
                            "Representative Slice": (
                                f"mode={representative_mode}, "
                                f"avail={representative_avail:.1f}, factor={representative_factor:.1f}"
                            ),
                            "Latency p50 (s)": _percentile_or_nan(scaled_subset["measured_wall_time_s"], 50),
                            "Latency p95 (s)": _percentile_or_nan(scaled_subset["measured_wall_time_s"], 95),
                            "Envelope p95 (s)": _build_envelope_p95(
                                public_tier2_algo,
                                "measured_wall_time_s",
                                cluster_type,
                                int(observed_scaled_gpu),
                                k,
                                public_view_cfg,
                                algorithm_names=[algorithm_name],
                            ),
                            "Predictor Calls p50": _percentile_or_nan(scaled_subset["predictor_calls"], 50),
                            "Predictor Time p50 (s)": _percentile_or_nan(scaled_subset["predictor_time_s"], 50),
                            "Non-Predictor Time p50 (s)": _percentile_or_nan(
                                scaled_subset["non_predictor_search_time_s"],
                                50,
                            ),
                            "PTS Usage Rate (%)": _mean_percent_or_nan(scaled_subset["hu_pts_usage_rate"]),
                        }
                        row.update(
                            _build_upper_bound_fields(
                                upper_scaled_subset,
                                algorithm_name=upper_bound_algorithm,
                                p50_value_column="measured_wall_time_s",
                                envelope_df=upper_tier2_algo,
                                cluster_type=cluster_type,
                                total_gpu=int(observed_scaled_gpu),
                                k=k,
                                public_view_cfg=public_view_cfg,
                            )
                        )
                        rows.append(row)

                if observed_synth_gpu is not None:
                    synth_subset = public_tier3_algo[
                        (public_tier3_algo["cluster_type"] == cluster_type)
                        & (public_tier3_algo["total_gpu"].astype(int) == int(observed_synth_gpu))
                        & (public_tier3_algo["k"].astype(int) == int(k))
                        & (public_tier3_algo["contention_mode"] == representative_mode)
                        & _match_float_series(public_tier3_algo["avail_ratio"], representative_avail)
                        & _match_float_series(public_tier3_algo["inter_pod_factor"], representative_factor)
                    ]
                    if not synth_subset.empty:
                        upper_synth_subset = upper_tier3_algo[
                            (upper_tier3_algo["cluster_type"] == cluster_type)
                            & (upper_tier3_algo["total_gpu"].astype(int) == int(observed_synth_gpu))
                            & (upper_tier3_algo["k"].astype(int) == int(k))
                            & (upper_tier3_algo["contention_mode"] == representative_mode)
                            & _match_float_series(upper_tier3_algo["avail_ratio"], representative_avail)
                            & _match_float_series(upper_tier3_algo["inter_pod_factor"], representative_factor)
                        ]
                        predictor_component = (
                            synth_subset["scaled_predictor_calls"] * synth_subset["predictor_e2e_p50_ms"] / 1000.0
                        )
                        row = {
                            "Cluster": cluster_type,
                            "Algorithm": _display_algorithm_name(algorithm_name),
                            "Evidence": "synthesized",
                            "Scale (GPUs)": int(observed_synth_gpu),
                            "k": int(k),
                            "Representative Slice": (
                                f"mode={representative_mode}, "
                                f"avail={representative_avail:.1f}, factor={representative_factor:.1f}"
                            ),
                            "Latency p50 (s)": _percentile_or_nan(
                                synth_subset["synthesized_wall_time_p50_s"],
                                50,
                            ),
                            "Latency p95 (s)": _percentile_or_nan(
                                synth_subset["synthesized_wall_time_p95_s"],
                                95,
                            ),
                            "Envelope p95 (s)": _build_envelope_p95(
                                public_tier3_algo,
                                "synthesized_wall_time_p95_s",
                                cluster_type,
                                int(observed_synth_gpu),
                                k,
                                public_view_cfg,
                                algorithm_names=[algorithm_name],
                            ),
                            "Predictor Calls p50": _percentile_or_nan(
                                synth_subset["scaled_predictor_calls"],
                                50,
                            ),
                            "Predictor Time p50 (s)": _percentile_or_nan(predictor_component, 50),
                            "Non-Predictor Time p50 (s)": _percentile_or_nan(
                                synth_subset["scaled_non_predictor_search_time_s"],
                                50,
                            ),
                            "PTS Usage Rate (%)": _mean_percent_or_nan(synth_subset["hu_pts_usage_rate"]),
                        }
                        row.update(
                            _build_upper_bound_fields(
                                upper_synth_subset,
                                algorithm_name=upper_bound_algorithm,
                                p50_value_column="synthesized_wall_time_p50_s",
                                p95_value_column="synthesized_wall_time_p95_s",
                                envelope_df=upper_tier3_algo,
                                cluster_type=cluster_type,
                                total_gpu=int(observed_synth_gpu),
                                k=k,
                                public_view_cfg=public_view_cfg,
                            )
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


def _write_latency_summary_table(summary_df: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    """Persist the latency summary table as CSV and LaTeX."""
    if summary_df.empty:
        return
    summary_df.to_csv(csv_path, index=False)

    preferred_columns = [
        "Cluster",
        "Algorithm",
        "Evidence",
        "Scale (GPUs)",
        "k",
        "Representative Slice",
        "Latency p50 (s)",
        "Latency p95 (s)",
        "Envelope p95 (s)",
        "Predictor Calls p50",
        "Predictor Time p50 (s)",
        "Non-Predictor Time p50 (s)",
        "PTS Usage Rate (%)",
        "Upper-Bound Algorithm",
        "Upper-Bound Latency p50 (s)",
        "Upper-Bound Latency p95 (s)",
        "Upper-Bound Envelope p95 (s)",
    ]
    available_columns = [column for column in preferred_columns if column in summary_df.columns]
    table_df = summary_df[available_columns].copy()
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(table_df.to_latex(index=False, float_format=lambda value: f"{value:.3f}"))


def _resolve_stage_cfg(benchmark_cfg: dict, new_key: str, legacy_key: Optional[str] = None) -> dict:
    """Prefer the new reviewer-aligned stage key while accepting legacy tier names."""
    if new_key in benchmark_cfg:
        return dict(benchmark_cfg.get(new_key, {}))
    if legacy_key is not None and legacy_key in benchmark_cfg:
        logger.warning(
            "scalability_benchmark config uses legacy key `%s`; prefer `%s`",
            legacy_key,
            new_key,
        )
        return dict(benchmark_cfg.get(legacy_key, {}))
    return {}


def run_scalability_benchmark_suite(
    cluster_configs: List[dict],
    benchmark_cfg: dict,
    output_dir: Path,
    random_seed: int,
) -> Dict[str, pd.DataFrame]:
    """Run the reviewer-aligned latency evidence pipeline and write artifacts."""
    ensure_directory(output_dir)
    clear_cache = bool(benchmark_cfg.get("clear_cache", False))
    if clear_cache:
        removed_paths = _clear_benchmark_artifacts(output_dir)
    else:
        removed_paths = []
    log_path = _configure_benchmark_logger(
        output_dir=output_dir,
        level=str(benchmark_cfg.get("log_level", "INFO")),
    )
    resume = bool(benchmark_cfg.get("resume", True))
    save_every_n_records = int(benchmark_cfg.get("save_every_n_records", 50))

    tier1_cfg = _resolve_stage_cfg(benchmark_cfg, "real_dispatch", "tier1")
    tier2_cfg = _resolve_stage_cfg(benchmark_cfg, "scaled_search", "tier2")
    tier3_cfg = _resolve_stage_cfg(benchmark_cfg, "predictor_profile", "tier3")
    tier4_cfg = _resolve_stage_cfg(benchmark_cfg, "latency_synthesis")
    public_view_cfg = _resolve_public_view_cfg(benchmark_cfg)
    global_algorithms = benchmark_cfg.get("algorithms")
    logger.info(
        "Benchmark suite start | output_dir=%s | log_path=%s | global_algorithms=%s | public_view=%s",
        output_dir,
        log_path,
        global_algorithms if global_algorithms is not None else ALGO_ORDER,
        public_view_cfg,
    )
    logger.info(
        "Benchmark suite caching | resume=%s | save_every_n_records=%s",
        resume,
        save_every_n_records,
    )
    if clear_cache:
        logger.info("Benchmark cache cleared | removed_files=%s", [str(path) for path in removed_paths])

    tier1_results = run_real_data_latency_benchmark(
        cluster_configs=cluster_configs,
        k_values=list(tier1_cfg.get("k_values", [4, 8, 12, 16, 20, 24, 28])),
        contention_modes=list(tier1_cfg.get("contention_modes", ["idle", "common", "intensive"])),
        repeat_num=int(tier1_cfg.get("repeat_num", 50)),
        random_seed=int(random_seed),
        algorithm_names=tier1_cfg.get("algorithms", global_algorithms),
        output_dir=output_dir,
        resume=resume,
        save_every_n_records=save_every_n_records,
    )
    if not tier1_results.empty and "cluster_type" in tier1_results.columns:
        for cluster_type, subset in tier1_results.groupby("cluster_type"):
            subset.to_csv(_build_cluster_output_path(output_dir, "tier1", cluster_type), index=False)
    elif tier1_results.empty:
        logger.info("Tier 1 returned no rows; skipping per-cluster Tier 1 CSV overwrite.")
    else:
        logger.warning(
            "Tier 1 results are missing `cluster_type`; skipping per-cluster Tier 1 CSV overwrite | columns=%s",
            tier1_results.columns.tolist(),
        )

    tier2_frames: List[pd.DataFrame] = []
    tier3_frames: List[pd.DataFrame] = []
    inference_frames: List[pd.DataFrame] = []
    target_gpu_counts = list(
        tier4_cfg.get(
            "target_gpu_counts",
            tier3_cfg.get("target_gpu_counts", [512, 1024, 2048, 4096]),
        )
    )
    inference_node_counts = list(tier3_cfg.get("inference_node_counts", [4, 8, 16, 32, 64, 128, 256, 512]))
    inference_repeats = int(tier3_cfg.get("inference_repeats", 100))
    scaled_repeat_num = int(tier2_cfg.get("repeat_num", 30))
    scaled_public_repeat_num = tier2_cfg.get("repeat_num_public_slice")

    for cluster_cfg in cluster_configs:
        cluster_type = str(cluster_cfg["cluster_type"])
        logger.info("Per-cluster scalability pipeline start | cluster=%s", cluster_type)
        inference_profile_path = _build_cluster_output_path(
            output_dir,
            "predictor_latency_profile",
            cluster_type,
        )
        inference_model = _load_predictor(
            cluster_cfg["model_path"],
            cluster_cfg["device"],
            cluster_cfg["model_cfg"],
        )
        inference_profile = profile_model_inference_scaling(
            model=inference_model,
            node_counts=inference_node_counts,
            device=cluster_cfg["device"],
            artifact_dir=cluster_cfg["artifact_dir"],
            cluster_type=cluster_type,
            repeats=inference_repeats,
            output_path=inference_profile_path,
            resume=resume,
        )
        if not inference_profile.empty:
            inference_profile.to_csv(inference_profile_path, index=False)
            inference_frames.append(inference_profile)

        scaled_output_path = _build_cluster_output_path(output_dir, "scaled_search", cluster_type)
        tier2_cluster = run_scaled_latency_benchmark(
            cluster_cfg=cluster_cfg,
            gpu_counts=list(tier2_cfg.get("gpu_counts", [64, 128, 256, 512, 1024])),
            k_values=list(tier2_cfg.get("k_values", [8, 16, 32, 64])),
            avail_ratios=list(tier2_cfg.get("avail_ratios", [0.5, 0.7, 0.9])),
            contention_modes=list(tier2_cfg.get("contention_modes", ["common", "intensive"])),
            inter_pod_factors=list(tier2_cfg.get("inter_pod_factors", [0.5, 0.7])),
            repeat_num=scaled_repeat_num,
            public_repeat_num=scaled_public_repeat_num,
            random_seed=int(random_seed),
            inference_profile=inference_profile,
            algorithm_names=tier2_cfg.get("algorithms", global_algorithms),
            output_path=scaled_output_path,
            resume=resume,
            save_every_n_records=save_every_n_records,
            public_view_cfg=public_view_cfg,
        )
        if not tier2_cluster.empty:
            tier2_cluster.to_csv(scaled_output_path, index=False)
            tier2_frames.append(tier2_cluster)

        synth_output_path = _build_cluster_output_path(output_dir, "synthesized_dispatch_latency", cluster_type)
        tier3_cluster = extrapolate_search_overhead(
            tier2_results=tier2_cluster,
            inference_profile=inference_profile,
            target_gpu_counts=target_gpu_counts,
            cluster_type=cluster_type,
        )
        if not tier3_cluster.empty:
            tier3_cluster.to_csv(synth_output_path, index=False)
            tier3_frames.append(tier3_cluster)
        logger.info(
            "Per-cluster scalability pipeline done | cluster=%s | scaled_rows=%s | synth_rows=%s | infer_rows=%s",
            cluster_type,
            len(tier2_cluster),
            len(tier3_cluster),
            len(inference_profile),
        )

    tier2_results = pd.concat(tier2_frames, ignore_index=True, sort=False) if tier2_frames else pd.DataFrame()
    tier3_results = pd.concat(tier3_frames, ignore_index=True, sort=False) if tier3_frames else pd.DataFrame()
    inference_profile = pd.concat(inference_frames, ignore_index=True, sort=False) if inference_frames else pd.DataFrame()

    _save_latency_real_plot(tier1_results, output_dir / PUBLIC_REAL_TRACE_PLOT_FILENAME, public_view_cfg)
    _save_scaled_latency_plot(tier2_results, output_dir / PUBLIC_SCALED_TRACE_PLOT_FILENAME, public_view_cfg)
    _save_inference_scaling_plot(inference_profile, output_dir / PUBLIC_PREDICTOR_PROFILE_PLOT_FILENAME)
    _save_extrapolation_plot(tier3_results, output_dir / PUBLIC_SYNTH_LATENCY_PLOT_FILENAME, public_view_cfg)
    _save_trigger_rate_plot(tier2_results, output_dir / PUBLIC_TRIGGER_RATE_PLOT_FILENAME, public_view_cfg)

    summary_df = _build_latency_summary_table(tier1_results, tier2_results, tier3_results, public_view_cfg)
    _write_latency_summary_table(
        summary_df,
        output_dir / PUBLIC_SUMMARY_FILENAME,
        output_dir / PUBLIC_SUMMARY_TEX_FILENAME,
    )
    logger.info(
        "Benchmark suite done | real_rows=%s | scaled_rows=%s | synth_rows=%s | inference_rows=%s",
        len(tier1_results),
        len(tier2_results),
        len(tier3_results),
        len(inference_profile),
    )

    return {
        "real_dispatch": tier1_results,
        "scaled_search": tier2_results,
        "latency_synthesis": tier3_results,
        "predictor_profile": inference_profile,
        "tier1": tier1_results,
        "tier2": tier2_results,
        "tier3": tier3_results,
        "inference_profile": inference_profile,
        "summary": summary_df,
    }
