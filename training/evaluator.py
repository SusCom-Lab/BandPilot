"""Model inference and evaluation utilities."""
from __future__ import annotations

import logging
import pickle
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values
from utils.helpers import build_artifact_filename, read_active_num_train_samples

logger = logging.getLogger(__name__)

_PREDICTION_ARTIFACT_KEYS = (
    "resolved_num_train_samples",
    "bw_scaler",
    "total_scaler",
    "y_scaler",
)


def compute_extra_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute R^2, MAPE (%), RMSE, MSE, MAE from prediction/target arrays.

    Returns dict with keys: r2, mape_percent, rmse, mse, mae.
    """
    errors = preds - targets
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    denom = np.clip(np.abs(targets), 1e-6, None)
    mape = float(np.mean(np.abs(errors) / denom) * 100.0)
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "r2": r2,
    }


class PredictionProfiler:
    """Simple accumulator to record predict_with_model latency."""

    __slots__ = ("total_time", "call_count", "sample_count")

    def __init__(self) -> None:
        self.total_time: float = 0.0

        self.call_count: int = 0
        self.sample_count: int = 0

    def add(self, duration: float, sample_count: int = 1) -> None:
        self.total_time += duration
        self.call_count += 1
        self.sample_count += int(sample_count)


_prediction_profiler_ctx: ContextVar[Optional[PredictionProfiler]] = ContextVar(
    "prediction_profiler_ctx", default=None
)


@contextmanager
def prediction_profiling_session():
    """Start a profiling session to record prediction latency for a single algorithm run."""
    profiler = PredictionProfiler()
    token = _prediction_profiler_ctx.set(profiler)
    try:
        yield profiler
    finally:
        _prediction_profiler_ctx.reset(token)


def _load_scaler(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _load_prediction_artifact_bundle(
    artifact_dir: Path, resolved_num_train_samples: int
) -> Dict[str, Any]:
    return {
        "resolved_num_train_samples": int(resolved_num_train_samples),
        "bw_scaler": _load_scaler(
            artifact_dir
            / build_artifact_filename("bw_scaler", resolved_num_train_samples, ".pkl")
        ),
        "total_scaler": _load_scaler(
            artifact_dir
            / build_artifact_filename(
                "total_counts_scaler", resolved_num_train_samples, ".pkl"
            )
        ),
        "y_scaler": _load_scaler(
            artifact_dir
            / build_artifact_filename("y_scaler", resolved_num_train_samples, ".pkl")
        ),
    }


def _ensure_num_train_samples(artifact_dir: Path, num_train_samples: Optional[int]) -> int:
    """
    Resolve the sample-count identifier to use.

    Prefer an explicit num_train_samples; if absent, attempt to infer from
    bw_scaler_ns*.pkl under artifact_dir. If multiple or unparsable, raise to
    avoid misuse.
    """
    if num_train_samples is not None:
        # Warn when an existing record differs from the provided value
        try:
            recorded = read_active_num_train_samples(artifact_dir)
            if recorded != num_train_samples:
                logger.warning(
                    "Provided num_train_samples=%s differs from recorded %s; using the provided value.",
                    num_train_samples,
                    recorded,
                )
        except FileNotFoundError:
            pass
        return num_train_samples

    try:
        return read_active_num_train_samples(artifact_dir)
    except (FileNotFoundError, ValueError):
        pass

    candidates = list(artifact_dir.glob("bw_scaler_ns*.pkl"))
    if len(candidates) == 1:
        stem = candidates[0].stem
        # Expected format: bw_scaler_ns{num}
        try:
            return int(stem.split("_ns")[-1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Cannot infer num_train_samples from filename {candidates[0].name}; please pass it explicitly."
            )

    raise ValueError(
        "Cannot infer num_train_samples; provide it explicitly "
        "(cause: bw_scaler_ns*.pkl missing or multiple versions found)."
    )


def preload_prediction_artifacts(
    artifact_dir: Path,
    num_train_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Load reusable predictor artifacts once for repeated inference calls."""
    resolved_num_train_samples = _ensure_num_train_samples(
        artifact_dir, num_train_samples
    )
    return _load_prediction_artifact_bundle(
        artifact_dir, resolved_num_train_samples
    )


def _resolve_preloaded_prediction_artifacts(
    preloaded_artifacts: Dict[str, Any],
    num_train_samples: Optional[int],
) -> Dict[str, Any]:
    missing_keys = [
        key for key in _PREDICTION_ARTIFACT_KEYS if key not in preloaded_artifacts
    ]
    if missing_keys:
        raise KeyError(
            "preloaded_artifacts missing keys: " + ", ".join(sorted(missing_keys))
        )
    resolved_num_train_samples = int(
        preloaded_artifacts["resolved_num_train_samples"]
    )
    if (
        num_train_samples is not None
        and int(num_train_samples) != resolved_num_train_samples
    ):
        raise ValueError(
            "num_train_samples does not match preloaded_artifacts "
            f"({num_train_samples} != {resolved_num_train_samples})"
        )
    return preloaded_artifacts


def predict_with_model(
    model,
    part_bws: np.ndarray,
    node_counts: np.ndarray,
    total_counts: np.ndarray,
    device: torch.device,
    artifact_dir: Path,
    num_train_samples: Optional[int] = None,
    preloaded_artifacts: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Predict bandwidth for multi-node configurations; single-node cases fall back to local bandwidth."""
    profiler = _prediction_profiler_ctx.get()
    start_time = time.perf_counter() if profiler is not None else None
    loaded_artifacts = (
        _resolve_preloaded_prediction_artifacts(
            preloaded_artifacts, num_train_samples
        )
        if preloaded_artifacts is not None
        else None
    )
    try:
        model.eval()
        preds = np.zeros(len(part_bws))
        multi_indices: List[int] = []
        multi_part_bws: List[np.ndarray] = []
        multi_node_counts: List[np.ndarray] = []
        multi_total_counts: List[np.ndarray] = []

        for idx, part_bw in enumerate(part_bws):
            nonzero = [j for j, value in enumerate(part_bw) if value > 0]
            if len(nonzero) == 1:
                preds[idx] = part_bw[nonzero[0]]
            else:
                multi_indices.append(idx)
                multi_part_bws.append(part_bw)
                multi_node_counts.append(node_counts[idx])
                multi_total_counts.append(total_counts[idx])

        if multi_part_bws:
            if loaded_artifacts is None:
                loaded_artifacts = preload_prediction_artifacts(
                    artifact_dir,
                    num_train_samples=num_train_samples,
                )

            bw_scaler = loaded_artifacts["bw_scaler"]
            total_scaler = loaded_artifacts["total_scaler"]
            y_scaler = loaded_artifacts["y_scaler"]

            multi_part_bws_np = np.array(multi_part_bws)
            multi_node_counts_np = np.array(multi_node_counts)
            multi_total_counts_np = np.array(multi_total_counts)

            num_samples, seq_len = multi_part_bws_np.shape
            scaled_part_bws = bw_scaler.transform(multi_part_bws_np.reshape(-1, 1)).reshape(num_samples, seq_len)
            scaled_total_counts = total_scaler.transform(multi_total_counts_np)

            t_bws = torch.tensor(scaled_part_bws, dtype=torch.float32, device=device)
            t_node_counts = torch.tensor(multi_node_counts_np, dtype=torch.long, device=device)
            t_total_counts = torch.tensor(scaled_total_counts, dtype=torch.float32, device=device)

            with torch.no_grad():
                outputs = model(t_bws, t_node_counts, t_total_counts)["final_bandwidth"].view(-1).cpu().numpy()
            preds_multi = y_scaler.inverse_transform(outputs.reshape(-1, 1)).flatten()

            for idx, pred in zip(multi_indices, preds_multi):
                preds[idx] = round(pred,0)
        return preds
    finally:
        if profiler is not None and start_time is not None:
            profiler.add(time.perf_counter() - start_time, sample_count=len(part_bws))


def evaluate_model(
    model,
    test_loader,
    device: torch.device,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    artifact_dir: Path,
    num_train_samples: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate the full model on a test loader.

    Collect all predictions/targets and persist them to a pickle file in
    artifact_dir named test_loader_data_{num_samples}Data.pkl.
    """
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    total_samples = 0
    
    # Collect all predictions and ground-truth values for later persistence.
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    resolved_num_train_samples = _ensure_num_train_samples(
        artifact_dir, num_train_samples
    )
    y_scaler = _load_scaler(
        artifact_dir
        / build_artifact_filename("y_scaler", resolved_num_train_samples, ".pkl")
    )

    with torch.no_grad():
        for x_bws, x_node_counts, x_total_counts, y_batch in test_loader:
            x_bws = x_bws.to(device)
            x_node_counts = x_node_counts.to(device)
            x_total_counts = x_total_counts.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_bws, x_node_counts, x_total_counts)["final_bandwidth"].view(-1)

            pred_np = outputs.cpu().numpy().reshape(-1, 1)
            target_np = y_batch.cpu().numpy().reshape(-1, 1)

            pred_inv = y_scaler.inverse_transform(pred_np).flatten()
            target_inv = y_scaler.inverse_transform(target_np).flatten()

            pred_tensor = torch.tensor(pred_inv, dtype=torch.float32, device=device)
            target_tensor = torch.tensor(target_inv, dtype=torch.float32, device=device)

            mse = F.mse_loss(pred_tensor, target_tensor, reduction="sum")
            mae = F.l1_loss(pred_tensor, target_tensor, reduction="sum")

            batch_size = x_bws.size(0)
            mse_total += mse.item()
            mae_total += mae.item()
            total_samples += batch_size
            
            # Record all predictions and targets (after inverse scaling) for later analysis.
            all_preds.append(pred_tensor.detach().cpu())
            all_targets.append(target_tensor.detach().cpu())

    mse_avg = mse_total / total_samples
    mae_avg = mae_total / total_samples
    
    # Persist the full test set data and prediction results for offline inspection.
    all_preds_np = torch.cat(all_preds, dim=0).numpy()
    all_targets_np = torch.cat(all_targets, dim=0).numpy()
    
    # Ensure artifact_dir exists
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename: test_loader_data_{num_samples}Data.pkl
    num_samples = len(all_preds_np)
    save_path = artifact_dir / f"test_loader_data_{num_samples}Data.pkl"
    
    with save_path.open("wb") as f:
        pickle.dump(
            {
                "preds": all_preds_np,
                "targets": all_targets_np,
            },
            f,
        )
    
    logger.info(f"Test set data and predictions saved to {save_path}")
    logger.info(f"Test samples: {num_samples}, MSE: {mse_avg:.4f}, MAE: {mae_avg:.4f}")

    return mse_avg, mae_avg


def evaluate_simple_model(
    model,
    test_loader,
    device: torch.device,
    artifact_dir: Path,
    num_train_samples: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate the simplified model that operates directly on part-bandwidth sequences."""
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    total_samples = 0

    resolved_num_train_samples = _ensure_num_train_samples(
        artifact_dir, num_train_samples
    )
    y_scaler = _load_scaler(
        artifact_dir
        / build_artifact_filename("simple_y_scaler", resolved_num_train_samples, ".pkl")
    )

    with torch.no_grad():
        for x_bws, y_batch in test_loader:
            x_bws = x_bws.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_bws)["final_bandwidth"].view(-1)

            pred_np = outputs.cpu().numpy().reshape(-1, 1)
            target_np = y_batch.cpu().numpy().reshape(-1, 1)

            pred_inv = y_scaler.inverse_transform(pred_np).flatten()
            target_inv = y_scaler.inverse_transform(target_np).flatten()

            pred_tensor = torch.tensor(pred_inv, dtype=torch.float32, device=device)
            target_tensor = torch.tensor(target_inv, dtype=torch.float32, device=device)

            mse = F.mse_loss(pred_tensor, target_tensor, reduction="sum")
            mae = F.l1_loss(pred_tensor, target_tensor, reduction="sum")

            batch_size = x_bws.size(0)
            mse_total += mse.item()
            mae_total += mae.item()
            total_samples += batch_size

    mse_avg = mse_total / total_samples
    mae_avg = mae_total / total_samples
    return mse_avg, mae_avg
