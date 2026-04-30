"""Linear bandwidth predictor baseline.

`LinearBandwidthRegressor` provides the lightweight `LinearBW` baseline used by
the compare pipeline and baseline suite. Checkpoints are trained by
`evaluation/baselines/train_linear_bw.py` and loaded through
`load_linear_bw_model(...)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils.helpers import build_artifact_filename, read_active_num_train_samples


class LinearBandwidthRegressor(nn.Module):
    """Single-layer regressor over compact bandwidth/count features."""

    def __init__(self) -> None:
        super().__init__()
        # Keep the baseline intentionally low-capacity.
        self.regressor = nn.Linear(10, 1)

    def _extract_features(
        self,
        x_bw: torch.Tensor,
        x_node_counts: torch.Tensor,
        x_total_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Extract fixed-size features from variable-length node tensors."""

        if x_bw.dim() == 3:
            x_bw = x_bw.squeeze(-1)
        if x_node_counts.dim() == 3:
            x_node_counts = x_node_counts.squeeze(-1)

        active_mask = (x_node_counts > 0).float()
        active_counts = active_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        bw_masked = x_bw * active_mask
        counts_float = x_node_counts.float()
        counts_masked = counts_float * active_mask

        # Aggregate bandwidth statistics across active nodes.
        bw_mean = bw_masked.sum(dim=1, keepdim=True) / active_counts
        bw_var = ((x_bw - bw_mean) ** 2 * active_mask).sum(dim=1, keepdim=True) / active_counts
        bw_std = torch.sqrt(bw_var + 1e-8)
        bw_max = x_bw.masked_fill(active_mask == 0, -1e9).max(dim=1, keepdim=True).values
        bw_min = x_bw.masked_fill(active_mask == 0, 1e9).min(dim=1, keepdim=True).values
        bw_max = torch.where(active_counts > 0, bw_max, torch.zeros_like(bw_max))
        bw_min = torch.where(active_counts > 0, bw_min, torch.zeros_like(bw_min))

        # Encode node-count spread and imbalance.
        node_mean = counts_masked.sum(dim=1, keepdim=True) / active_counts
        node_var = ((counts_float - node_mean) ** 2 * active_mask).sum(dim=1, keepdim=True) / active_counts
        node_std = torch.sqrt(node_var + 1e-8)
        node_max = counts_float.masked_fill(active_mask == 0, -1e9).max(dim=1, keepdim=True).values
        node_min = counts_float.masked_fill(active_mask == 0, 1e9).min(dim=1, keepdim=True).values
        node_max = torch.where(active_counts > 0, node_max, torch.zeros_like(node_max))
        node_min = torch.where(active_counts > 0, node_min, torch.zeros_like(node_min))
        imbalance = node_max - node_min

        total_count_feature = x_total_counts.float().view(-1, 1)

        return torch.cat(
            [
                bw_mean,
                bw_std,
                bw_max,
                bw_min,
                active_counts,
                node_mean,
                node_std,
                node_max,
                imbalance,
                total_count_feature,
            ],
            dim=1,
        )

    def forward(
        self,
        x_bw: torch.Tensor,
        x_node_counts: torch.Tensor,
        x_total_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return the prediction using the compare-compatible output key."""

        features = self._extract_features(x_bw, x_node_counts, x_total_counts)
        prediction = self.regressor(features)
        return {"final_bandwidth": prediction}


def _infer_num_train_samples(artifact_dir: Path) -> int:
    """Infer the sample-size suffix used by saved LinearBW artifacts."""

    try:
        return int(read_active_num_train_samples(artifact_dir))
    except (FileNotFoundError, ValueError):
        candidates = sorted(artifact_dir.glob("bandwidth_predictor_ns*.pth"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Cannot infer LinearBW checkpoint version under {artifact_dir}; "
                "provide an explicit checkpoint path or keep one active model marker."
            )
        stem = candidates[0].stem
        try:
            return int(stem.split("_ns")[-1])
        except (IndexError, ValueError) as exc:
            raise FileNotFoundError(
                f"Cannot parse num_train_samples from {candidates[0].name}"
            ) from exc


def resolve_linear_bw_model_path(
    *,
    cluster_type: str,
    model_root: Path,
    num_train_samples: Optional[int] = None,
) -> Path:
    """Resolve the artifact path for a `LinearBW` checkpoint."""

    artifact_dir = Path(model_root) / str(cluster_type)
    resolved_num_train_samples = (
        int(num_train_samples) if num_train_samples is not None else _infer_num_train_samples(artifact_dir)
    )
    return artifact_dir / build_artifact_filename(
        "bandwidth_predictor",
        resolved_num_train_samples,
        ".pth",
    )


def load_linear_bw_model(
    *,
    model_path: Path,
    device: torch.device,
) -> Tuple[LinearBandwidthRegressor, Path]:
    """Load a `LinearBW` checkpoint and return the model plus artifact directory."""

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"LinearBW checkpoint not found: {model_path}. "
            "Run evaluation.baselines.train_linear_bw first."
        )

    model = LinearBandwidthRegressor()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, Path(model_path).parent
