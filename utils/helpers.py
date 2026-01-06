"""General utility helpers."""
from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> None:
    """Ensure directory exists (mkdir -p)."""
    path.mkdir(parents=True, exist_ok=True)


def build_artifact_filename(base_name: str, num_train_samples: int, ext: str) -> str:
    """
    Build a filename with sample-count suffix to avoid mixing artifacts of different training sizes.

    Args:
        base_name: Base name without extension, e.g. "bandwidth_predictor".
        num_train_samples: Number of training samples to distinguish models/scalers.
        ext: Extension including dot, e.g. ".pth" or ".pkl".
    """
    return f"{base_name}_ns{num_train_samples}{ext}"


def record_active_num_train_samples(artifact_dir: Path, num_train_samples: int) -> None:
    """
    Record the active num_train_samples in artifact_dir for later explicit loading.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    marker = artifact_dir / "active_num_train_samples.txt"
    marker.write_text(str(num_train_samples), encoding="utf-8")


def read_active_num_train_samples(artifact_dir: Path) -> int:
    """
    Read the last recorded num_train_samples from artifact_dir; raise FileNotFoundError if missing.
    """
    marker = artifact_dir / "active_num_train_samples.txt"
    content = marker.read_text(encoding="utf-8").strip()
    return int(content)

