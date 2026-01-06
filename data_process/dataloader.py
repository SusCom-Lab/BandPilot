"""Data loaders and normalization logic."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from core.bandwidth import SwitchBandwidthConfig, prepare_model_inputs
from utils.helpers import build_artifact_filename


def _save_scaler(scaler: StandardScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(scaler, f)


def _load_scaler(path: Path) -> StandardScaler:
    with path.open("rb") as f:
        return pickle.load(f)


def get_group_data_loader(
    gpu_train: np.ndarray,
    bw_train: np.ndarray,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    artifact_dir: Path,
    num_train_samples: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training/validation DataLoaders with node features."""
    part_bws, node_counts, total_counts = prepare_model_inputs(
        gpu_train, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
    )

    bw_scaler = StandardScaler()
    part_bws_reshaped = part_bws.reshape(-1, 1)
    bw_scaler.fit(part_bws_reshaped)
    _save_scaler(
        bw_scaler,
        artifact_dir / build_artifact_filename("bw_scaler", num_train_samples, ".pkl"),
    )
    scaled_part_bws = bw_scaler.transform(part_bws_reshaped).reshape(part_bws.shape)

    total_scaler = StandardScaler()
    total_scaler.fit(total_counts)
    _save_scaler(
        total_scaler,
        artifact_dir
        / build_artifact_filename("total_counts_scaler", num_train_samples, ".pkl"),
    )
    scaled_total_counts = total_scaler.transform(total_counts)

    y_scaler = StandardScaler()
    y_scaler.fit(bw_train.reshape(-1, 1))
    _save_scaler(
        y_scaler,
        artifact_dir / build_artifact_filename("y_scaler", num_train_samples, ".pkl"),
    )
    scaled_targets = y_scaler.transform(bw_train.reshape(-1, 1)).flatten()

    dataset = TensorDataset(
        torch.tensor(scaled_part_bws, dtype=torch.float32),
        torch.tensor(node_counts, dtype=torch.long),
        torch.tensor(scaled_total_counts, dtype=torch.float32),
        torch.tensor(scaled_targets, dtype=torch.float32),
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def get_group_test_loader(
    num_samples: int,
    total_gpu: int,
    gpu_configs: np.ndarray,
    bandwidth_targets: np.ndarray,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    training_data_path: str,
    artifact_dir: Path,
    num_train_samples: int,
    batch_size: int = 100,
) -> DataLoader:
    """Build cross-node test DataLoader."""
    part_bws, node_counts, total_counts = prepare_model_inputs(
        gpu_configs, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
    )

    bw_scaler = _load_scaler(
        artifact_dir / build_artifact_filename("bw_scaler", num_train_samples, ".pkl")
    )
    total_scaler = _load_scaler(
        artifact_dir
        / build_artifact_filename("total_counts_scaler", num_train_samples, ".pkl")
    )
    y_scaler = _load_scaler(
        artifact_dir / build_artifact_filename("y_scaler", num_train_samples, ".pkl")
    )

    scaled_part_bws = bw_scaler.transform(part_bws.reshape(-1, 1)).reshape(part_bws.shape)
    scaled_total_counts = total_scaler.transform(total_counts)
    scaled_targets = y_scaler.transform(bandwidth_targets.reshape(-1, 1)).flatten()

    dataset = TensorDataset(
        torch.tensor(scaled_part_bws, dtype=torch.float32),
        torch.tensor(node_counts, dtype=torch.long),
        torch.tensor(scaled_total_counts, dtype=torch.float32),
        torch.tensor(scaled_targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_simple_group_data_loader(
    gpu_train: np.ndarray,
    bw_train: np.ndarray,
    artifact_dir: Path,
    num_train_samples: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Data loaders for the Simple model (train/val)."""
    bw_scaler = StandardScaler()
    bw_scaler.fit(gpu_train.reshape(-1, 1))
    _save_scaler(
        bw_scaler,
        artifact_dir
        / build_artifact_filename("simple_bw_scaler", num_train_samples, ".pkl"),
    )
    scaled_bws = bw_scaler.transform(gpu_train.reshape(-1, 1)).reshape(gpu_train.shape)

    y_scaler = StandardScaler()
    y_scaler.fit(bw_train.reshape(-1, 1))
    _save_scaler(
        y_scaler,
        artifact_dir
        / build_artifact_filename("simple_y_scaler", num_train_samples, ".pkl"),
    )
    scaled_targets = y_scaler.transform(bw_train.reshape(-1, 1)).flatten()

    dataset = TensorDataset(
        torch.tensor(scaled_bws, dtype=torch.float32),
        torch.tensor(scaled_targets, dtype=torch.float32),
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
    )


def get_simple_group_test_loader(
    num_samples: int,
    gpu_configs: np.ndarray,
    targets: np.ndarray,
    artifact_dir: Path,
    num_train_samples: int,
    batch_size: int = 100,
) -> DataLoader:
    """DataLoader for the Simple model test set."""
    bw_scaler = _load_scaler(
        artifact_dir
        / build_artifact_filename("simple_bw_scaler", num_train_samples, ".pkl")
    )
    y_scaler = _load_scaler(
        artifact_dir
        / build_artifact_filename("simple_y_scaler", num_train_samples, ".pkl")
    )

    scaled_bws = bw_scaler.transform(gpu_configs.reshape(-1, 1)).reshape(gpu_configs.shape)
    scaled_targets = y_scaler.transform(targets.reshape(-1, 1)).flatten()

    dataset = TensorDataset(
        torch.tensor(scaled_bws, dtype=torch.float32),
        torch.tensor(scaled_targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

