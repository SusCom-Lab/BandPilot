"""模型推理与评估工具。"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.bandwidth import SwitchBandwidthConfig, calculate_bandwidth_values

logger = logging.getLogger(__name__)


def _load_scaler(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def predict_with_model(
    model,
    part_bws: np.ndarray,
    node_counts: np.ndarray,
    total_counts: np.ndarray,
    device: torch.device,
    artifact_dir: Path,
) -> np.ndarray:
    """对多节点配置使用模型进行预测，单节点直接返回局部带宽。"""
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
        bw_scaler = _load_scaler(artifact_dir / "bw_scaler.pkl")
        total_scaler = _load_scaler(artifact_dir / "total_counts_scaler.pkl")
        y_scaler = _load_scaler(artifact_dir / "y_scaler.pkl")

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
            preds[idx] = pred

    return preds


def evaluate_model(
    model,
    test_loader,
    device: torch.device,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
    artifact_dir: Path,
) -> Tuple[float, float]:
    """评估主模型性能。
    
    评估过程中会收集所有预测值和真实值，并在评估结束后保存到 pickle 文件中，
    文件保存在 artifact_dir 目录下，文件名格式为 test_loader_data_{样本数}Data.pkl
    """
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    total_samples = 0

    # 用于收集所有预测值和真实值
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    y_scaler = _load_scaler(artifact_dir / "y_scaler.pkl")

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

            # 记录所有预测值和真实值（使用逆变换后的值）
            all_preds.append(pred_tensor.detach().cpu())
            all_targets.append(target_tensor.detach().cpu())

    mse_avg = mse_total / total_samples
    mae_avg = mae_total / total_samples

    # 保存测试集数据和预测结果
    all_preds_np = torch.cat(all_preds, dim=0).numpy()
    all_targets_np = torch.cat(all_targets, dim=0).numpy()
    
    # 确保 artifact_dir 存在
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名：test_loader_data_{样本数}Data.pkl
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
    
    logger.info(f"测试集数据及预测结果已保存到 {save_path}")
    logger.info(f"测试集样本数: {num_samples}, MSE: {mse_avg:.4f}, MAE: {mae_avg:.4f}")

    return mse_avg, mae_avg


def evaluate_simple_model(
    model,
    test_loader,
    device: torch.device,
    artifact_dir: Path,
) -> Tuple[float, float]:
    """评估简化模型。"""
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    total_samples = 0

    y_scaler = _load_scaler(artifact_dir / "simple_y_scaler.pkl")

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

