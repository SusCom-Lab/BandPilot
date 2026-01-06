"""Model training utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch import optim

from data_process.dataloader import (
    get_group_data_loader,
    get_group_test_loader,
    get_simple_group_data_loader,
    get_simple_group_test_loader,
)
from data_process.dataset import (
    get_balanced_train_dataset,
    get_random_train_dataset,
    get_simple_balanced_train_dataset,
)
from core.bandwidth import prepare_model_inputs
from models.bandwidth_predictor import BandwidthPredictor
from models.simple_predictor import SimpleBandwidthPredictor
from training.evaluator import evaluate_model, evaluate_simple_model
from utils.helpers import build_artifact_filename, record_active_num_train_samples


def _print_samples(title: str, gpu_configs, bandwidths, max_samples: int = 10) -> None:
    count = min(max_samples, len(gpu_configs))
    print(f"\n===== {title} (raw) =====")
    if count == 0:
        print("(no samples)")
        return
    for idx in range(count):
        print(f"Sample {idx + 1}: GPU config={gpu_configs[idx].tolist()}, bandwidth={bandwidths[idx]:.2f}")


def _print_processed_group_samples(
    title: str,
    gpu_configs,
    bandwidths,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
    max_samples: int = 5,
) -> None:
    """Print processed samples aligned with the training DataLoader (x_bws, x_node_counts, x_total_counts, y)."""
    if len(gpu_configs) == 0:
        print(f"\n===== {title} (processed examples) =====")
        print("(no samples)")
        return

    # Mirror the input construction and normalization logic used in get_group_data_loader
    part_bws, node_counts, total_counts = prepare_model_inputs(
        gpu_configs, total_gpu, gpu_bw_dict_list, switch_config, training_data_path
    )

    bw_scaler = StandardScaler()
    part_bws_reshaped = part_bws.reshape(-1, 1)
    bw_scaler.fit(part_bws_reshaped)
    scaled_part_bws = bw_scaler.transform(part_bws_reshaped).reshape(part_bws.shape)

    total_scaler = StandardScaler()
    total_scaler.fit(total_counts)
    scaled_total_counts = total_scaler.transform(total_counts)

    y_scaler = StandardScaler()
    y_scaler.fit(bandwidths.reshape(-1, 1))
    scaled_targets = y_scaler.transform(bandwidths.reshape(-1, 1)).flatten()

    count = min(max_samples, len(gpu_configs))
    print(f"\n===== {title} (processed examples, model input tensors) =====")
    for idx in range(count):
        print(f"Sample {idx + 1}: GPU config={gpu_configs[idx].tolist()}, bandwidth={bandwidths[idx]:.2f}")
        print(f"    x_bws[{idx}] = {scaled_part_bws[idx].tolist()}")
        print(f"    x_node_counts[{idx}] = {node_counts[idx].tolist()}")
        print(f"    x_total_counts[{idx}] = {scaled_total_counts[idx].tolist()}")
        print(f"    y[{idx}] = {scaled_targets[idx]:.4f}")


def _print_progress(tag: str, epoch: int, num_epochs: int, train_loss: float, val_loss: float) -> None:
    bar_len = 30
    progress = (epoch + 1) / num_epochs
    filled = int(progress * bar_len)
    bar = "█" * filled + "-" * (bar_len - filled)
    print(
        f"[{tag}] [{bar}] {progress*100:5.1f}% | Epoch {epoch + 1}/{num_epochs} | "
        f"Train={train_loss:.4f} | Val={val_loss:.4f}"
    )


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    lambda_ewc: float = 1.0,
) -> Tuple[torch.nn.Module, float]:
    """Generic training loop for the full model."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_bws, x_node_counts, x_total_counts, y_batch in train_loader:
            x_bws = x_bws.to(device)
            x_node_counts = x_node_counts.to(device)
            x_total_counts = x_total_counts.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_bws, x_node_counts, x_total_counts)["final_bandwidth"].view(-1)
            loss = F.smooth_l1_loss(outputs, y_batch)
            if hasattr(model, "ewc_loss"):
                loss = loss + model.ewc_loss(lambda_ewc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x_bws.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_bws, x_node_counts, x_total_counts, y_batch in val_loader:
                x_bws = x_bws.to(device)
                x_node_counts = x_node_counts.to(device)
                x_total_counts = x_total_counts.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_bws, x_node_counts, x_total_counts)["final_bandwidth"].view(-1)
                loss = F.smooth_l1_loss(outputs, y_batch)
                val_loss += loss.item() * x_bws.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        if epoch % 50 == 0:
            _print_progress("FullModel", epoch, num_epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def train_simple_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
) -> Tuple[torch.nn.Module, float]:
    """Training loop for the Simple model."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_bws, y_batch in train_loader:
            x_bws = x_bws.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_bws)["final_bandwidth"].view(-1)
            loss = F.smooth_l1_loss(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x_bws.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_bws, y_batch in val_loader:
                x_bws = x_bws.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_bws)["final_bandwidth"].view(-1)
                loss = F.smooth_l1_loss(outputs, y_batch)
                val_loss += loss.item() * x_bws.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        _print_progress("SimpleModel", epoch, num_epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def model_train_pipeline(
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
    artifact_dir: Path,
    device: torch.device,
    config: dict,
) -> Tuple[float, float, Path]:
    """Full training pipeline that returns evaluation metrics."""
    num_train_samples = config["training"]["num_train_samples"]
    gpu_train, bw_train = get_balanced_train_dataset(
        num_samples=num_train_samples,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
    )
    _print_samples("Training dataset preview", gpu_train, bw_train)
    _print_processed_group_samples(
        "Training dataset preview",
        gpu_train,
        bw_train,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
    )
    train_loader, val_loader = get_group_data_loader(
        gpu_train,
        bw_train,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
        artifact_dir,
        num_train_samples=num_train_samples,
        batch_size=config["training"]["batch_size"],
    )

    model = BandwidthPredictor(
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    )

    model, _ = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=config["training"]["num_epochs"],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["patience"],
        lambda_ewc=config["training"]["lambda_ewc"],
    )

    gpu_test, bw_test = get_random_train_dataset(
        num_samples=config["training"]["num_test_samples"],
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
    )
    _print_samples("Test dataset preview", gpu_test, bw_test)
    _print_processed_group_samples(
        "Test dataset preview",
        gpu_test,
        bw_test,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
    )
    test_loader = get_group_test_loader(
        num_samples=len(gpu_test),
        total_gpu=total_gpu,
        gpu_configs=gpu_test,
        bandwidth_targets=bw_test,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
        artifact_dir=artifact_dir,
        num_train_samples=num_train_samples,
    )

    mse, mae = evaluate_model(
        model,
        test_loader,
        device,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
        artifact_dir,
        num_train_samples=num_train_samples,
    )
    model_path = artifact_dir / build_artifact_filename(
        "bandwidth_predictor", num_train_samples, ".pth"
    )
    torch.save(model.state_dict(), model_path)
    record_active_num_train_samples(artifact_dir, num_train_samples)
    return mse, mae, model_path


def simple_model_train_pipeline(
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
    artifact_dir: Path,
    device: torch.device,
    config: dict,
) -> Tuple[float, float, Path]:
    """Training pipeline for the Simple model."""
    num_train_samples = config["training"]["num_train_samples"]
    gpu_train, bw_train = get_simple_balanced_train_dataset(
        num_samples=num_train_samples,
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
    )
    _print_samples("Training dataset preview", gpu_train, bw_train)
    train_loader, val_loader = get_simple_group_data_loader(
        gpu_train,
        bw_train,
        artifact_dir,
        num_train_samples=num_train_samples,
        batch_size=config["training"]["batch_size"],
    )

    model = SimpleBandwidthPredictor(
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    )

    model, _ = train_simple_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=config["training"]["num_epochs"],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["patience"],
    )

    gpu_test, bw_test = get_simple_balanced_train_dataset(
        num_samples=config["training"]["num_test_samples"],
        total_gpu=total_gpu,
        gpu_bw_dict_list=gpu_bw_dict_list,
        switch_config=switch_config,
        training_data_path=training_data_path,
    )
    _print_samples("Test dataset preview", gpu_test, bw_test)
    test_loader = get_simple_group_test_loader(
        num_samples=len(gpu_test),
        gpu_configs=gpu_test,
        targets=bw_test,
        artifact_dir=artifact_dir,
        num_train_samples=num_train_samples,
    )

    mse, mae = evaluate_simple_model(
        model, test_loader, device, artifact_dir, num_train_samples=num_train_samples
    )
    model_path = artifact_dir / build_artifact_filename(
        "simple_bandwidth_predictor", num_train_samples, ".pth"
    )
    torch.save(model.state_dict(), model_path)
    record_active_num_train_samples(artifact_dir, num_train_samples)
    return mse, mae, model_path
