"""带宽预测主模型。"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import AttentionPooling, PositionalEncoding


class BandwidthPredictor(nn.Module):
    """Transformer + 注意力池化的带宽预测模型。"""

    def __init__(
        self,
        input_dim: int = 1,
        node_count_embedding_dim: int = 8,
        total_count_feature_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bw_input_proj = nn.Linear(input_dim, hidden_dim)
        self.node_count_embed = nn.Linear(1, node_count_embedding_dim)
        self.combined_feature_proj = nn.Linear(hidden_dim + node_count_embedding_dim, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim + total_count_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.old_params: Dict[str, torch.Tensor] | None = None
        self.fisher_diag: Dict[str, torch.Tensor] | None = None

    def forward(
        self,
        x_bw: torch.Tensor,
        x_node_counts: torch.Tensor,
        x_total_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if x_bw.dim() == 2:
            x_bw = x_bw.unsqueeze(-1)
        if x_node_counts.dim() == 2:
            x_node_counts = x_node_counts.unsqueeze(-1)

        bw_embed = self.bw_input_proj(x_bw)
        node_embed = self.node_count_embed(x_node_counts.float())
        combined = torch.cat([bw_embed, node_embed], dim=-1)
        projected = F.relu(self.combined_feature_proj(combined))
        x = self.pos_encoder(projected)

        src_key_padding_mask = (torch.sum(x_bw, dim=-1) == 0)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        pooled = self.attn_pool(x)
        final_features = torch.cat([pooled, x_total_counts], dim=-1)
        final_bandwidth = self.prediction_head(final_features)
        return {"final_bandwidth": final_bandwidth}

    def save_old_params(self) -> None:
        """保存当前参数供EWC使用。"""
        self.old_params = {name: param.data.clone() for name, param in self.named_parameters()}

    def update_fisher(self, data_loader, device: torch.device) -> None:
        """估计Fisher对角近似。"""
        fisher = {name: torch.zeros_like(param) for name, param in self.named_parameters()}
        total_samples = 0

        for batch in data_loader:
            x_bws, x_node_counts, x_total_counts, y_batch = batch
            x_bws = x_bws.to(device)
            x_node_counts = x_node_counts.to(device)
            x_total_counts = x_total_counts.to(device)
            y_batch = y_batch.to(device)

            self.zero_grad()
            outputs = self(x_bws, x_node_counts, x_total_counts)
            loss = F.mse_loss(outputs["final_bandwidth"].view(-1), y_batch, reduction="mean")
            loss.backward()

            batch_size = x_bws.size(0)
            total_samples += batch_size

            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad.data.pow(2) * batch_size)

        for name in fisher:
            fisher[name] /= float(total_samples)

        self.fisher_diag = fisher

    def ewc_loss(self, lambda_ewc: float = 1.0) -> torch.Tensor:
        """Elastic Weight Consolidation 约束。"""
        if self.old_params is None or self.fisher_diag is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        ewc = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.named_parameters():
            if name in self.old_params and name in self.fisher_diag:
                ewc += torch.sum(self.fisher_diag[name] * (param - self.old_params[name]).pow(2))
        return lambda_ewc * ewc

