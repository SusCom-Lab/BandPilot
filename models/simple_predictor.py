"""Simplified bandwidth prediction model."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.layers import AttentionPooling, PositionalEncoding


class SimpleBandwidthPredictor(nn.Module):
    """Simplified model that directly uses GPU masks as input."""

    def __init__(
        self,
        input_chunk_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_chunk_dim = input_chunk_dim
        self.input_projection = nn.Linear(input_chunk_dim, hidden_dim)
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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1, self.input_chunk_dim).float()
        x_projected = self.input_projection(x_reshaped)
        x_pos = self.pos_encoder(x_projected)
        src_key_padding_mask = (torch.sum(x_reshaped, dim=-1) == 0)
        encoded = self.transformer_encoder(x_pos, src_key_padding_mask=src_key_padding_mask)
        pooled = self.attn_pool(encoded)
        final_bandwidth = self.prediction_head(pooled)
        return {"final_bandwidth": final_bandwidth}

