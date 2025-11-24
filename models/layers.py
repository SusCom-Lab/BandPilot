"""通用神经网络层。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Transformer位置编码。"""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x


class AttentionPooling(nn.Module):
    """自适应注意力池化。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = F.softmax(self.query(x), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), x)
        return context.squeeze(1)

