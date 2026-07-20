"""Pure Transformer-encoder architecture with learnable positional encoding
and attention-based global pooling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttentionPooling(nn.Module):
    """Learned attention pooling over the time axis."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, hidden_dim)
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (kept for completeness; the
    active TransformerModel below uses a learnable positional parameter
    instead, matching the original implementation)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :].transpose(0, 1)


class TransformerModel(nn.Module):
    """Transformer encoder with a learnable positional-encoding parameter."""

    def __init__(
        self,
        it_time: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(12, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.positional_encoding = nn.Parameter(torch.randn(1, it_time, d_model))

        encoder_layers = [
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(num_layers)
        ]
        self.transformer_encoder = nn.Sequential(*encoder_layers)

        self.global_pool = GlobalAttentionPooling(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 2),
            nn.Linear(d_model // 2, 66),
        )

        self.grad_clip_value = 1.0
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def clip_gradients(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.global_pool(x)
        return self.classifier(x)