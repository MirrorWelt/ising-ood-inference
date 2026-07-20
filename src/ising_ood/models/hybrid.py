"""Hybrid CNN-Transformer: local convolutional features feeding a
Transformer encoder for long-range temporal dependence."""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridCNNTransformer(nn.Module):
    def __init__(self, it_time: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(it_time // 2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 66),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cnn = x.transpose(1, 2)          # (B, 12, it_time)
        x_cnn = self.cnn(x_cnn)            # (B, 128, it_time//2)
        x_cnn = x_cnn.transpose(1, 2)      # (B, it_time//2, 128)

        x_trans = self.transformer(x_cnn)
        x_pool = torch.mean(x_trans, dim=1)
        return self.classifier(x_pool)