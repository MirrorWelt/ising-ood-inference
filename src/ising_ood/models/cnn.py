"""Convolutional architectures: CNN-2 (basic) and CNN-3 (deeper, 'improved').

Both predict the 66-dimensional upper-triangular vector of the L=12
adjacency matrix from a (it_time, 12) magnetization trajectory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """CNN-2: a shallow two-convolution-block network."""

    def __init__(self, it_time: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=5, padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5, padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(100)
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(100, 144)
        self.fc2 = nn.Linear(144, 66)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, it_time, 12)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ImprovedCNNModel(nn.Module):
    """CNN-3: a deeper three-convolution-block network with BatchNorm."""

    def __init__(self, it_time: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 3), padding=(2, 1))
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1))
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 3), padding=(2, 1))
        self.bn3 = nn.BatchNorm2d(256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 66)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x