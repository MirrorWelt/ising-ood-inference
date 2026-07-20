"""Graph neural network: treats each time step as a graph node with a fixed
banded ("distance <= k") adjacency template, and message-passes over time."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPooling(nn.Module):
    """Global average pooling over the node (time-step) axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, hidden_dim)
        return torch.mean(x, dim=1)


class SimpleGCNLayer(nn.Module):
    """A minimal graph-convolution layer, structurally analogous to a CNN block."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        adj_norm = self._normalize_adj(adj)

        x = self.linear(x)
        x = torch.bmm(adj_norm, x)

        x = x.reshape(-1, x.size(-1))
        x = self.bn(x)
        x = x.reshape(batch_size, num_nodes, -1)
        x = F.relu(x)
        x = self.dropout(x)
        return x

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adj.shape
        device = adj.device
        eye = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        degree = adj.sum(dim=2, keepdim=True)
        return adj / (degree + 1e-10)


class GNNModel(nn.Module):
    """Graph neural network over the time axis with a fixed banded adjacency."""

    def __init__(self, it_time: int, hidden_dim: int = 128, gnn_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.it_time = it_time
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gnn_layers = nn.ModuleList(
            [SimpleGCNLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(gnn_layers)]
        )

        self.global_pool = GlobalAvgPooling()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 66),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def build_simple_graph(self, x: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Vectorized construction of a banded (|i - j| <= k) adjacency template."""
        batch_size, num_nodes, _ = x.shape
        device = x.device

        row_indices = torch.arange(num_nodes, device=device).view(-1, 1)
        col_indices = torch.arange(num_nodes, device=device).view(1, -1)
        distance_matrix = torch.abs(row_indices - col_indices)

        adj_template = (distance_matrix <= k).float()
        adj_template.fill_diagonal_(0)

        return adj_template.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        adj = self.build_simple_graph(x)

        x_flat = x.reshape(-1, 12)
        x_node = self.node_embedding(x_flat)
        x_node = x_node.view(batch_size, self.it_time, self.hidden_dim)

        for layer in self.gnn_layers:
            x_node = layer(x_node, adj)

        x_pool = self.global_pool(x_node)
        return self.classifier(x_pool)