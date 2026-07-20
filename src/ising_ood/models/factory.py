"""Model factory, tensor <-> 66-dim-vector conversion, and small training
utilities shared by every entry point (generation, training, evaluation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import CNNModel, ImprovedCNNModel
from .gnn import GNNModel
from .transformer import TransformerModel
from .hybrid import HybridCNNTransformer

_MODEL_REGISTRY = {
    "cnn": CNNModel,
    "improved_cnn": ImprovedCNNModel,
    "gnn": GNNModel,
    "transformer": TransformerModel,
    "attention": TransformerModel,  # legacy alias
    "hybrid": HybridCNNTransformer,
}

VALID_MODEL_TYPES = ("cnn", "improved_cnn", "transformer", "gnn", "hybrid")


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, it_time: int, **kwargs) -> nn.Module:
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model type: {model_type}. Available: {list(_MODEL_REGISTRY)}"
            )
        model_class = _MODEL_REGISTRY[model_type]

        if model_type == "gnn":
            return model_class(
                it_time,
                hidden_dim=kwargs.get("hidden_dim", 128),
                gnn_layers=kwargs.get("gnn_layers", 2),
            )
        if model_type in ("transformer", "attention"):
            return model_class(
                it_time,
                d_model=kwargs.get("d_model", 64),
                nhead=kwargs.get("nhead", 4),
                num_layers=kwargs.get("num_layers", 2),
            )
        return model_class(it_time)


def get_model(model_type: str, it_time: int, **kwargs) -> nn.Module:
    """Legacy-compatible alias for :meth:`ModelFactory.create_model`."""
    return ModelFactory.create_model(model_type, it_time, **kwargs)


def build_model(model_type: str, it_time: int, device: torch.device | None = None) -> nn.Module:
    """Convenience constructor with the paper's fixed hyper-parameters,
    matching what every training/evaluation script previously reimplemented.
    """
    if model_type == "cnn":
        model = CNNModel(it_time=it_time)
    elif model_type == "improved_cnn":
        model = ImprovedCNNModel(it_time=it_time)
    elif model_type == "transformer":
        model = TransformerModel(it_time=it_time, d_model=64, nhead=4, num_layers=2)
    elif model_type == "gnn":
        model = GNNModel(it_time=it_time, hidden_dim=128)
    elif model_type == "hybrid":
        model = HybridCNNTransformer(it_time=it_time)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model.to(device) if device is not None else model


def get_loss_function(loss_type: str = "bce") -> nn.Module:
    if loss_type == "bce":
        return nn.BCELoss()
    if loss_type == "mse":
        return nn.MSELoss()
    return nn.BCELoss()


def get_optimizer(model: nn.Module, optimizer_type: str = "adam", lr: float = 1e-3, weight_decay: float = 1e-4):
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def one_hot_to_class_indices(one_hot_labels: torch.Tensor) -> torch.Tensor:
    return torch.argmax(one_hot_labels, dim=1)


def class_indices_to_one_hot(class_indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(class_indices, num_classes).float()


def bce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.BCELoss()(output, target)


# ---------------------------------------------------------------------------
# 66-dim vector <-> 12x12 symmetric adjacency-matrix conversion
# ---------------------------------------------------------------------------

def output_to_tensor(output: torch.Tensor, n: int = 12) -> torch.Tensor:
    """Fold a 66-dim upper-triangular vector into a symmetric n x n matrix.

    Args:
        output: model output/label, shape (batch, 66).
        n: number of spins (default 12 -> 66 candidate links).

    Returns:
        Symmetric matrix with zero diagonal, shape (batch, n, n).
    """
    batch_size = output.size(0)
    tensor = torch.zeros((batch_size, n, n), dtype=torch.float32, device=output.device)

    for j in range(batch_size):
        start_index = 0
        for i in range(n - 1):
            end_index = start_index + (n - 1 - i)
            tensor[j, i, i + 1:] = output[j, start_index:end_index]
            start_index = end_index
        tensor[j] = tensor[j] + tensor[j].T

    return tensor


def inverse_tensor(tensor: torch.Tensor, n: int = 12) -> torch.Tensor:
    """Inverse of :func:`output_to_tensor`: flatten a symmetric n x n matrix's
    upper triangle (excluding the diagonal) into a 66-dim vector.

    Args:
        tensor: symmetric adjacency matrix, shape (batch, n, n).
        n: number of spins.

    Returns:
        Vector of shape (batch, n*(n-1)//2).
    """
    batch_size = tensor.size(0)
    dim = n * (n - 1) // 2
    result = torch.zeros((batch_size, dim), dtype=torch.float32, device=tensor.device)

    for i in range(batch_size):
        slice_2d = tensor[i, :, :]
        pieces = []
        for row in range(n - 1):
            num_elements = (n - 1) - row
            start_col = n - num_elements
            pieces.append(slice_2d[row, start_col:])
        result[i, :] = torch.cat(pieces)[:dim]

    return result