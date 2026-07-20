"""Reproduces Fig. 2(b) (in-distribution accuracy vs N_L) and
Fig. 2(c) (topology-shift OOD accuracy vs N_L) of arXiv:2607.03039.

Fig. 2(b): models trained on N_L in-distribution topologies are evaluated on
held-out trajectories from the *same* topologies (ID test set).

Fig. 2(c): the same models are evaluated on trajectories generated from
topologies *absent* from training (topology-shift OOD test set).
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..data.discovery import get_available_class_ids
from ..models.factory import build_model, count_parameters, inverse_tensor
from ..data.dataset import MultiClassCouplingDataset


# ---------------------------------------------------------------------------
# Fig. 2(b): ID accuracy vs N_L, by scanning already-trained checkpoints
# ---------------------------------------------------------------------------

def _extract_model_info(filename: str):
    match = re.search(r"regression_(\w+)_(\d+)classes\.pth", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def _load_model_for_eval(model_dir: str, model_type: str, n_classes: int, it_time: int, device):
    model = build_model(model_type, it_time, device=device)
    path = os.path.join(model_dir, f"regression_{model_type}_{n_classes}classes.pth")
    if not os.path.exists(path):
        path = os.path.join(model_dir, f"regression_{model_type}_{n_classes}classes_simple.pth")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _accuracy(model, loader, device) -> float:
    total_acc, total = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == targets).float().mean()
            total_acc += acc.item() * inputs.size(0)
            total += inputs.size(0)
    return 100.0 * total_acc / total


def evaluate_id_accuracy_vs_nl(
    data_dir: str,
    model_dir: str,
    it_time: int = 1000,
    total_samples: int = 35_000,
    test_ratio: float = 0.2,
    seed: int = 42,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Fig. 2(b): scan every ``regression_{model_type}_{N_L}classes.pth`` found
    in ``model_dir``, rebuild the matching ID test split, and report accuracy.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_files = glob.glob(os.path.join(model_dir, "regression_*.pth"))
    model_info: dict[int, list[str]] = {}
    for f in model_files:
        base = os.path.basename(f)
        if "simple" in base:
            continue
        model_type, n_classes = _extract_model_info(base)
        if model_type is None:
            continue
        model_info.setdefault(n_classes, []).append(model_type)

    available_ids = get_available_class_ids(data_dir, "nodes")
    rows = []

    for n_classes in sorted(model_info):
        class_ids = available_ids[:n_classes]
        full_dataset = MultiClassCouplingDataset(data_dir, class_ids, total_samples, dataset_mode="nodes")
        test_size = int(test_ratio * len(full_dataset))
        train_size = len(full_dataset) - test_size
        torch.manual_seed(seed)
        _, test_dataset = random_split(full_dataset, [train_size, test_size])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        for model_type in model_info[n_classes]:
            try:
                model = _load_model_for_eval(model_dir, model_type, n_classes, it_time, device)
                acc = _accuracy(model, test_loader, device)
                params = count_parameters(model)
                print(f"  N_L={n_classes} {model_type}: ID test accuracy = {acc:.2f}%")
                rows.append({"model": model_type, "N_L": n_classes, "id_test_accuracy": acc, "params": params})
            except Exception as e:  # noqa: BLE001
                print(f"  Evaluation failed for {model_type} @ N_L={n_classes}: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig. 2(c): topology-shift OOD accuracy vs N_L (held-out topologies)
# ---------------------------------------------------------------------------

class TopologyOODDataset(Dataset):
    """Trajectories from topologies *not* seen during training, each labeled
    by its own (known) adjacency matrix.
    """

    def __init__(
        self,
        data_dir: str,
        class_ids: list[int],
        adjacency_matrices: list[np.ndarray],
        samples_per_class: int = 100,
        file_prefix: str = "tensor_coupling_dataset_33links_matrix",
    ):
        self.time_series: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

        for class_id in class_ids:
            pattern = os.path.join(data_dir, f"{file_prefix}_{class_id}_samples_*.pt")
            matches = sorted(
                glob.glob(pattern),
                key=lambda p: int(re.search(r"_samples_(\d+)\.pt$", os.path.basename(p)).group(1))
                if re.search(r"_samples_(\d+)\.pt$", os.path.basename(p)) else -1,
                reverse=True,
            )
            if not matches:
                print(f"Warning: no file found for OOD class {class_id}. Skipping.")
                continue
            data = torch.load(matches[0])
            tensors = data["tensors"] if isinstance(data, dict) else data

            adj = adjacency_matrices[class_id - 1]
            label = inverse_tensor(torch.tensor(adj, dtype=torch.float32).unsqueeze(0)).squeeze(0)

            n_available = tensors.shape[0]
            n_use = min(n_available, samples_per_class)
            idx = np.random.choice(n_available, n_use, replace=False) if n_available > n_use else slice(None)
            selected = tensors[idx]

            self.time_series.append(selected)
            self.labels.append(label.repeat(n_use, 1))

        if not self.time_series:
            raise ValueError("No OOD data loaded; check paths and class_ids.")

        self.time_series = torch.cat(self.time_series, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx], self.labels[idx]


def evaluate_topology_ood_vs_nl(
    ood_data_dir: str,
    model_dir: str,
    adjacency_matrices: list[np.ndarray],
    ood_class_ids: list[int],
    it_time: int = 1000,
    samples_per_class: int = 100,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Fig. 2(c): evaluate every discovered ``regression_{model_type}_{N_L}classes.pth``
    checkpoint on the fixed pool of held-out (OOD) topologies.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ood_dataset = TopologyOODDataset(ood_data_dir, ood_class_ids, adjacency_matrices, samples_per_class)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    model_files = glob.glob(os.path.join(model_dir, "regression_*.pth"))
    rows = []
    for f in model_files:
        base = os.path.basename(f)
        if "simple" in base:
            continue
        model_type, n_classes = _extract_model_info(base)
        if model_type is None:
            continue
        try:
            model = _load_model_for_eval(model_dir, model_type, n_classes, it_time, device)
            acc = _accuracy(model, ood_loader, device)
            print(f"  N_L={n_classes} {model_type}: topology-OOD accuracy = {acc:.2f}%")
            rows.append({"model": model_type, "N_L": n_classes, "topology_ood_accuracy": acc})
        except Exception as e:  # noqa: BLE001
            print(f"  Evaluation failed for {model_type} @ N_L={n_classes}: {e}")

    return pd.DataFrame(rows)