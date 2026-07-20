"""Reproduces Fig. 3(a) of arXiv:2607.03039: temperature-shift OOD accuracy
gamma^tem_OOD as a function of T, for topologies fixed at the training
ensemble but trajectories generated at unseen temperatures.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.discovery import discover_models, discover_temperature_files
from ..data.dataset import MultiClassCouplingDataset  # noqa: F401  (kept for API symmetry)
from ..models.factory import build_model, inverse_tensor
from torch.utils.data import Dataset


class _SingleFileTemperatureDataset(Dataset):
    """Loads one beta/T-sweep ``.pt`` file into a (trajectory, label) dataset."""

    def __init__(self, file_path: str):
        data = torch.load(file_path, map_location="cpu")
        if "tensors" not in data:
            raise KeyError(f"{file_path}: missing 'tensors'")
        self.tensors = data["tensors"]
        if "labels" in data:
            self.labels = data["labels"]
        elif "matrices" in data:
            self.labels = inverse_tensor(data["matrices"])
        else:
            raise KeyError(f"{file_path}: missing 'labels'/'matrices'")
        if len(self.tensors) != len(self.labels):
            raise ValueError(f"{file_path}: tensors/labels size mismatch")

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx].float(), self.labels[idx].float()


def _load_model(model_type: str, model_path: str, default_it_time: int, device):
    checkpoint = torch.load(model_path, map_location="cpu")
    it_time = checkpoint.get("it_time", default_it_time) if isinstance(checkpoint, dict) else default_it_time
    model = build_model(model_type, it_time, device=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _accuracy(model, loader, device) -> float:
    total_correct, total = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == targets).float().sum().item()
            total += inputs.size(0) * targets.size(1)
    return 100.0 * total_correct / total if total > 0 else 0.0


def evaluate_temperature_shift_accuracy(
    data_dir: str,
    model_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    only_classes: list[int] | None = None,
    only_model_types: list[str] | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """For every discovered class (topology group) and every discovered model
    checkpoint, evaluate edge-wise accuracy across all available temperatures.

    Returns a long-form DataFrame with columns: class, T, x_type, x_value_raw,
    model, accuracy, file.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_index = discover_temperature_files(data_dir)
    classes = sorted(c for c in data_index if only_classes is None or c in only_classes)
    if not classes:
        raise ValueError("No temperature-sweep dataset files discovered.")

    model_files = discover_models(model_dir, only_model_types=only_model_types)
    if not model_files:
        raise ValueError("No model checkpoints discovered.")

    models = {}
    for mt, path in model_files.items():
        try:
            models[mt] = _load_model(mt, path, default_it_time=1000, device=device)
        except Exception as e:  # noqa: BLE001
            print(f"[warning] failed to load {mt}: {e}")
            models[mt] = None

    rows = []
    for class_idx in classes:
        for entry in data_index[class_idx]:
            fp, T, x_type, x_raw = entry["file_path"], entry["T"], entry["x_type"], entry["x_value_raw"]
            try:
                ds = _SingleFileTemperatureDataset(fp)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] failed to load {fp}: {e}")
                continue

            for mt, model in models.items():
                if model is None:
                    acc = np.nan
                else:
                    try:
                        acc = _accuracy(model, loader, device)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [warning] {mt} evaluation failed on {fp}: {e}")
                        acc = np.nan
                rows.append(
                    {"class": class_idx, "T": T, "x_type": x_type, "x_value_raw": x_raw,
                     "model": mt, "accuracy": acc, "file": fp}
                )

    return pd.DataFrame(rows).sort_values(by=["class", "T", "model"]).reset_index(drop=True)