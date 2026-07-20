"""Reproduces the predicted-link-population diagnostic N-hat_c of
arXiv:2607.03039: Fig. 2/3's companion panels showing the average number of
predicted links vs N_L (Fig. 3b) and vs T (Fig. 3c), plus the Fig. S4(b)
balanced-density control.

This diagnostic is what distinguishes "density-preserving" (Transformer-like)
from "sparse/no-link-collapse" (CNN-3-like) inference strategies discussed in
the paper.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.discovery import discover_models, discover_temperature_files, get_available_class_ids
from ..data.dataset import MultiClassCouplingDataset
from ..models.factory import build_model
from .temperature_shift import _SingleFileTemperatureDataset  # reuse loader


def _load_model(model_type: str, model_path: str, default_it_time: int, device):
    checkpoint = torch.load(model_path, map_location="cpu")
    it_time = checkpoint.get("it_time", default_it_time) if isinstance(checkpoint, dict) else default_it_time
    model = build_model(model_type, it_time, device=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _mean_predicted_and_true_links(model, loader, device) -> tuple[float, float]:
    pred_counts, true_counts = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            pred_counts.append(preds.sum(dim=1).cpu())
            true_counts.append(targets.sum(dim=1).cpu())
    pred_counts = torch.cat(pred_counts).numpy() if pred_counts else np.array([])
    true_counts = torch.cat(true_counts).numpy() if true_counts else np.array([])
    pred_mean = float(np.mean(pred_counts)) if pred_counts.size else np.nan
    true_mean = float(np.mean(true_counts)) if true_counts.size else np.nan
    return pred_mean, true_mean


# ---------------------------------------------------------------------------
# Fig. 3(b)-style: N-hat_c vs N_L on the ID test set
# ---------------------------------------------------------------------------

def predicted_links_vs_class_count(
    data_dir: str,
    model_dir: str,
    class_counts: list[int],
    it_time: int = 1000,
    total_samples: int = 35_000,
    batch_size: int = 32,
    dataset_mode: str = "nodes",
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_ids = get_available_class_ids(data_dir, dataset_mode)
    rows = []

    for n_classes in class_counts:
        if n_classes > len(all_ids):
            print(f"Warning: N_L={n_classes} exceeds available topologies ({len(all_ids)}); skipping.")
            continue
        class_ids = all_ids[:n_classes]

        full_dataset = MultiClassCouplingDataset(
            data_dir, class_ids, total_samples, dataset_mode=dataset_mode, track_sample_class_ids=True
        )
        from torch.utils.data import random_split

        train_size = int(0.8 * len(full_dataset))
        _, test_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for model_type, model_path in discover_models(model_dir, only_class_count=n_classes).items():
            try:
                model = _load_model(model_type, model_path, it_time, device)
                pred_avg, true_avg = _mean_predicted_and_true_links(model, loader, device)
                print(f"  N_L={n_classes} {model_type}: pred={pred_avg:.2f}, true={true_avg:.2f}")
                rows.append(
                    {"model": model_type, "N_L": n_classes, "pred_avg_links": pred_avg, "true_avg_links": true_avg}
                )
            except Exception as e:  # noqa: BLE001
                print(f"  Evaluation failed for {model_type} @ N_L={n_classes}: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig. 3(c) / Fig. S4(b)-style: N-hat_c vs T
# ---------------------------------------------------------------------------

def predicted_links_vs_temperature(
    data_dir: str,
    model_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    only_classes: list[int] | None = None,
    only_model_types: list[str] | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_index = discover_temperature_files(data_dir)
    classes = sorted(c for c in data_index if only_classes is None or c in only_classes)
    if not classes:
        raise ValueError("No temperature-sweep dataset files discovered.")

    model_files = discover_models(model_dir, only_model_types=only_model_types)
    if not model_files:
        raise ValueError("No model checkpoints discovered.")

    models = {mt: _load_model(mt, path, 1000, device) for mt, path in model_files.items()}

    rows = []
    for class_idx in classes:
        for entry in data_index[class_idx]:
            fp, T = entry["file_path"], entry["T"]
            try:
                ds = _SingleFileTemperatureDataset(fp)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] failed to load {fp}: {e}")
                continue

            for mt, model in models.items():
                try:
                    pred_avg, true_avg = _mean_predicted_and_true_links(model, loader, device)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warning] {mt} evaluation failed on {fp}: {e}")
                    pred_avg, true_avg = np.nan, np.nan
                rows.append(
                    {"class": class_idx, "T": T, "model": mt,
                     "pred_avg_links": pred_avg, "true_avg_links": true_avg, "file": fp}
                )

    return pd.DataFrame(rows).sort_values(by=["class", "T", "model"]).reset_index(drop=True)