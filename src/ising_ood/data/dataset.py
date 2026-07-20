"""Torch Datasets used both at generation time (:class:`SingleMatrixDataset`)
and at training/evaluation time (:class:`MultiClassCouplingDataset`).

This module consolidates roughly five near-identical implementations of
``MultiClassCouplingDataset`` that previously existed across the training and
evaluation scripts (differing only in file-naming convention and minor
bookkeeping such as whether per-sample class ids were retained).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from ..dynamics.glauber import generate_tensor_coupling
from ..models.factory import inverse_tensor

DatasetMode = Literal["nodes", "33links"]


class SingleMatrixDataset(Dataset):
    """On-the-fly trajectory generator for a single fixed adjacency matrix.

    Used by the data-generation tools (see ``tools/generate_trajectory_dataset.py``).
    """

    def __init__(
        self,
        num_samples: int,
        length: int,
        beta: float,
        dt: float,
        it_time: int,
        adjacency_matrix: np.ndarray,
        keep_matrix_tensor: bool = False,
        log_every: int = 500,
    ):
        self.num_samples = num_samples
        self.length = length
        self.beta = beta
        self.dt = dt
        self.it_time = it_time
        self.adjacency_matrix = adjacency_matrix
        self.keep_matrix_tensor = keep_matrix_tensor
        self.data: list = []

        a_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32) if keep_matrix_tensor else None

        for i in range(num_samples):
            trajectory = generate_tensor_coupling(adjacency_matrix, beta, dt, it_time, length)
            trajectory_t = torch.tensor(trajectory, dtype=torch.float32)
            self.data.append((trajectory_t, a_tensor) if keep_matrix_tensor else trajectory_t)

            if log_every and (i + 1) % log_every == 0:
                print(f"    SingleMatrixDataset progress: {i + 1}/{num_samples}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class MultiClassCouplingDataset(Dataset):
    """Loads pre-generated ``.pt`` trajectory files for one or more topologies
    ("classes") and exposes ``(trajectory, 66-dim label)`` pairs.

    Args:
        data_dir: directory containing ``tensor_coupling_dataset_*.pt`` files.
        class_ids: list of topology ids to include.
        total_samples: target total number of samples across all classes,
            distributed as evenly as possible.
        dataset_mode: "nodes" for the original per-topology file naming
            (``tensor_coupling_dataset_nodes_{id}_samples_*.pt``), or
            "33links" for the fixed-edge-count naming
            (``tensor_coupling_dataset_33links_matrix_{id}_samples_*.pt``).
        track_sample_class_ids: if True, also stores the originating class id
            for every retained sample (needed by some diagnostics).
    """

    def __init__(
        self,
        data_dir: str | Path,
        class_ids: list[int],
        total_samples: int = 35_000,
        transform=None,
        dataset_mode: DatasetMode = "nodes",
        track_sample_class_ids: bool = False,
    ):
        self.data_dir = str(data_dir)
        self.class_ids = class_ids
        self.num_classes = len(class_ids)
        self.transform = transform
        self.dataset_mode = dataset_mode
        self.track_sample_class_ids = track_sample_class_ids

        if self.num_classes <= 0:
            raise ValueError("class_ids is empty.")

        samples_per_class = total_samples // self.num_classes
        remainder = total_samples % self.num_classes

        self.time_series: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.sample_class_ids: list[int] = []

        for class_idx, class_id in enumerate(class_ids):
            cur_samples = samples_per_class + (1 if class_idx < remainder else 0)
            file_path = self._resolve_file(class_id)
            if file_path is None:
                print(f"Warning: no file found for class {class_id} (mode={dataset_mode}). Skipping.")
                continue

            data = torch.load(file_path)
            tensors = data["tensors"]
            n_available = len(tensors)

            if cur_samples > n_available:
                print(
                    f"Warning: requested {cur_samples} samples for class {class_id}, "
                    f"but only {n_available} available. Using all."
                )
                cur_samples = n_available

            indices = np.random.choice(n_available, cur_samples, replace=False)
            selected_time_series = tensors[indices]

            if "labels" in data:
                labels_66d = data["labels"][indices]
            elif "matrices" in data:
                labels_66d = inverse_tensor(data["matrices"][indices])
            else:
                # Pseudo-labels: only meaningful for smoke-testing the pipeline.
                n_features = 66
                labels_66d = torch.zeros(cur_samples, n_features)
                for i in range(cur_samples):
                    labels_66d[i, : min(class_id, n_features)] = 1.0

            self.time_series.append(selected_time_series)
            self.labels.append(labels_66d)
            if self.track_sample_class_ids:
                self.sample_class_ids.extend([class_id] * cur_samples)

            print(f"[{dataset_mode}] Loaded {cur_samples} samples from class {class_id}")

        if not self.time_series:
            raise ValueError(f"No data loaded for mode={dataset_mode}. Check file paths and class_ids.")

        self.time_series = torch.cat(self.time_series, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        if self.track_sample_class_ids:
            self.sample_class_ids = torch.tensor(self.sample_class_ids, dtype=torch.long)

        print(f"[{dataset_mode}] Total samples: {len(self.time_series)} (target: {total_samples})")

    def _resolve_file(self, class_id: int) -> str | None:
        if self.dataset_mode == "nodes":
            exact = os.path.join(self.data_dir, f"tensor_coupling_dataset_nodes_{class_id}_samples_35000.pt")
            pattern = os.path.join(self.data_dir, f"tensor_coupling_dataset_nodes_{class_id}_samples_*.pt")
        elif self.dataset_mode == "33links":
            exact = os.path.join(
                self.data_dir, f"tensor_coupling_dataset_33links_matrix_{class_id}_samples_3500.pt"
            )
            pattern = os.path.join(
                self.data_dir, f"tensor_coupling_dataset_33links_matrix_{class_id}_samples_*.pt"
            )
        else:
            raise ValueError(f"Unsupported dataset_mode: {self.dataset_mode}")

        if os.path.exists(exact):
            return exact
        files = glob.glob(pattern)
        return files[0] if files else None

    def __len__(self) -> int:
        return len(self.time_series)

    def __getitem__(self, idx):
        time_series = self.time_series[idx]
        label = self.labels[idx]
        if self.transform:
            time_series = self.transform(time_series)
        if self.track_sample_class_ids:
            return time_series, label, self.sample_class_ids[idx]
        return time_series, label