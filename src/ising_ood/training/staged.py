"""Two-stage training: pretrain on the 'nodes' (original) ensemble, then
fine-tune on the '33links' (fixed-edge) ensemble, matching the paper's
transfer-training setup.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ..data.dataset import MultiClassCouplingDataset
from ..models.factory import build_model, count_parameters
from .engine import evaluate_epoch, save_checkpoint


def _split(ds, train_frac: float = 0.8):
    train_size = int(train_frac * len(ds))
    return random_split(ds, [train_size, len(ds) - train_size])


def train_staged(
    it_time: int,
    class_ids: list[int],
    nodes_dir: str,
    links33_dir: str,
    model_type: str = "cnn",
    num_epochs: int = 80,
    phase1_epochs: int = 20,
    phase2_epochs: int = 50,
    total_samples: int = 35_000,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    save_model: bool = True,
    out_dir: str | Path = "results/saved_models",
) -> dict:
    """Stage 1 (epochs < phase1_epochs): train on the 'nodes' ensemble.
    Stage 2 (remaining epochs): train on the '33links' ensemble.
    """
    planned = phase1_epochs + phase2_epochs
    if planned > num_epochs:
        print(f"[Warning] phase1+phase2={planned} > total={num_epochs}; truncating phase2.")
        phase2_epochs = max(0, num_epochs - phase1_epochs)
    elif planned < num_epochs:
        print(f"[Info] {num_epochs - planned} extra epochs will continue on the 33links stage.")

    dataset_nodes = MultiClassCouplingDataset(nodes_dir, class_ids, total_samples, dataset_mode="nodes")
    dataset_33 = MultiClassCouplingDataset(links33_dir, class_ids, total_samples, dataset_mode="33links")

    train_nodes, test_nodes = _split(dataset_nodes)
    train_33, test_33 = _split(dataset_33)

    loaders = {
        "nodes": (
            DataLoader(train_nodes, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test_nodes, batch_size=batch_size, shuffle=False, num_workers=2),
        ),
        "33links": (
            DataLoader(train_33, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test_33, batch_size=batch_size, shuffle=False, num_workers=2),
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_type, it_time, device=device)
    model_params = count_parameters(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_test_acc, best_train_acc, best_state = 0.0, 0.0, None

    print(f"\n{'=' * 60}\nStaged training: {model_type.upper()} | classes={class_ids}")
    print(f"Stage-1(nodes): {phase1_epochs} epochs, Stage-2(33links): {phase2_epochs} epochs")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        phase = "nodes" if epoch < phase1_epochs else "33links"
        train_loader, test_loader = loaders[phase]

        model.train()
        running_loss, running_acc, total = 0.0, 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == targets).float().mean()
            running_loss += loss.item() * inputs.size(0)
            running_acc += acc.item() * inputs.size(0)
            total += inputs.size(0)
        train_loss, train_acc = running_loss / total, 100.0 * running_acc / total

        test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)
        scheduler.step(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {
                "epoch": epoch + 1,
                "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "train_acc": train_acc,
                "test_acc": test_acc,
                "phase_name": phase,
            }
        best_train_acc = max(best_train_acc, train_acc)

        print(
            f"[{phase}] Epoch [{epoch + 1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

    if save_model and best_state is not None:
        full_path, simple_path = save_checkpoint(
            model, optimizer, model_type, class_ids, it_time,
            best_state["train_acc"], best_state["test_acc"], best_state["epoch"],
            out_dir, extra={"phase_name": best_state["phase_name"]},
        )
        print(f"Model saved to: {full_path}\nSimple model saved to: {simple_path}")

    return {
        "best_test_acc": best_test_acc,
        "best_train_acc": best_train_acc,
        "model_params": model_params,
    }