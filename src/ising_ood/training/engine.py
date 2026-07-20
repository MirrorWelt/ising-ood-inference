"""Single-stage training/evaluation loop.

Consolidates the near-identical ``for epoch in range(num_epochs): ...`` loops
that were previously duplicated across the original- and 33-links-dataset
training scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.factory import build_model, count_parameters


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion, device) -> tuple[float, float]:
    """Run one evaluation pass. Returns (mean_loss, elementwise_accuracy_pct)."""
    model.eval()
    running_loss, running_acc, total = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == targets).float().mean()

            running_loss += loss.item() * inputs.size(0)
            running_acc += acc.item() * inputs.size(0)
            total += inputs.size(0)
    return running_loss / total, 100.0 * running_acc / total


def _train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss, running_acc, total = 0.0, 0.0, 0
    for inputs, targets in loader:
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
    return running_loss / total, 100.0 * running_acc / total


def save_checkpoint(
    model: nn.Module,
    optimizer,
    model_type: str,
    class_ids: list[int],
    it_time: int,
    train_acc: float,
    test_acc: float,
    epoch: int,
    out_dir: str | Path,
    extra: dict | None = None,
) -> tuple[str, str]:
    """Save a full checkpoint (+ a weights-only 'simple' copy)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_str = f"{len(class_ids)}classes"
    full_path = out_dir / f"regression_{model_type}_{class_str}.pth"
    simple_path = out_dir / f"regression_{model_type}_{class_str}_simple.pth"

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "model_type": model_type,
        "class_ids": class_ids,
        "it_time": it_time,
    }
    if extra:
        state.update(extra)

    torch.save(state, full_path)
    torch.save(model.state_dict(), simple_path)
    return str(full_path), str(simple_path)


def train_one_stage(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    verbose_prefix: str = "",
) -> dict:
    """Train ``model`` for ``num_epochs`` epochs, returning the best-epoch summary.

    Returns:
        dict with keys: best_test_acc, best_train_acc, model, optimizer,
        model_params, history (list of per-epoch dicts).
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_test_acc, best_train_acc = 0.0, 0.0
    best_state = None
    history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)
        scheduler.step(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {
                "epoch": epoch + 1,
                "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        best_train_acc = max(best_train_acc, train_acc)

        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
             "test_loss": test_loss, "test_acc": test_acc}
        )
        print(
            f"{verbose_prefix}Epoch [{epoch + 1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

    return {
        "best_test_acc": best_test_acc,
        "best_train_acc": best_train_acc,
        "best_state": best_state,
        "model": model,
        "optimizer": optimizer,
        "model_params": count_parameters(model),
        "history": history,
    }