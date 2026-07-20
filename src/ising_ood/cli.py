"""Command-line entry point: ``ising-ood <subcommand> --config <path>``.

Every subcommand consumes a YAML configuration (see ``configs/*.yaml``) so
that experiment parameters are never hard-coded in source files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, PathConfig, load_yaml_config
from .dynamics.topology import generate_and_save_matrices
from .training.engine import train_one_stage, save_checkpoint
from .training.staged import train_staged
from .data.dataset import MultiClassCouplingDataset
from .models.factory import build_model


def _merge_config(defaults, overrides: dict):
    for k, v in overrides.items():
        if hasattr(defaults, k):
            setattr(defaults, k, v)
    return defaults


def cmd_generate_topology(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config) if args.config else {}
    mode = cfg.get("mode", "original")
    out_path = cfg.get("out_path", "./data/topologies/matrices.xml")
    generate_and_save_matrices(
        out_path=out_path,
        num_matrices=cfg.get("num_matrices", 500),
        mode=mode,
        n=cfg.get("n", 12),
        num_edges=cfg.get("num_edges", 25),
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config)
    paths = PathConfig()
    exp = _merge_config(ExperimentConfig(), cfg)

    dataset_mode = cfg.get("dataset_mode", "nodes")
    data_dir = cfg.get("data_dir", str(paths.nodes_dataset_dir if dataset_mode == "nodes" else paths.links33_dataset_dir))
    class_ids = cfg.get("class_ids") or list(range(1, exp.num_lattices + 1))

    if cfg.get("staged", False):
        train_staged(
            it_time=exp.it_time,
            class_ids=class_ids,
            nodes_dir=cfg.get("nodes_dir", str(paths.nodes_dataset_dir)),
            links33_dir=cfg.get("links33_dir", str(paths.links33_dataset_dir)),
            model_type=exp.model_type,
            num_epochs=exp.num_epochs,
            phase1_epochs=cfg.get("phase1_epochs", 20),
            phase2_epochs=cfg.get("phase2_epochs", 50),
            total_samples=exp.total_samples,
            batch_size=exp.batch_size,
            lr=exp.lr,
            weight_decay=exp.weight_decay,
            out_dir=str(paths.model_dir),
        )
        return

    import torch
    from torch.utils.data import DataLoader, random_split

    dataset = MultiClassCouplingDataset(data_dir, class_ids, exp.total_samples, dataset_mode=dataset_mode)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=exp.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=exp.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(exp.model_type, exp.it_time, device=device)

    result = train_one_stage(
        model, train_loader, test_loader, device,
        num_epochs=exp.num_epochs, lr=exp.lr, weight_decay=exp.weight_decay,
        verbose_prefix=f"[{dataset_mode}] ",
    )

    if result["best_state"] is not None:
        model.load_state_dict(result["best_state"]["model_state_dict"])
        save_checkpoint(
            model, result["optimizer"], exp.model_type, class_ids, exp.it_time,
            result["best_state"]["train_acc"], result["best_state"]["test_acc"],
            result["best_state"]["epoch"], paths.model_dir,
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config)
    paths = PathConfig()
    kind = cfg.get("kind")

    if kind == "topology_ood":
        from .evaluation.id_ood_topology import evaluate_id_accuracy_vs_nl
        df = evaluate_id_accuracy_vs_nl(
            data_dir=cfg.get("data_dir", str(paths.nodes_dataset_dir)),
            model_dir=cfg.get("model_dir", str(paths.model_dir)),
        )
    elif kind == "temperature_shift":
        from .evaluation.temperature_shift import evaluate_temperature_shift_accuracy
        df = evaluate_temperature_shift_accuracy(
            data_dir=cfg.get("data_dir", str(paths.links33_dataset_dir)),
            model_dir=cfg.get("model_dir", str(paths.model_dir)),
        )
    elif kind == "link_diagnostics_nl":
        from .evaluation.link_diagnostics import predicted_links_vs_class_count
        df = predicted_links_vs_class_count(
            data_dir=cfg.get("data_dir", str(paths.nodes_dataset_dir)),
            model_dir=cfg.get("model_dir", str(paths.model_dir)),
            class_counts=cfg.get("class_counts", list(range(10, 51, 2))),
        )
    elif kind == "link_diagnostics_temperature":
        from .evaluation.link_diagnostics import predicted_links_vs_temperature
        df = predicted_links_vs_temperature(
            data_dir=cfg.get("data_dir", str(paths.links33_dataset_dir)),
            model_dir=cfg.get("model_dir", str(paths.model_dir)),
        )
    else:
        raise ValueError(f"Unknown evaluation kind: {kind}")

    out_csv = Path(cfg.get("out_csv", f"results/{kind}.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ising-ood")
    sub = parser.add_subparsers(dest="command", required=True)

    p_topo = sub.add_parser("generate-topology", help="Generate an adjacency-matrix ensemble XML.")
    p_topo.add_argument("--config", required=True)
    p_topo.set_defaults(func=cmd_generate_topology)

    p_train = sub.add_parser("train", help="Train a regression model.")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Run an evaluation/diagnostic.")
    p_eval.add_argument("--config", required=True)
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()