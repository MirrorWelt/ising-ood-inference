# Architecture

## Package layout

    src/ising_ood/
    ├── config.py        # paths & experiment hyper-parameters (no hard-coded paths)
    ├── dynamics/         # forward problem: Eq. (2) Glauber dynamics + topology generation
    ├── data/              # trajectory Dataset classes + file-discovery utilities
    ├── models/            # CNN-2, CNN-3, GNN, Transformer, Hybrid (Fig. S1-S3)
    ├── training/          # single-stage and staged (nodes -> 33links) training loops
    ├── evaluation/        # Fig. 2 (ID/topology-OOD) and Fig. 3 (temperature-OOD, link diagnostics)
    └── cli.py             # `ising-ood generate-topology|train|evaluate --config <yaml>`

## Design rule: stable core + thin adapters

Following the "stable core + thin adapters" principle, the physics core
(`dynamics/glauber.py`, `dynamics/topology.py`) and the five network
architectures (`models/*.py`) are treated as frozen reference
implementations: any new experiment should be expressed as a new YAML
config or a new thin script in `tools/`, not as a modification of the core
algorithms.

## Network architectures

| Model | File | Paper figure |
|---|---|---|
| CNN-2 | `models/cnn.py::CNNModel` | Fig. S1 (shallow variant) |
| CNN-3 | `models/cnn.py::ImprovedCNNModel` | Fig. S1 (deep variant) |
| GNN | `models/gnn.py::GNNModel` | Fig. S3 |
| Transformer | `models/transformer.py::TransformerModel` | Fig. S2 |
| Hybrid | `models/hybrid.py::HybridCNNTransformer` | combines Fig. S1 + Fig. S2 blocks |