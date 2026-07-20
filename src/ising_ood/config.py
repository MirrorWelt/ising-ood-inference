"""Central, override-able configuration.

All previously hard-coded absolute paths (e.g.
``/root/shared-nvme/home/zyb_glauber/paper_model/...``) are replaced by this
module. Every path can be overridden by an environment variable, falling
back to a sane relative default so the project runs out of the box on any
machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Physical / experimental constants (shared across the paper's experiments)
# ---------------------------------------------------------------------------

LENGTH: int = 12          # number of spins L (=> L*(L-1)/2 = 66 candidate links)
DT: float = 0.001         # Glauber integration step size
IT_TIME: int = 1000       # number of recorded time steps per trajectory

# 23 inverse temperatures used for the temperature-shift OOD sweep (T = 1/beta)
BETA_LIST: list[float] = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.57, 0.6, 0.8, 1,
    1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 10, 20,
]

NUM_MATRICES_TOTAL: int = 500      # total generated topologies per ensemble
NUM_MATRICES_TRAIN_POOL: int = 50  # topologies actually used for training (N_L <= 50)


def _env_path(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default)).expanduser()


@dataclass
class PathConfig:
    """Filesystem layout. Every field is overridable via an env var."""

    data_root: Path = field(
        default_factory=lambda: _env_path("ISING_OOD_DATA_ROOT", "./data")
    )

    # Topology XML files (adjacency-matrix ensembles)
    matrices_original_xml: Path = field(
        default_factory=lambda: _env_path(
            "ISING_OOD_MATRICES_ORIGINAL_XML", "./data/topologies/matrices_original.xml"
        )
    )
    matrices_33links_xml: Path = field(
        default_factory=lambda: _env_path(
            "ISING_OOD_MATRICES_33LINKS_XML", "./data/topologies/matrices_33links.xml"
        )
    )

    # Trajectory datasets (.pt files)
    nodes_dataset_dir: Path = field(
        default_factory=lambda: _env_path(
            "ISING_OOD_NODES_DATASET_DIR", "./data/lattice_datasets_original"
        )
    )
    links33_dataset_dir: Path = field(
        default_factory=lambda: _env_path(
            "ISING_OOD_LINKS33_DATASET_DIR", "./data/lattice_datasets_33links"
        )
    )

    # Model checkpoints and run outputs
    model_dir: Path = field(
        default_factory=lambda: _env_path("ISING_OOD_MODEL_DIR", "./results/saved_models")
    )
    results_dir: Path = field(
        default_factory=lambda: _env_path("ISING_OOD_RESULTS_DIR", "./results")
    )

    def ensure_dirs(self) -> None:
        for p in (
            self.data_root,
            self.nodes_dataset_dir,
            self.links33_dataset_dir,
            self.model_dir,
            self.results_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        self.matrices_original_xml.parent.mkdir(parents=True, exist_ok=True)
        self.matrices_33links_xml.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    """One run's tunable hyper-parameters (mirrors the paper's N_L, N_c, T)."""

    length: int = LENGTH
    dt: float = DT
    it_time: int = IT_TIME
    beta: float = 1.0
    num_links: int = 25          # N_c in the paper (25 = imbalanced main setting, 33 = balanced control)
    num_lattices: int = 50       # N_L: number of distinct training topologies
    total_samples: int = 35_000  # N: total training trajectories
    model_type: str = "cnn"      # one of: cnn, improved_cnn, gnn, transformer, hybrid
    num_epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42


def load_yaml_config(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}