"""Thin CLI wrapper: generate Glauber-trajectory ``.pt`` datasets for a range
of topologies, optionally sweeping temperature.

Replaces the six near-duplicate ``generate_*`` scripts from the original
codebase with a single parametrised tool.

Examples:
    # Initial 50-lattice training pool (fixed T = 1), 33-links ensemble
    python tools/generate_trajectory_dataset.py \
        --matrices-xml data/topologies/matrices_33links.xml \
        --out-dir data/lattice_datasets_33links \
        --start-matrix 1 --end-matrix 50 --samples-per-matrix 3500 \
        --file-prefix tensor_coupling_dataset_33links_matrix

    # Remaining 450 lattices for topology-shift OOD evaluation
    python tools/generate_trajectory_dataset.py \
        --matrices-xml data/topologies/matrices_33links.xml \
        --out-dir data/lattice_datasets_33links \
        --start-matrix 51 --end-matrix 500 --samples-per-matrix 100 \
        --file-prefix tensor_coupling_dataset_33links_matrix

    # Temperature sweep for two 50-lattice classes (original ensemble)
    python tools/generate_trajectory_dataset.py \
        --matrices-xml data/topologies/matrices_original.xml \
        --out-dir data/lattice_datasets_original \
        --temperature-sweep --beta-list 0.01,0.02,0.05,0.1,1,10,20 \
        --samples-per-group 5000
"""

from __future__ import annotations

import argparse
import os

import torch

from ising_ood.dynamics.topology import load_matrices_from_xml, count_links_undirected
from ising_ood.data.dataset import SingleMatrixDataset


def _generate_single_matrix_file(
    matrix_num: int,
    A,
    out_dir: str,
    file_prefix: str,
    num_samples: int,
    length: int,
    beta: float,
    dt: float,
    it_time: int,
) -> None:
    dataset = SingleMatrixDataset(
        num_samples=num_samples, length=length, beta=beta, dt=dt, it_time=it_time,
        adjacency_matrix=A, keep_matrix_tensor=True,
    )
    tensors = torch.stack([d[0] for d in dataset])
    matrices = torch.stack([d[1] for d in dataset])

    save_data = {
        "tensors": tensors,
        "matrices": matrices,
        "parameters": {
            "num_samples": num_samples, "length": length, "Beta": beta,
            "dt": dt, "it_time": it_time, "matrix_num": matrix_num,
            "link_count": count_links_undirected(A),
        },
    }
    filename = os.path.join(out_dir, f"{file_prefix}_{matrix_num}_samples_{num_samples}.pt")
    torch.save(save_data, filename)
    print(f"Saved: {filename} | tensors={tuple(tensors.shape)} matrices={tuple(matrices.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-xml", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--it-time", type=int, default=1000)
    parser.add_argument("--file-prefix", default="tensor_coupling_dataset_nodes")

    # Fixed-temperature range-of-matrices mode
    parser.add_argument("--start-matrix", type=int)
    parser.add_argument("--end-matrix", type=int)
    parser.add_argument("--samples-per-matrix", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)

    # Temperature-sweep mode
    parser.add_argument("--temperature-sweep", action="store_true")
    parser.add_argument("--beta-list", type=str, default="")
    parser.add_argument("--samples-per-group", type=int, default=5000)
    parser.add_argument("--class1-range", type=str, default="0:50")
    parser.add_argument("--class2-range", type=str, default="50:100")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    all_matrices = load_matrices_from_xml(args.matrices_xml)

    if args.temperature_sweep:
        beta_list = [float(b) for b in args.beta_list.split(",") if b.strip()]
        c1_lo, c1_hi = map(int, args.class1_range.split(":"))
        c2_lo, c2_hi = map(int, args.class2_range.split(":"))
        class_groups = {1: all_matrices[c1_lo:c1_hi], 2: all_matrices[c2_lo:c2_hi]}
        num_per_matrix = args.samples_per_group // len(class_groups[1])

        for beta in beta_list:
            for class_idx, matrices in class_groups.items():
                all_tensors, all_matrix_tensors = [], []
                for A in matrices:
                    for _ in range(num_per_matrix):
                        from ising_ood.dynamics.glauber import generate_tensor_coupling
                        traj = generate_tensor_coupling(A, beta, args.dt, args.it_time, args.length)
                        all_tensors.append(torch.tensor(traj, dtype=torch.float32))
                        all_matrix_tensors.append(torch.tensor(A, dtype=torch.float32))

                save_data = {
                    "tensors": torch.stack(all_tensors),
                    "matrices": torch.stack(all_matrix_tensors),
                    "parameters": {"Beta": beta, "class": f"class{class_idx}", "it_time": args.it_time},
                }
                filename = os.path.join(
                    args.out_dir,
                    f"tensor_coupling_dataset_beta_{beta}_class{class_idx}_samples_{args.samples_per_group}.pt",
                )
                torch.save(save_data, filename)
                print(f"Saved: {filename}")
        return

    if args.start_matrix is None or args.end_matrix is None:
        parser.error("--start-matrix/--end-matrix are required outside --temperature-sweep mode.")

    for matrix_num in range(args.start_matrix, args.end_matrix + 1):
        A = all_matrices[matrix_num - 1]
        _generate_single_matrix_file(
            matrix_num, A, args.out_dir, args.file_prefix,
            args.samples_per_matrix, args.length, args.beta, args.dt, args.it_time,
        )


if __name__ == "__main__":
    main()