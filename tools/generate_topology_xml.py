"""Thin CLI wrapper: generate and save an adjacency-matrix ensemble.

Example:
    python tools/generate_topology_xml.py --mode original --num-matrices 500 \
        --out data/topologies/matrices_original.xml

    python tools/generate_topology_xml.py --mode fixed_edges --num-edges 33 \
        --num-matrices 500 --out data/topologies/matrices_33links.xml
"""

from __future__ import annotations

import argparse

from ising_ood.dynamics.topology import generate_and_save_matrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "fixed_edges"], default="original")
    parser.add_argument("--num-matrices", type=int, default=500)
    parser.add_argument("--num-edges", type=int, default=25)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    generate_and_save_matrices(
        out_path=args.out,
        num_matrices=args.num_matrices,
        mode=args.mode,
        n=args.n,
        num_edges=args.num_edges,
    )


if __name__ == "__main__":
    main()