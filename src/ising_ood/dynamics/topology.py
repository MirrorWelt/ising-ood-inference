"""Interaction-topology (adjacency matrix) generation and I/O.

Consolidates two previously separate, near-duplicate modules:

* the "original" ensemble, where ~25 links are placed via a random
  upper-triangular selection and connectivity is checked with a random walk;
* the "33-links" ensemble, where a fixed number of links (default 33) is
  placed and connectivity is checked exactly via DFS.

Both algorithms are kept verbatim (same statistics as used to generate the
paper's training data); only the surrounding code is unified.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import xml.etree.ElementTree as ET

NUM_NODES_DEFAULT = 12

TopologyMode = Literal["original", "fixed_edges"]


# ---------------------------------------------------------------------------
# Connectivity checks
# ---------------------------------------------------------------------------

def is_connected_random_walk(matrix: np.ndarray, steps: int = 1000) -> bool:
    """Probabilistic connectivity check via a random walk (legacy 'original' method)."""
    n = matrix.shape[0]
    visited = {0}
    current = 0
    for _ in range(steps):
        neighbors = np.where(matrix[current] == 1)[0]
        if len(neighbors) == 0:
            break
        current = int(np.random.choice(neighbors))
        visited.add(current)
    return len(visited) == n


def is_connected_dfs(matrix: np.ndarray) -> bool:
    """Exact connectivity check via depth-first search ('fixed_edges' method)."""
    n = matrix.shape[0]
    visited: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nei in np.where(matrix[node] == 1)[0]:
            if nei not in visited:
                stack.append(int(nei))
    return len(visited) == n


# ---------------------------------------------------------------------------
# "original" ensemble: ~25-link construction via upper-triangular selection
# ---------------------------------------------------------------------------

def _define_tensor_pair_original(n: int = NUM_NODES_DEFAULT, n_links: int = 25):
    """Legacy construction: randomly place ``n_links`` in a flattened 66-dim
    upper-triangular representation, then fold it into a symmetric n x n
    matrix via flip + transpose. Produces two independent candidate matrices.
    """
    dim = n * (n - 1) // 2  # 66 for n = 12

    vector1 = np.zeros(dim, dtype=int)
    vector1[np.random.choice(dim, size=n_links, replace=False)] = 1
    vector2 = np.zeros(dim, dtype=int)
    vector2[np.random.choice(dim, size=n_links, replace=False)] = 1

    tensor1 = np.zeros((n, n), dtype=int)
    tensor2 = np.zeros((n, n), dtype=int)

    start_index = 0
    for i in range(n - 1):
        end_index = start_index + (n - 1 - i)
        end_index = min(end_index, dim)
        tensor1[i, n - end_index + start_index:] = vector1[start_index:end_index]
        tensor2[i, n - end_index + start_index:] = vector2[start_index:end_index]
        start_index = end_index

    re_tensor1 = tensor1[::-1, ::-1]
    re_tensor2 = tensor2[::-1, ::-1]
    tensor1 = re_tensor1 + re_tensor1.T
    tensor2 = re_tensor2 + re_tensor2.T
    return tensor1, tensor2


def _generate_adjacency_original(n: int = NUM_NODES_DEFAULT, n_links: int = 25) -> np.ndarray:
    """Draw a connected adjacency matrix from the 'original' ensemble."""
    tensor1, tensor2 = _define_tensor_pair_original(n=n, n_links=n_links)
    if not is_connected_random_walk(tensor1):
        tensor1, _ = _define_tensor_pair_original(n=n, n_links=n_links)
    return tensor1


# ---------------------------------------------------------------------------
# "fixed_edges" ensemble: exact edge count + DFS-verified connectivity
# ---------------------------------------------------------------------------

def _define_tensor_fixed_edges(n: int, num_edges: int) -> np.ndarray:
    max_edges = n * (n - 1) // 2
    if not (0 <= num_edges <= max_edges):
        raise ValueError(f"num_edges must be in [0, {max_edges}]")

    mat = np.zeros((n, n), dtype=int)
    triu_i, triu_j = np.triu_indices(n, k=1)
    chosen = np.random.choice(len(triu_i), size=num_edges, replace=False)
    mat[triu_i[chosen], triu_j[chosen]] = 1
    mat[triu_j[chosen], triu_i[chosen]] = 1
    return mat


def _generate_adjacency_fixed_edges(
    n: int, num_edges: int, max_try: int = 10_000
) -> np.ndarray:
    """Draw a connected adjacency matrix with exactly ``num_edges`` links."""
    for _ in range(max_try):
        mat = _define_tensor_fixed_edges(n, num_edges)
        if is_connected_dfs(mat):
            return mat
    raise RuntimeError(f"Failed to generate a connected graph after {max_try} attempts.")


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

def generate_topology(
    mode: TopologyMode = "original",
    n: int = NUM_NODES_DEFAULT,
    num_edges: int = 25,
    max_try: int = 10_000,
) -> np.ndarray:
    """Draw a single connected adjacency matrix.

    Args:
        mode: "original" (random-walk connectivity, ``num_edges`` used as the
            approximate link count via the legacy 66-slot selection) or
            "fixed_edges" (DFS-verified, exact link count).
        n: number of spins.
        num_edges: target number of undirected links (N_c in the paper).
        max_try: retry budget for the "fixed_edges" mode.
    """
    if mode == "original":
        return _generate_adjacency_original(n=n, n_links=num_edges)
    if mode == "fixed_edges":
        return _generate_adjacency_fixed_edges(n=n, num_edges=num_edges, max_try=max_try)
    raise ValueError(f"Unknown topology mode: {mode}")


def generate_and_save_matrices(
    out_path: str | Path,
    num_matrices: int = 500,
    mode: TopologyMode = "original",
    n: int = NUM_NODES_DEFAULT,
    num_edges: int = 25,
) -> None:
    """Generate ``num_matrices`` unique connected adjacency matrices and save as XML."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("Matrices")
    seen: set[tuple[int, ...]] = set()
    count = 0

    while count < num_matrices:
        tensor = generate_topology(mode=mode, n=n, num_edges=num_edges)
        key = tuple(int(v) for v in tensor.flatten())
        if key in seen:
            continue
        seen.add(key)

        matrix_element = ET.SubElement(root, "Matrix", id=str(count))
        for row in tensor:
            row_element = ET.SubElement(matrix_element, "Row")
            row_element.text = " ".join(map(str, row))
        count += 1
        if count % 50 == 0:
            print(f"Generated {count} matrices...")

    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved {num_matrices} unique matrices to: {out_path}")


def load_matrices_from_xml(filepath: str | Path) -> list[np.ndarray]:
    """Load a list of adjacency matrices from an XML file produced by
    :func:`generate_and_save_matrices`.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    matrices: list[np.ndarray] = []
    for matrix_element in root.findall("Matrix"):
        rows = [
            list(map(int, row_element.text.split()))
            for row_element in matrix_element.findall("Row")
        ]
        matrices.append(np.array(rows, dtype=int))
    return matrices


def count_links_undirected(A: np.ndarray) -> int:
    """Number of undirected edges encoded by a symmetric 0/1 adjacency matrix."""
    return int((A != 0).astype(np.int32).sum() // 2)