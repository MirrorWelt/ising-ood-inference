import numpy as np

from ising_ood.dynamics.glauber import coupling_term, generate_tensor_coupling
from ising_ood.dynamics.topology import (
    generate_topology,
    is_connected_dfs,
    count_links_undirected,
)


def test_coupling_term_output_bounded():
    rng = np.random.default_rng(0)
    A = np.zeros((5, 5))
    m = rng.uniform(-1, 1, size=5)
    m_next = coupling_term(m, beta=1.0, dt=0.1, A=A)
    # No coupling (A=0) => tanh(0)=0, pure decay toward 0.
    assert np.all(np.abs(m_next) <= np.abs(m) + 1e-9)


def test_generate_tensor_coupling_shape():
    rng = np.random.default_rng(1)
    A = generate_topology(mode="fixed_edges", n=8, num_edges=6)
    traj = generate_tensor_coupling(A, beta=1.0, dt=0.01, it_time=20, length=8, rng=rng)
    assert traj.shape == (20, 8)
    assert np.all(np.abs(traj) <= 1.0 + 1e-6)


def test_fixed_edges_topology_is_connected_with_exact_edge_count():
    A = generate_topology(mode="fixed_edges", n=12, num_edges=15)
    assert is_connected_dfs(A)
    assert count_links_undirected(A) == 15
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)