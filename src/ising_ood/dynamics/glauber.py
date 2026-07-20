"""Forward problem: mean-field Glauber dynamics for the kinetic Ising model.

Implements Eq. (2) of Zhu, Qiao & Ran, "Out-of-distribution Neural Inference
in Dynamical Ising Models" (arXiv:2607.03039):

    dm_i/dt = -m_i + tanh( beta * J * sum_j A_ij m_j )

discretised with a first-order mixing step of size ``dt``:

    m_i(t+dt) = (1 - dt) * m_i(t) + dt * tanh( beta * (A @ m)_i )

This module consolidates the ``coupling_term`` / ``generate_tensor_coupling``
pair that was previously duplicated, almost verbatim, across six separate
data-generation scripts.
"""

from __future__ import annotations

import numpy as np


def coupling_term(m: np.ndarray, beta: float, dt: float, A: np.ndarray) -> np.ndarray:
    """One discretised step of the mean-field Glauber equation (Eq. 2).

    Args:
        m: current local-magnetization vector, shape (L,).
        beta: inverse temperature (J is absorbed into beta), beta = 1/T.
        dt: mixing/time-step size, typically small and in (0, 1].
        A: adjacency matrix of the interaction topology, shape (L, L).

    Returns:
        Updated magnetization vector, shape (L,).
    """
    m0 = m.copy()
    driven = np.tanh(beta * (A @ m))
    return m0 * (1.0 - dt) + driven * dt


def generate_tensor_coupling(
    A: np.ndarray,
    beta: float,
    dt: float,
    it_time: int,
    length: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Integrate Eq. (2) for ``it_time`` steps from a random initial condition.

    Args:
        A: adjacency matrix, shape (length, length).
        beta: inverse temperature.
        dt: integration step size.
        it_time: number of recorded time steps.
        length: number of spins L.
        rng: optional NumPy random Generator for reproducibility. If None,
            the legacy global ``numpy.random`` state is used (kept for
            bit-for-bit compatibility with the original scripts).

    Returns:
        Trajectory array of shape (it_time, length), dtype float32.
    """
    if rng is not None:
        m = rng.uniform(-1.0, 1.0, size=length)
    else:
        m = 2 * np.random.rand(length) - 1  # legacy behaviour, kept intentionally

    trajectory = np.zeros((it_time, length), dtype=np.float32)
    for t in range(it_time):
        m = coupling_term(m, beta, dt, A)
        trajectory[t] = m
    return trajectory