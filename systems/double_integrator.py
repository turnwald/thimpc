"""Double-integrator attitude model for Chapter 4 Project 1."""

from __future__ import annotations

import numpy as np


def attitude_matrices(dt: float = 0.1, inertia: float = 1.0, exact: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return x[k+1] = A x[k] + B tau[k] for x = [theta, omega]."""
    if exact:
        B = np.array([[0.5 * dt * dt / inertia], [dt / inertia]], dtype=float)
    else:
        B = np.array([[0.0], [dt / inertia]], dtype=float)
    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    return A, B


def simulate_linear(A: np.ndarray, B: np.ndarray, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Simulate a linear system for a provided input sequence."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    U = np.asarray(U, dtype=float)
    X = np.zeros((U.shape[0] + 1, A.shape[0]))
    X[0] = np.asarray(x0, dtype=float).reshape(-1)
    for k, u in enumerate(U):
        X[k + 1] = A @ X[k] + B @ np.asarray(u, dtype=float).reshape(-1)
    return X
