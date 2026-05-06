"""Linear-quadratic regulator helpers used in the Chapter 4 labs."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


def discrete_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the infinite-horizon discrete LQR gain K and DARE solution P."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def lqr_control(
    x: np.ndarray,
    x_ref: np.ndarray,
    K: np.ndarray,
    u_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Compute u = u_ref - K (x - x_ref) with column-vector-safe shapes."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if u_ref is None:
        u_ref = np.zeros(K.shape[0])
    u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
    return u_ref - K @ (x - x_ref)


def simulate_lqr(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    x0: np.ndarray,
    steps: int,
    x_ref: np.ndarray | None = None,
    u_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a linear system under LQR feedback."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1)
    if x_ref is None:
        x_ref = np.zeros_like(x)
    if u_ref is None:
        u_ref = np.zeros(B.shape[1])

    X = np.zeros((steps + 1, A.shape[0]))
    U = np.zeros((steps, B.shape[1]))
    X[0] = x
    for k in range(steps):
        u = lqr_control(X[k], x_ref, K, u_ref)
        U[k] = u
        X[k + 1] = A @ X[k] + B @ u
    return X, U

