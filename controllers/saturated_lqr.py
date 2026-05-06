"""LQR feedback with simple actuator clipping."""

from __future__ import annotations

import numpy as np

from controllers.lqr import lqr_control


def saturated_lqr_control(
    x: np.ndarray,
    x_ref: np.ndarray,
    K: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    u_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Compute an LQR command and clip it to input bounds."""
    u = lqr_control(x, x_ref, K, u_ref)
    return np.clip(u, np.asarray(u_min, dtype=float), np.asarray(u_max, dtype=float))


def simulate_saturated_lqr(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    x0: np.ndarray,
    steps: int,
    u_min: np.ndarray,
    u_max: np.ndarray,
    x_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a linear system under saturated LQR feedback."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1)
    if x_ref is None:
        x_ref = np.zeros_like(x)

    X = np.zeros((steps + 1, A.shape[0]))
    U = np.zeros((steps, B.shape[1]))
    X[0] = x
    for k in range(steps):
        u = saturated_lqr_control(X[k], x_ref, K, u_min, u_max)
        U[k] = u
        X[k + 1] = A @ X[k] + B @ u
    return X, U

