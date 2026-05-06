"""Terminal-cost and geometry helpers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


def dare_terminal_cost(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return the DARE solution P_infty."""
    return solve_discrete_are(A, B, Q, R)


def ellipse_points(P: np.ndarray, level: float, num: int = 200) -> np.ndarray:
    """Return points x with x.T P x = level for a positive-definite 2x2 P."""
    P = np.asarray(P, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, num)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    transform = np.linalg.cholesky(np.linalg.inv(P) * float(level))
    return (transform @ circle).T
