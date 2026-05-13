"""Shared mobile-robot linearization used by Projects 2 and 3."""

from __future__ import annotations

import numpy as np


def tracking_error_matrices(dt: float, v_r: float, omega_r: float) -> tuple[np.ndarray, np.ndarray]:
    """Euler-discretized small-error model for circular tracking."""
    Ac = np.array(
        [
            [0.0, omega_r, 0.0],
            [-omega_r, 0.0, v_r],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    Bc = np.array([[0.0], [0.0], [1.0]], dtype=float)
    return np.eye(3) + dt * Ac, dt * Bc
