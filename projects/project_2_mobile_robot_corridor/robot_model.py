"""Mobile robot model and local feedback helpers for Project 2."""

from __future__ import annotations

import numpy as np


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


def corrected_yaw_rate(
    omega_r: float,
    delta_omega: float,
    omega_min: float | None = None,
    omega_max: float | None = None,
) -> float:
    """Combine nominal yaw rate with a correction and optional actuator bounds."""
    omega = float(omega_r + delta_omega)
    if omega_min is not None or omega_max is not None:
        lo = -np.inf if omega_min is None else omega_min
        hi = np.inf if omega_max is None else omega_max
        omega = float(np.clip(omega, lo, hi))
    return omega


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def unicycle_step(
    state: np.ndarray,
    v: float,
    omega: float,
    dt: float,
    yaw_rate_bias: float = 0.0,
) -> np.ndarray:
    """Euler-step the nonlinear unicycle model."""
    x, y, psi = np.asarray(state, dtype=float).reshape(3)
    omega_real = omega + yaw_rate_bias
    return np.array(
        [
            x + dt * v * np.cos(psi),
            y + dt * v * np.sin(psi),
            float(wrap_angle(psi + dt * omega_real)),
        ],
        dtype=float,
    )


def circular_reference(t: np.ndarray, radius: float, v_r: float, omega_r: float) -> np.ndarray:
    """Return [x_ref, y_ref, psi_ref] samples for a counter-clockwise circle."""
    t = np.asarray(t, dtype=float)
    phi = omega_r * t
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    psi = wrap_angle(phi + np.pi / 2.0)
    return np.column_stack([x, y, psi])


def tracking_error(state: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return body-frame path error [xi, eta, psi_error]."""
    x, y, psi = np.asarray(state, dtype=float).reshape(3)
    xr, yr, psir = np.asarray(reference, dtype=float).reshape(3)
    dx = x - xr
    dy = y - yr
    c = np.cos(psir)
    s = np.sin(psir)
    xi = c * dx + s * dy
    eta = -s * dx + c * dy
    psi_error = wrap_angle(psi - psir)
    return np.array([xi, eta, psi_error], dtype=float)


def state_from_error(error: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reconstruct global state from [xi, eta, psi_error] and reference."""
    xi, eta, psi_error = np.asarray(error, dtype=float).reshape(3)
    xr, yr, psir = np.asarray(reference, dtype=float).reshape(3)
    c = np.cos(psir)
    s = np.sin(psir)
    dx = c * xi - s * eta
    dy = s * xi + c * eta
    return np.array([xr + dx, yr + dy, float(wrap_angle(psir + psi_error))], dtype=float)
