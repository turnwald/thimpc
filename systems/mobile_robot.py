"""Mobile robot simulator and tracking-error model for Chapter 4 Project 2."""

from __future__ import annotations

import numpy as np


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


def corridor_bounds(
    path_angle: np.ndarray,
    half_width: float = 0.55,
    narrow_half_width: float = 0.22,
    narrow_center: float = 0.5 * np.pi,
    narrow_width: float = 1.4,
    narrow_offset: float = 0.32,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time-varying lateral bounds eta_min, eta_max for an annular corridor.

    In the narrowed section, the admissible lateral-error interval is shifted by
    ``narrow_offset``. A positive offset means the reference path ``eta = 0`` can
    lie outside the allowed interval.
    """
    angle = np.asarray(path_angle, dtype=float)
    wrapped = wrap_angle(angle - narrow_center)
    inside_narrow = np.abs(wrapped) <= narrow_width / 2.0
    half = np.full_like(angle, half_width - margin, dtype=float)
    center = np.zeros_like(angle, dtype=float)
    half[inside_narrow] = narrow_half_width - margin
    center[inside_narrow] = narrow_offset
    half = np.maximum(half, 0.03)
    return center - half, center + half
