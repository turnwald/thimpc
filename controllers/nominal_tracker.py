"""Nominal circular-path tracking commands for the mobile robot lab."""

from __future__ import annotations

import numpy as np


def nominal_yaw_rate(omega_r: float) -> float:
    """Return the reference yaw rate used by the nominal tracker."""
    return float(omega_r)


def corrected_yaw_rate(omega_r: float, delta_omega: float, omega_min: float | None = None, omega_max: float | None = None) -> float:
    """Combine nominal yaw rate with a correction and optional actuator bounds."""
    omega = float(omega_r + delta_omega)
    if omega_min is not None or omega_max is not None:
        lo = -np.inf if omega_min is None else omega_min
        hi = np.inf if omega_max is None else omega_max
        omega = float(np.clip(omega, lo, hi))
    return omega
