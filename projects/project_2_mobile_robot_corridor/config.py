"""Numerical settings for Project 2."""

from __future__ import annotations

import numpy as np

DT = 0.1
STEPS = 120
HORIZON = 15
V_REF = 0.8
OMEGA_REF = 0.2
RADIUS = V_REF / OMEGA_REF
NARROW_CENTER = 0.5 * np.pi
NARROW_WIDTH = 1.4

Q = np.diag([1.0, 25.0, 4.0])
R = np.array([[0.35]])
DELTA_MIN = np.array([-0.7])
DELTA_MAX = np.array([0.7])
DELTA_RATE = np.array([0.12])

INITIAL_ERROR = np.array([0.0, 0.38, 0.12])
