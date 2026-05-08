"""Numerical settings for Project 1."""

from __future__ import annotations

import numpy as np

DT = 0.1
STEPS = 80
HORIZON = 18

A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([[0.5 * DT * DT], [DT]])
Q = np.diag([20.0, 2.0])
R = np.array([[0.25]])
X0 = np.array([1.0, 0.0])

U_MIN = np.array([-0.45])
U_MAX = np.array([0.45])
X_MIN = np.array([-1.2, -2.5])
X_MAX = np.array([1.2, 2.5])
RATE_BOUND = np.array([0.25])
