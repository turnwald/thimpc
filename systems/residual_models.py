"""Least-squares residual model for Chapter 4 Project 3."""

from __future__ import annotations

import numpy as np


def residual_features(error: np.ndarray, delta_omega: np.ndarray | float) -> np.ndarray:
    """Feature map phi(error, delta_omega) used for residual learning."""
    error = np.asarray(error, dtype=float)
    u = np.asarray(delta_omega, dtype=float).reshape(-1, 1)
    if error.ndim == 1:
        e = error.reshape(1, -1)
        one_row = True
    else:
        e = error
        one_row = False
    if u.shape[0] == 1 and e.shape[0] > 1:
        u = np.repeat(u, e.shape[0], axis=0)
    phi = np.column_stack(
        [
            np.ones(e.shape[0]),
            e,
            u.reshape(-1),
            e[:, 1] * u.reshape(-1),
            e[:, 2] * u.reshape(-1),
        ]
    )
    return phi.reshape(-1) if one_row else phi


def fit_residual_least_squares(
    errors: np.ndarray,
    inputs: np.ndarray,
    residuals: np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Fit W so residual ~= phi @ W.T."""
    Phi = residual_features(errors, inputs)
    residuals = np.asarray(residuals, dtype=float)
    reg = ridge * np.eye(Phi.shape[1])
    return (np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ residuals)).T


def predict_residual(W: np.ndarray, error: np.ndarray, delta_omega: np.ndarray | float) -> np.ndarray:
    """Predict residual from learned least-squares weights."""
    Phi = residual_features(error, delta_omega)
    if Phi.ndim == 1:
        return np.asarray(W, dtype=float) @ Phi
    return Phi @ np.asarray(W, dtype=float).T

