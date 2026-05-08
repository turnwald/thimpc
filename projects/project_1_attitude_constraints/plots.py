"""Project-local plots for constrained attitude control."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def savefig(path: str | Path, fig: plt.Figure) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def _ellipse_points(P: np.ndarray, level: float, num: int = 200) -> np.ndarray:
    """Return phase-plane points satisfying x.T P x = level."""
    P = np.asarray(P, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, num)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    transform = np.linalg.cholesky(np.linalg.inv(P) * float(level))
    return (transform @ circle).T


def plot_time_histories(
    path: str | Path,
    t: np.ndarray,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    x_min: np.ndarray,
    x_max: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    for label, (X, U) in runs.items():
        axes[0].plot(t, X[:, 0], label=label)
        axes[1].plot(t, X[:, 1], label=label)
        axes[2].step(t[:-1], U[:, 0], where="post", label=label)
    axes[0].axhline(x_min[0], color="k", linestyle="--", linewidth=0.9)
    axes[0].axhline(x_max[0], color="k", linestyle="--", linewidth=0.9)
    axes[2].axhline(u_min[0], color="k", linestyle="--", linewidth=0.9)
    axes[2].axhline(u_max[0], color="k", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("theta [rad]")
    axes[1].set_ylabel("omega [rad/s]")
    axes[2].set_ylabel("torque")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axes


def plot_phase_plane(
    path: str | Path,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    x_min: np.ndarray,
    x_max: np.ndarray,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for label, (X, _) in runs.items():
        ax.plot(X[:, 0], X[:, 1], label=label)
        ax.plot(X[0, 0], X[0, 1], "o", markersize=4)
    ax.axvline(x_min[0], color="k", linestyle="--", linewidth=0.9)
    ax.axvline(x_max[0], color="k", linestyle="--", linewidth=0.9)
    ax.axhline(x_min[1], color="k", linestyle=":", linewidth=0.9)
    ax.axhline(x_max[1], color="k", linestyle=":", linewidth=0.9)
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_terminal_geometry(
    path: str | Path,
    P: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for level in [0.5, 1.5, 3.0, 6.0]:
        pts = _ellipse_points(P, level)
        ax.plot(pts[:, 0], pts[:, 1], label=f"x^T P x = {level:g}")
    ax.axvline(x_min[0], color="k", linestyle="--", linewidth=0.9)
    ax.axvline(x_max[0], color="k", linestyle="--", linewidth=0.9)
    ax.axhline(x_min[1], color="k", linestyle=":", linewidth=0.9)
    ax.axhline(x_max[1], color="k", linestyle=":", linewidth=0.9)
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("omega [rad/s]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_constraint_activity(
    path: str | Path,
    t: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    slack: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    theta_margin = np.minimum(X[:, 0] - x_min[0], x_max[0] - X[:, 0])
    input_margin = np.minimum(U[:, 0] - u_min[0], u_max[0] - U[:, 0])
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.8), sharex=True)
    axes[0].plot(t, theta_margin)
    axes[0].axhline(0.0, color="k", linewidth=0.9)
    axes[0].set_ylabel("theta margin")
    axes[1].step(t[:-1], input_margin, where="post")
    axes[1].axhline(0.0, color="k", linewidth=0.9)
    axes[1].set_ylabel("input margin")
    axes[2].step(t[:-1], slack, where="post")
    axes[2].set_ylabel("theta slack")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axes
