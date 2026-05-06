"""Plotting helpers shared by Chapter 4 scenario scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def savefig(path: str | Path, fig: plt.Figure | None = None) -> None:
    """Save a figure with lecture-friendly defaults without closing it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.gcf() if fig is None else fig
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def plot_attitude_time(
    path: str | Path,
    t: np.ndarray,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    bounds: dict[str, float],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot theta, omega, and input for attitude-control runs."""
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    for label, (X, U) in runs.items():
        axes[0].plot(t, X[:, 0], label=label)
        axes[1].plot(t, X[:, 1], label=label)
        axes[2].step(t[:-1], U[:, 0], where="post", label=label)
    axes[0].axhline(bounds["theta_max"], color="k", linestyle="--", linewidth=0.9)
    axes[0].axhline(bounds["theta_min"], color="k", linestyle="--", linewidth=0.9)
    axes[2].axhline(bounds["u_max"], color="k", linestyle="--", linewidth=0.9)
    axes[2].axhline(bounds["u_min"], color="k", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("theta [rad]")
    axes[1].set_ylabel("omega [rad/s]")
    axes[2].set_ylabel("tau")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[2].grid(True, alpha=0.25)
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axes


def plot_robot_top_view(
    path: str | Path,
    reference_xy: np.ndarray,
    trajectories: dict[str, np.ndarray],
    radius: float,
    inner_radius: float,
    outer_radius: float,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot robot trajectories, reference circle, obstacle, wall, and corridor."""
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], "k--", linewidth=1.2, label="reference")
    ax.plot(inner_radius * np.cos(theta), inner_radius * np.sin(theta), color="tab:red", linewidth=1.2, label="obstacle boundary")
    ax.plot(outer_radius * np.cos(theta), outer_radius * np.sin(theta), color="tab:gray", linewidth=1.2, label="wall boundary")
    narrow_theta = np.linspace(np.pi - 0.35, np.pi + 0.35, 80)
    narrow_inner = radius - 0.22
    narrow_outer = radius + 0.22
    ax.plot(narrow_inner * np.cos(narrow_theta), narrow_inner * np.sin(narrow_theta), color="tab:orange", linewidth=2.2, label="narrowed corridor")
    ax.plot(narrow_outer * np.cos(narrow_theta), narrow_outer * np.sin(narrow_theta), color="tab:orange", linewidth=2.2)
    for label, X in trajectories.items():
        ax.plot(X[:, 0], X[:, 1], linewidth=1.6, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax
