"""Project-local plots for the mobile robot corridor project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from projects.project_2_mobile_robot_corridor import config


def savefig(path: str | Path, fig: plt.Figure) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def plot_top_view(
    path: str | Path,
    reference_xy: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    radius: float,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], "k--", linewidth=1.2, label="reference")
    ax.plot((radius - 0.55) * np.cos(theta), (radius - 0.55) * np.sin(theta), color="tab:red", linewidth=1.2, label="obstacle boundary")
    ax.plot((radius + 0.55) * np.cos(theta), (radius + 0.55) * np.sin(theta), color="tab:gray", linewidth=1.2, label="wall boundary")
    narrow_theta = np.linspace(
        config.NARROW_CENTER - 0.5 * config.NARROW_WIDTH,
        config.NARROW_CENTER + 0.5 * config.NARROW_WIDTH,
        100,
    )
    ax.plot((radius - 0.22) * np.cos(narrow_theta), (radius - 0.22) * np.sin(narrow_theta), color="tab:orange", linewidth=2.2, label="narrowed corridor")
    ax.plot((radius + 0.22) * np.cos(narrow_theta), (radius + 0.22) * np.sin(narrow_theta), color="tab:orange", linewidth=2.2)
    for label, run in runs.items():
        states = run["states"]
        ax.plot(states[:, 0], states[:, 1], linewidth=1.6, label=label)
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


def plot_lateral_error(
    path: str | Path,
    t: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    first = next(iter(runs.values()))
    ax.fill_between(t, first["eta_min"], first["eta_max"], color="0.9", label="corridor")
    for label, run in runs.items():
        ax.plot(t, run["errors"][:, 1], label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lateral error eta [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_input(
    path: str | Path,
    t: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    lower: float,
    upper: float,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for label, run in runs.items():
        ax.step(t[:-1], run["delta"], where="post", label=label)
    ax.axhline(lower, color="k", linestyle="--", linewidth=0.9)
    ax.axhline(upper, color="k", linestyle="--", linewidth=0.9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("yaw-rate correction [rad/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_margin(
    path: str | Path,
    t: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for label, run in runs.items():
        ax.plot(t, run["margin"], label=label)
    ax.axhline(0.0, color="k", linewidth=0.9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("minimum corridor margin [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_horizon_comparison(
    path: str | Path,
    t: np.ndarray,
    horizon_runs: dict[int, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for horizon, run in horizon_runs.items():
        ax.plot(t, run["errors"][:, 1], label=f"N = {horizon}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lateral error eta [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax
