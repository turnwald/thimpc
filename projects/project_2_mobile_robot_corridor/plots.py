"""Project-local plots for the mobile robot corridor project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from projects.project_2_mobile_robot_corridor import config

CORRIDOR_HALF_WIDTH = getattr(config, "CORRIDOR_HALF_WIDTH", 0.55)
NARROW_HALF_WIDTH = getattr(config, "NARROW_HALF_WIDTH", 0.22)
NARROW_OFFSET = getattr(config, "NARROW_OFFSET", 0.32)


def _wrapped_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _corridor_error_bounds(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrapped = _wrapped_angle(theta - config.NARROW_CENTER)
    half_angle = config.NARROW_WIDTH / 2.0
    inside = np.abs(wrapped) <= half_angle
    blend = np.zeros_like(theta, dtype=float)
    blend[inside] = 0.5 * (1.0 + np.cos(np.pi * wrapped[inside] / half_angle))
    half_width = CORRIDOR_HALF_WIDTH - blend * (CORRIDOR_HALF_WIDTH - NARROW_HALF_WIDTH)
    center = blend * NARROW_OFFSET
    return center - half_width, center + half_width


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
    theta = np.linspace(0.0, 2.0 * np.pi, 420)
    eta_min, eta_max = _corridor_error_bounds(theta)
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], "k--", linewidth=1.2, label="reference")
    ax.plot(
        (radius - eta_max) * np.cos(theta),
        (radius - eta_max) * np.sin(theta),
        color="tab:red",
        linewidth=2.4,
        label="obstacle boundary",
    )
    ax.plot(
        (radius - eta_min) * np.cos(theta),
        (radius - eta_min) * np.sin(theta),
        color="tab:gray",
        linewidth=2.4,
        label="wall boundary",
    )
    narrow_theta = np.linspace(
        config.NARROW_CENTER - 0.5 * config.NARROW_WIDTH,
        config.NARROW_CENTER + 0.5 * config.NARROW_WIDTH,
        100,
    )
    narrow_eta_min, narrow_eta_max = _corridor_error_bounds(narrow_theta)
    ax.plot(
        (radius - narrow_eta_min) * np.cos(narrow_theta),
        (radius - narrow_eta_min) * np.sin(narrow_theta),
        color="tab:orange",
        linewidth=3.0,
        label="critical narrowed region",
    )
    ax.plot(
        (radius - narrow_eta_max) * np.cos(narrow_theta),
        (radius - narrow_eta_max) * np.sin(narrow_theta),
        color="tab:orange",
        linewidth=3.0,
    )
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
