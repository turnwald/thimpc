"""Project-local animations for the mobile robot corridor project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.transforms import Affine2D

from projects.project_2_mobile_robot_corridor import config

CORRIDOR_HALF_WIDTH = getattr(config, "CORRIDOR_HALF_WIDTH", 0.55)
NARROW_HALF_WIDTH = getattr(config, "NARROW_HALF_WIDTH", 0.22)
NARROW_OFFSET = getattr(config, "NARROW_OFFSET", 0.32)
SCENE_MAX_ANGLE = np.deg2rad(120.0)
RUN_COLORS = {
    "baseline": "tab:gray",
    "nominal": "tab:gray",
    "saturated LQR": "tab:green",
    "MPC": "tab:blue",
}


def _frame_indices(length: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    indices = np.arange(0, length, stride, dtype=int)
    if indices[-1] != length - 1:
        indices = np.r_[indices, length - 1]
    return indices


def _run_order(runs: dict[str, dict[str, np.ndarray]]) -> list[str]:
    preferred = ["baseline", "nominal", "saturated LQR", "MPC"]
    ordered = [label for label in preferred if label in runs]
    ordered.extend(label for label in runs if label not in ordered)
    return ordered


def _sequential_frames(runs: dict[str, dict[str, np.ndarray]], stride: int) -> list[tuple[str, int, tuple[str, ...]]]:
    steps = _scene_steps(runs)
    sample_indices = _frame_indices(steps, stride)
    frames: list[tuple[str, int, tuple[str, ...]]] = []
    completed: list[str] = []
    for label in _run_order(runs):
        for k in sample_indices:
            frames.append((label, int(k), tuple(completed)))
        completed.append(label)
    return frames


def _wrapped_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _rotated_points(center: tuple[float, float], points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray(center) + points @ rotation.T


def _rotated_box(center: tuple[float, float], theta: float, x0: float, y0: float, width: float, height: float) -> np.ndarray:
    corners = np.array(
        [
            [x0, y0],
            [x0 + width, y0],
            [x0 + width, y0 + height],
            [x0, y0 + height],
        ]
    )
    return _rotated_points(center, corners, theta)


def _corridor_error_bounds(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrapped = _wrapped_angle(theta - config.NARROW_CENTER)
    half_angle = config.NARROW_WIDTH / 2.0
    inside = np.abs(wrapped) <= half_angle
    blend = np.zeros_like(theta, dtype=float)
    blend[inside] = 0.5 * (1.0 + np.cos(np.pi * wrapped[inside] / half_angle))
    half_width = CORRIDOR_HALF_WIDTH - blend * (CORRIDOR_HALF_WIDTH - NARROW_HALF_WIDTH)
    center = blend * NARROW_OFFSET
    return center - half_width, center + half_width


def _scene_steps(runs: dict[str, dict[str, np.ndarray]]) -> int:
    steps = min(len(run["states"]) for run in runs.values())
    first_run = next(iter(runs.values()))
    reference = np.asarray(first_run["reference"], dtype=float)[:steps]
    phi = np.unwrap(np.arctan2(reference[:, 1], reference[:, 0]))
    visible = np.flatnonzero(phi <= phi[0] + SCENE_MAX_ANGLE)
    if visible.size == 0:
        return min(steps, 1)
    return int(visible[-1] + 1)


def _scene_limits(radius: float) -> tuple[tuple[float, float], tuple[float, float]]:
    theta = np.linspace(0.0, SCENE_MAX_ANGLE, 240)
    eta_min, eta_max = _corridor_error_bounds(theta)
    inner = radius - eta_max
    outer = radius - eta_min
    x = np.r_[inner * np.cos(theta), outer * np.cos(theta)]
    y = np.r_[inner * np.sin(theta), outer * np.sin(theta)]
    pad = 0.65
    return (float(np.min(x) - pad), float(np.max(x) + pad)), (float(np.min(y) - pad), float(np.max(y) + pad))


def _draw_corridor(ax: plt.Axes, radius: float) -> None:
    theta = np.linspace(0.0, SCENE_MAX_ANGLE, 260)
    eta_min, eta_max = _corridor_error_bounds(theta)
    inner = radius - eta_max
    outer = radius - eta_min
    ax.fill(
        np.r_[inner * np.cos(theta), outer[::-1] * np.cos(theta[::-1])],
        np.r_[inner * np.sin(theta), outer[::-1] * np.sin(theta[::-1])],
        color="#eef6f2",
        linewidth=0.0,
        zorder=0,
    )
    ax.plot(inner * np.cos(theta), inner * np.sin(theta), color="tab:red", linewidth=2.8, label="obstacle boundary", zorder=1)
    ax.plot(outer * np.cos(theta), outer * np.sin(theta), color="0.35", linewidth=2.8, label="wall boundary", zorder=1)

    narrow_theta = np.linspace(
        config.NARROW_CENTER - 0.5 * config.NARROW_WIDTH,
        min(config.NARROW_CENTER + 0.5 * config.NARROW_WIDTH, SCENE_MAX_ANGLE),
        120,
    )
    eta_min, eta_max = _corridor_error_bounds(narrow_theta)
    narrow_inner = radius - eta_max
    narrow_outer = radius - eta_min
    ax.plot(narrow_inner * np.cos(narrow_theta), narrow_inner * np.sin(narrow_theta), color="tab:orange", linewidth=3.2, label="critical narrowed region", zorder=2)
    ax.plot(narrow_outer * np.cos(narrow_theta), narrow_outer * np.sin(narrow_theta), color="tab:orange", linewidth=3.2, zorder=2)
    ax.fill(
        np.r_[narrow_inner * np.cos(narrow_theta), narrow_outer * np.cos(narrow_theta[::-1])],
        np.r_[narrow_inner * np.sin(narrow_theta), narrow_outer * np.sin(narrow_theta[::-1])],
        color="tab:orange",
        alpha=0.10,
        linewidth=0.0,
        zorder=1,
    )


def _draw_robot(ax: plt.Axes, state: np.ndarray, *, color: str) -> None:
    x, y, heading = np.asarray(state, dtype=float).reshape(3)
    center = (float(x), float(y))
    transform = Affine2D().rotate_around(x, y, heading) + ax.transData
    shadow = Ellipse((x - 0.04, y - 0.04), 0.78, 0.50, facecolor="0.0", edgecolor="none", alpha=0.12, zorder=5)
    shadow.set_transform(transform)
    ax.add_patch(shadow)
    body = Ellipse((x, y), 0.78, 0.52, facecolor="#f8fbff", edgecolor=color, linewidth=2.6, zorder=6)
    body.set_transform(transform)
    ax.add_patch(body)
    for y0 in (-0.31, 0.23):
        wheel = Polygon(
            _rotated_box(center, heading, -0.23, y0, 0.46, 0.08),
            closed=True,
            facecolor="0.18",
            edgecolor="0.18",
            linewidth=0.8,
            zorder=7,
        )
        ax.add_patch(wheel)
    bumper = Polygon(
        _rotated_points(center, np.array([[0.26, -0.16], [0.50, 0.0], [0.26, 0.16]]), heading),
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=0.9,
        zorder=8,
    )
    ax.add_patch(bumper)
    cabin = Circle(center, 0.15, facecolor="#dff0ff", edgecolor=color, linewidth=1.3, zorder=9)
    cabin.set_transform(transform)
    ax.add_patch(cabin)
    nose = _rotated_points(center, np.array([[0.03, 0.0], [0.61, 0.0]]), heading)
    ax.annotate(
        "",
        xy=nose[1],
        xytext=nose[0],
        arrowprops={"arrowstyle": "->", "color": "0.12", "linewidth": 1.7},
        zorder=10,
    )


def _draw_corridor_axes(
    ax: plt.Axes,
    runs: dict[str, dict[str, np.ndarray]],
    k: int,
    *,
    active_label: str | None = None,
    completed_labels: tuple[str, ...] = (),
    radius: float = config.RADIUS,
) -> None:
    steps = _scene_steps(runs)

    ax.clear()
    _draw_corridor(ax, radius)
    first_run = next(iter(runs.values()))
    reference_xy = first_run["reference"][:steps, :2]
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], "k--", linewidth=1.2, label="reference")
    if active_label == "MPC":
        horizon_stop = min(k + config.HORIZON + 1, steps)
        ax.scatter(
            reference_xy[k:horizon_stop, 0],
            reference_xy[k:horizon_stop, 1],
            s=18,
            color=RUN_COLORS["MPC"],
            alpha=0.20,
            label="MPC look-ahead window",
            zorder=2,
        )
    if active_label is None:
        active_label = _run_order(runs)[0]
    active_margin = float(runs[active_label]["margin"][k])
    margin_lines = [f"running: {active_label}", f"sample {k}, margin {active_margin:+.3f} m"]
    if completed_labels:
        margin_lines.append("completed trails: " + ", ".join(completed_labels))
    for label in _run_order(runs):
        run = runs[label]
        color = RUN_COLORS.get(label, None)
        states = run["states"][:steps]
        if label in completed_labels:
            ax.plot(states[:, 0], states[:, 1], color=color, linewidth=2.2, alpha=0.80)
            ax.scatter(states[-1, 0], states[-1, 1], color=color, edgecolor="white", s=32, zorder=5)
        elif label == active_label:
            ax.plot(states[: k + 1, 0], states[: k + 1, 1], color=color, linewidth=2.4)
            ax.plot(states[:, 0], states[:, 1], color=color, linewidth=0.9, alpha=0.18)
            _draw_robot(ax, states[k], color=color or "0.25")
            statuses = run.get("statuses", [])
            if label == "MPC" and k < len(statuses) and statuses[k] != "Solve_Succeeded":
                margin_lines.append(f"MPC status: {statuses[k]}")

    ax.text(
        0.02,
        0.98,
        "\n".join(margin_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    ax.text(
        0.02,
        0.03,
        "gray=baseline   green=saturated LQR   blue=MPC",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
    )

    xlim, ylim = _scene_limits(radius)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("corridor tracking, first 120 deg")
    ax.grid(True, alpha=0.25)


def draw_corridor_frame(
    runs: dict[str, dict[str, np.ndarray]],
    *,
    frame_index: int = 0,
    radius: float = config.RADIUS,
) -> plt.Figure:
    """Draw one representative top-view frame."""
    frames = _sequential_frames(runs, stride=1)
    active_label, k, completed_labels = frames[int(np.clip(frame_index, 0, len(frames) - 1))]
    fig, ax_scene = plt.subplots(figsize=(7.8, 6.0))
    _draw_corridor_axes(ax_scene, runs, k, active_label=active_label, completed_labels=completed_labels, radius=radius)
    fig.tight_layout()
    return fig


def animate_corridor_comparison(
    runs: dict[str, dict[str, np.ndarray]],
    *,
    radius: float = config.RADIUS,
    stride: int = 3,
    interval: int = 80,
) -> FuncAnimation:
    """Animate the baseline, saturated LQR, and MPC trajectories."""
    frames = _sequential_frames(runs, stride)
    fig, ax_scene = plt.subplots(figsize=(7.8, 6.0))

    def update(i: int) -> list[object]:
        active_label, k, completed_labels = frames[i]
        _draw_corridor_axes(ax_scene, runs, k, active_label=active_label, completed_labels=completed_labels, radius=radius)
        fig.tight_layout()
        return []

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)


def save_corridor_frame(
    path: str | Path,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    frame_index: int = 0,
) -> None:
    """Save one frame for lightweight validation."""
    fig = draw_corridor_frame(runs, frame_index=frame_index)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_corridor_animation(
    path: str | Path,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    radius: float = config.RADIUS,
    stride: int = 3,
    interval: int = 80,
    fps: int = 12,
) -> Path:
    """Save an animated replay as GIF when possible, with HTML fallback."""
    path = Path(path)
    anim = animate_corridor_comparison(runs, radius=radius, stride=stride, interval=interval)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".gif":
            anim.save(path, writer=PillowWriter(fps=fps))
        else:
            html = anim.to_jshtml(fps=fps)
            path.write_text(html, encoding="utf-8")
    finally:
        plt.close("all")
    return path


def save_corridor_replay_html(
    path: str | Path,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    radius: float = config.RADIUS,
    stride: int = 3,
    interval: int = 80,
    fps: int = 12,
) -> Path:
    """Save a standalone HTML replay of the corridor comparison."""
    return save_corridor_animation(path, runs, radius=radius, stride=stride, interval=interval, fps=fps)
