"""Project-local animations for constrained attitude control."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from projects.project_1_attitude_constraints import config

RUN_COLORS = {
    "saturated LQR": "tab:orange",
    "MPC": "tab:blue",
}


def _frame_indices(length: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    indices = np.arange(0, length, stride, dtype=int)
    if indices[-1] != length - 1:
        indices = np.r_[indices, length - 1]
    return indices


def _input_at(U: np.ndarray, k: int) -> float:
    if len(U) == 0:
        return 0.0
    return float(U[min(k, len(U) - 1), 0])


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


def _draw_spacecraft_shape(
    ax: plt.Axes,
    center: tuple[float, float],
    theta: float,
    *,
    color: str,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    panel_fill = "#d9ecff" if alpha >= 0.9 else "none"
    body_fill = "#f7f7f2" if alpha >= 0.9 else "none"
    for y0 in (-0.36, 0.22):
        panel = Polygon(
            _rotated_box(center, theta, -0.34, y0, 0.68, 0.14),
            closed=True,
            facecolor=panel_fill,
            edgecolor=color,
            linewidth=1.4,
            alpha=alpha,
            linestyle=linestyle,
        )
        ax.add_patch(panel)
    body = Polygon(
        _rotated_box(center, theta, -0.18, -0.18, 0.36, 0.36),
        closed=True,
        facecolor=body_fill,
        edgecolor=color,
        linewidth=2.0,
        alpha=alpha,
        linestyle=linestyle,
    )
    ax.add_patch(body)


def _draw_spacecraft(
    ax: plt.Axes,
    center: tuple[float, float],
    theta: float,
    *,
    color: str,
    label: str,
    u: float,
) -> None:
    ax.add_patch(plt.Circle(center, 0.74, edgecolor="0.86", facecolor="none", linewidth=1.0))
    for limit, text_y in [(config.X_MIN[0], -0.82), (config.X_MAX[0], 0.82)]:
        direction = np.array([np.cos(limit), np.sin(limit)])
        end = np.asarray(center) + 0.72 * direction
        ax.plot([center[0], end[0]], [center[1], end[1]], color="tab:red", linestyle="--", linewidth=1.2)
        ax.text(end[0] + 0.04, center[1] + text_y * 0.08, "theta limit", color="tab:red", fontsize=7)

    _draw_spacecraft_shape(ax, center, 0.0, color="0.45", alpha=0.55, linestyle="--")
    _draw_spacecraft_shape(ax, center, theta, color=color)

    heading = _rotated_points(center, np.array([[0.0, 0.0], [0.52, 0.0]]), theta)
    ax.annotate(
        "",
        xy=heading[1],
        xytext=heading[0],
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2.0},
    )

    saturated = np.isclose(u, config.U_MIN[0]) or np.isclose(u, config.U_MAX[0])
    if abs(u) > 1e-9:
        sign = 1.0 if u >= 0.0 else -1.0
        start_angle = theta - sign * 0.55
        end_angle = theta + sign * 0.55
        start = np.asarray(center) + 0.55 * np.array([np.cos(start_angle), np.sin(start_angle)])
        end = np.asarray(center) + 0.55 * np.array([np.cos(end_angle), np.sin(end_angle)])
        torque_color = "tab:red" if saturated else color
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.8,
            color=torque_color,
            connectionstyle=f"arc3,rad={0.35 * sign}",
        )
        ax.add_patch(arrow)

    ax.text(center[0] - 0.82, center[1] + 0.78, label, color=color, fontsize=10, weight="bold")
    ax.text(
        center[0] - 0.82,
        center[1] - 0.88,
        f"theta={theta:+.2f} rad, u={u:+.2f}" + (" clipped" if saturated else ""),
        color="tab:red" if saturated else "0.25",
        fontsize=8.5,
    )


def _draw_attitude_axes(
    fig: plt.Figure,
    time: np.ndarray,
    saturated_lqr: tuple[np.ndarray, np.ndarray],
    mpc: tuple[np.ndarray, np.ndarray],
    k: int,
) -> None:
    X_sat, U_sat = saturated_lqr
    X_mpc, U_mpc = mpc

    fig.clear()
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax_scene = fig.add_subplot(grid[:, 0])
    ax_state = fig.add_subplot(grid[0, 1])
    ax_phase = fig.add_subplot(grid[1, 1])

    _draw_spacecraft(
        ax_scene,
        (-0.05, 0.75),
        float(X_sat[k, 0]),
        color=RUN_COLORS["saturated LQR"],
        label="saturated LQR",
        u=_input_at(U_sat, k),
    )
    _draw_spacecraft(
        ax_scene,
        (-0.05, -0.75),
        float(X_mpc[k, 0]),
        color=RUN_COLORS["MPC"],
        label="MPC",
        u=_input_at(U_mpc, k),
    )
    ax_scene.axvline(0.95, color="0.88", linewidth=1.0)
    ax_scene.text(1.03, 0.10, "dashed ghost = target attitude", rotation=90, color="0.35", fontsize=9, va="center")
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.set_xlim(-1.05, 1.25)
    ax_scene.set_ylim(-1.75, 1.75)
    ax_scene.set_xticks([])
    ax_scene.set_yticks([])
    ax_scene.set_title("2D spacecraft attitude")

    ax_state.plot(time[: k + 1], X_sat[: k + 1, 0], color=RUN_COLORS["saturated LQR"], linewidth=2.0, label="sat. LQR theta")
    ax_state.plot(time[: k + 1], X_mpc[: k + 1, 0], color=RUN_COLORS["MPC"], linewidth=2.0, label="MPC theta")
    ax_state.axhline(config.X_MIN[0], color="tab:red", linestyle="--", linewidth=1.0)
    ax_state.axhline(config.X_MAX[0], color="tab:red", linestyle="--", linewidth=1.0)
    ax_state.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    ax_state.axvline(time[k], color="0.25", linewidth=0.9, alpha=0.7)
    ax_state.set_xlim(time[0], time[-1])
    ax_state.set_ylim(config.X_MIN[0] - 0.2, config.X_MAX[0] + 0.2)
    ax_state.set_ylabel("theta [rad]")
    ax_state.set_title(f"angle history, t = {time[k]:.1f} s")
    ax_state.grid(True, alpha=0.25)
    ax_state.legend(loc="best", fontsize=8)

    ax_phase.add_patch(
        Rectangle(
            (config.X_MIN[0], config.X_MIN[1]),
            config.X_MAX[0] - config.X_MIN[0],
            config.X_MAX[1] - config.X_MIN[1],
            edgecolor="tab:red",
            facecolor="tab:red",
            alpha=0.08,
            linewidth=1.8,
            label="state limits",
        )
    )
    ax_phase.plot(X_sat[: k + 1, 0], X_sat[: k + 1, 1], color=RUN_COLORS["saturated LQR"], linewidth=2.0, label="saturated LQR")
    ax_phase.plot(X_mpc[: k + 1, 0], X_mpc[: k + 1, 1], color=RUN_COLORS["MPC"], linewidth=2.0, label="MPC")
    ax_phase.scatter([X_sat[k, 0]], [X_sat[k, 1]], color=RUN_COLORS["saturated LQR"], s=45)
    ax_phase.scatter([X_mpc[k, 0]], [X_mpc[k, 1]], color=RUN_COLORS["MPC"], s=45)
    ax_phase.axvline(config.X_MIN[0], color="tab:red", linestyle="--", linewidth=1.2)
    ax_phase.axvline(config.X_MAX[0], color="tab:red", linestyle="--", linewidth=1.2)
    ax_phase.axhline(config.X_MIN[1], color="tab:red", linestyle="--", linewidth=1.2)
    ax_phase.axhline(config.X_MAX[1], color="tab:red", linestyle="--", linewidth=1.2)
    ax_phase.set_xlabel("theta [rad]")
    ax_phase.set_ylabel("omega [rad/s]")
    ax_phase.set_title("phase plane")
    ax_phase.grid(True, alpha=0.25)
    ax_phase.legend(loc="best")
    fig.tight_layout()


def draw_attitude_frame(
    time: np.ndarray,
    saturated_lqr: tuple[np.ndarray, np.ndarray],
    mpc: tuple[np.ndarray, np.ndarray],
    *,
    frame_index: int = 0,
) -> plt.Figure:
    """Draw one representative attitude-comparison frame."""
    k = int(np.clip(frame_index, 0, len(time) - 1))
    fig = plt.figure(figsize=(9.4, 5.2))
    _draw_attitude_axes(fig, time, saturated_lqr, mpc, k)
    return fig


def animate_attitude_comparison(
    time: np.ndarray,
    saturated_lqr: tuple[np.ndarray, np.ndarray],
    mpc: tuple[np.ndarray, np.ndarray],
    *,
    stride: int = 2,
    interval: int = 80,
) -> FuncAnimation:
    """Animate saturated LQR and MPC attitude histories."""
    frames = _frame_indices(len(time), stride)

    fig = plt.figure(figsize=(9.4, 5.2))

    def update(i: int) -> list[object]:
        k = int(frames[i])
        _draw_attitude_axes(fig, time, saturated_lqr, mpc, k)
        return []

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)


def save_attitude_frame(
    path: str | Path,
    time: np.ndarray,
    saturated_lqr: tuple[np.ndarray, np.ndarray],
    mpc: tuple[np.ndarray, np.ndarray],
    *,
    frame_index: int = 0,
) -> None:
    """Save one frame for lightweight validation."""
    fig = draw_attitude_frame(time, saturated_lqr, mpc, frame_index=frame_index)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_attitude_replay_html(
    path: str | Path,
    time: np.ndarray,
    saturated_lqr: tuple[np.ndarray, np.ndarray],
    mpc: tuple[np.ndarray, np.ndarray],
    *,
    stride: int = 2,
    interval: int = 80,
    fps: int = 12,
) -> Path:
    """Save a standalone HTML replay of the attitude comparison."""
    path = Path(path)
    anim = animate_attitude_comparison(
        time,
        saturated_lqr,
        mpc,
        stride=stride,
        interval=interval,
    )
    try:
        html = anim.to_jshtml(fps=fps)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    finally:
        plt.close("all")
    return path
