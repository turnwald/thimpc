"""Project-local animations for learning-enhanced prediction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


STATE_LABELS = ("xi", "eta", "psi")


def _frame_indices(length: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    indices = np.arange(0, length, stride, dtype=int)
    if indices[-1] != length - 1:
        indices = np.r_[indices, length - 1]
    return indices


def _draw_prediction_axes(
    ax: plt.Axes,
    measured_next: np.ndarray,
    nominal_next: np.ndarray,
    learned_next: np.ndarray,
    k: int,
    *,
    state_index: int = 1,
) -> None:
    state_index = int(state_index)
    label = STATE_LABELS[state_index]

    measured_value = float(measured_next[k, state_index])
    nominal_value = float(nominal_next[k, state_index])
    learned_value = float(learned_next[k, state_index])
    nominal_error = measured_value - nominal_value
    learned_error = measured_value - learned_value

    all_values = np.r_[measured_next[:, state_index], nominal_next[:, state_index], learned_next[:, state_index]]
    pad = max(0.05, 0.12 * float(np.ptp(all_values)))
    x_min = float(np.min(all_values) - pad)
    x_max = float(np.max(all_values) + pad)

    ax.clear()
    y_positions = {"measured": 2.0, "nominal": 1.0, "learned": 0.0}
    ax.hlines(list(y_positions.values()), x_min, x_max, color="0.9", linewidth=1.0)
    ax.plot([measured_value, nominal_value], [2.0, 1.0], color="tab:orange", linewidth=2.0, alpha=0.8)
    ax.plot([measured_value, learned_value], [2.0, 0.0], color="tab:blue", linewidth=2.0, alpha=0.8)
    ax.scatter([measured_value], [2.0], color="black", s=70, label="true next state", zorder=4)
    ax.scatter([nominal_value], [1.0], color="tab:orange", s=70, label="nominal prediction", zorder=4)
    ax.scatter([learned_value], [0.0], color="tab:blue", s=70, label="learned prediction", zorder=4)
    ax.axvline(measured_value, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks([0.0, 1.0, 2.0], ["learned", "nominal", "true"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel(f"next {label}")
    ax.set_title(f"validation sample {k}: prediction error")
    ax.text(
        0.02,
        0.05,
        f"nominal error = {nominal_error:+.4f}\nlearned error = {learned_error:+.4f}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.25)


def draw_prediction_frame(
    measured_next: np.ndarray,
    nominal_next: np.ndarray,
    learned_next: np.ndarray,
    *,
    frame_index: int = 0,
    state_index: int = 1,
) -> plt.Figure:
    """Draw one validation sample comparing measured and predicted next state."""
    k = int(np.clip(frame_index, 0, len(measured_next) - 1))
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    _draw_prediction_axes(
        ax,
        measured_next,
        nominal_next,
        learned_next,
        k,
        state_index=state_index,
    )
    fig.tight_layout()
    return fig


def animate_prediction_comparison(
    measured_next: np.ndarray,
    nominal_next: np.ndarray,
    learned_next: np.ndarray,
    *,
    state_index: int = 1,
    stride: int = 2,
    interval: int = 120,
) -> FuncAnimation:
    """Animate validation samples for nominal and learned predictions."""
    frames = _frame_indices(len(measured_next), stride)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))

    def update(i: int) -> list[object]:
        _draw_prediction_axes(
            ax,
            measured_next,
            nominal_next,
            learned_next,
            int(frames[i]),
            state_index=state_index,
        )
        fig.tight_layout()
        return []

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)


def save_prediction_frame(
    path: str | Path,
    measured_next: np.ndarray,
    nominal_next: np.ndarray,
    learned_next: np.ndarray,
    *,
    frame_index: int = 0,
    state_index: int = 1,
) -> None:
    """Save one frame for lightweight validation."""
    fig = draw_prediction_frame(
        measured_next,
        nominal_next,
        learned_next,
        frame_index=frame_index,
        state_index=state_index,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_prediction_replay_html(
    path: str | Path,
    measured_next: np.ndarray,
    nominal_next: np.ndarray,
    learned_next: np.ndarray,
    *,
    state_index: int = 1,
    stride: int = 2,
    interval: int = 120,
    fps: int = 10,
) -> Path:
    """Save a standalone HTML replay of the prediction comparison."""
    path = Path(path)
    anim = animate_prediction_comparison(
        measured_next,
        nominal_next,
        learned_next,
        state_index=state_index,
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
