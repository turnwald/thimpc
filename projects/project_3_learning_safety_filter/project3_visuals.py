"""Planar-arm visual helpers for Project 3.

All spatial plots replay the end-effector point trajectory as a two-link
planar arm.  The point model is the control/training model; the arm is the
student-facing visual embodiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def inverse_kinematics_2link(p, l1=1.2, l2=1.2, elbow="up"):
    """Map an end-effector point to a feasible two-link planar arm pose."""

    p = np.asarray(p, dtype=float)
    x, y = float(p[0]), float(p[1])
    r2 = x * x + y * y
    cos_q2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    sin_sign = 1.0 if elbow == "up" else -1.0
    q2 = np.arctan2(sin_sign * np.sqrt(max(0.0, 1.0 - cos_q2 * cos_q2)), cos_q2)
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2], dtype=float)


def arm_points_from_p(p, l1=1.2, l2=1.2, elbow="up"):
    """Return shoulder, elbow, end-effector points for the IK replay."""

    q1, q2 = inverse_kinematics_2link(p, l1=l1, l2=l2, elbow=elbow)
    shoulder = np.array([0.0, 0.0])
    elbow_pt = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    ee = elbow_pt + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    return np.vstack([shoulder, elbow_pt, ee])


def _setup_arm_axis(ax, scenario, title=None):
    p_min = scenario["p_min"]
    p_max = scenario["p_max"]
    pad = 0.18
    ax.set_xlim(float(p_min[0] - pad), float(p_max[0] + pad))
    ax.set_ylim(float(p_min[1] - pad), float(p_max[1] + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    # Workspace box.
    box_x = [p_min[0], p_max[0], p_max[0], p_min[0], p_min[0]]
    box_y = [p_min[1], p_min[1], p_max[1], p_max[1], p_min[1]]
    ax.plot(box_x, box_y, linewidth=1.0)
    # Obstacle / keep-out zone and target.
    obs = plt.Circle(scenario["p_obs"], scenario["r_safe"], fill=False, linewidth=1.5)
    ax.add_patch(obs)
    ax.scatter([scenario["p_goal"][0]], [scenario["p_goal"][1]], marker="*", s=90, label="goal")
    ax.scatter([scenario["p_start"][0]], [scenario["p_start"][1]], marker="o", s=35, label="start")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_arm_samples(ax, trajectory, stride=8, label=None):
    trajectory = np.asarray(trajectory, dtype=float)
    ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1.4, label=label)
    sample_ids = list(range(0, len(trajectory), max(1, stride)))
    if sample_ids[-1] != len(trajectory) - 1:
        sample_ids.append(len(trajectory) - 1)
    for idx in sample_ids:
        pts = arm_points_from_p(trajectory[idx])
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2.5, linewidth=1.0, alpha=0.7)


def plot_training_history(history):
    """Plot training/validation loss from a saved history dictionary."""

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE imitation loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_policy_scatter_or_error(dataset, policy, scenario, core):
    """Non-spatial diagnostic: expert vs learned velocity components."""

    v_pred = core.policy_velocity(policy, dataset["p"], scenario)
    v_true = dataset["v_mpc"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    labels = [r"$v_x$", r"$v_y$"]
    for i, ax in enumerate(axes):
        ax.scatter(v_true[:, i], v_pred[:, i], s=14, alpha=0.75)
        lo = min(v_true[:, i].min(), v_pred[:, i].min())
        hi = max(v_true[:, i].max(), v_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], linewidth=1.0)
        ax.set_xlabel("MPC expert")
        ax.set_ylabel("learned policy")
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_arm_rollout_comparison(rollouts, scenario, stride=8):
    """Render each rollout as a planar-arm end-effector replay."""

    n = len(rollouts)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.1), squeeze=False)
    for ax, (name, rollout) in zip(axes[0], rollouts.items()):
        _setup_arm_axis(ax, scenario, title=name)
        _draw_arm_samples(ax, rollout["p"], stride=stride, label="end-effector trace")
        ax.text(
            0.02,
            0.02,
            f"min h = {float(np.min(rollout['h'])):.3f}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round", "alpha": 0.15},
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_cbf_diagnostics(rollouts, scenario, robust=True):
    """Plot non-spatial safety diagnostics h, rho, and intervention."""

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    for name, rollout in rollouts.items():
        t_h = np.arange(len(rollout["h"])) * scenario["Ts"]
        axes[0].plot(t_h, rollout["h"], label=name)
        rho_key = "rho_applied_robust" if robust else "rho_applied"
        if rho_key in rollout:
            t_rho = np.arange(len(rollout[rho_key])) * scenario["Ts"]
            axes[1].plot(t_rho, rollout[rho_key], label=name)
        if "intervention" in rollout:
            t_int = np.arange(len(rollout["intervention"])) * scenario["Ts"]
            axes[2].plot(t_int, rollout["intervention"], label=name)
    axes[0].axhline(0.0, linewidth=1.0)
    axes[1].axhline(0.0, linewidth=1.0)
    axes[0].set_ylabel(r"$h(p_k)$")
    axes[1].set_ylabel(r"robust CBF residual $\rho_{rob}$" if robust else r"CBF residual $\rho$")
    axes[2].set_ylabel(r"$\|v_{safe}-v_{NN}\|$")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_critical_cbf_frame(unsafe_rollout, safe_rollout, scenario):
    """Show the frame with the strongest robust CBF violation before filtering."""

    rho = unsafe_rollout.get("rho_nominal_robust", unsafe_rollout.get("rho_nominal"))
    k = int(np.argmin(rho))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), squeeze=False)
    cases = [("disturbed learned policy", unsafe_rollout), ("disturbed learned + robust CBF", safe_rollout)]
    for ax, (title, rollout) in zip(axes[0], cases):
        _setup_arm_axis(ax, scenario, title=f"{title}\ncritical step k={k}")
        _draw_arm_samples(ax, rollout["p"][: k + 2], stride=max(1, k // 4 + 1), label="trace so far")
        pts = arm_points_from_p(rollout["p"][k])
        ax.plot(pts[:, 0], pts[:, 1], marker="o", linewidth=2.4)
        p = rollout["p"][k]
        v_nom = rollout["v_nom"][k]
        v_app = rollout["v"][k]
        ax.arrow(p[0], p[1], 0.12 * v_nom[0], 0.12 * v_nom[1], head_width=0.025, length_includes_head=True)
        ax.arrow(p[0], p[1], 0.12 * v_app[0], 0.12 * v_app[1], head_width=0.025, length_includes_head=True)
        ax.text(
            0.02,
            0.02,
            f"h={rollout['h'][k]:.3f}\n"
            f"rho_rob={rollout['rho_applied_robust'][k]:.3f}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round", "alpha": 0.15},
        )
    fig.tight_layout()
    return fig, axes


def animate_arm_safe_vs_unsafe(unsafe_rollout, safe_rollout, scenario, output_path, fps=12):
    """Save a side-by-side arm replay comparing unsafe and filtered execution."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = min(len(unsafe_rollout["p"]), len(safe_rollout["p"]))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), squeeze=False)
    for ax, title in zip(axes[0], ["learned + disturbance", "learned + robust CBF"]):
        _setup_arm_axis(ax, scenario, title=title)
    lines = []
    traces = []
    texts = []
    for ax in axes[0]:
        line, = ax.plot([], [], marker="o", linewidth=2.2)
        trace, = ax.plot([], [], linewidth=1.2)
        txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom", ha="left", bbox={"boxstyle":"round", "alpha":0.15})
        lines.append(line); traces.append(trace); texts.append(txt)

    def update(frame):
        for j, rollout in enumerate([unsafe_rollout, safe_rollout]):
            pts = arm_points_from_p(rollout["p"][frame])
            lines[j].set_data(pts[:,0], pts[:,1])
            traces[j].set_data(rollout["p"][:frame+1,0], rollout["p"][:frame+1,1])
            k = min(frame, len(rollout.get("rho_applied_robust", [0])) - 1)
            texts[j].set_text(f"h={rollout['h'][frame]:.3f}\nrho_rob={rollout.get('rho_applied_robust', np.array([0]))[k]:.3f}")
        return lines + traces + texts

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    return output_path


def animate_training_arm_snapshots(snapshots, scenario, output_path, fps=10):
    """Save a 3x4 grid animation/replay of training snapshot rollouts."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = snapshots["epochs"]
    paths = snapshots["p"]
    n_panels = min(12, len(epochs))
    n_frames = paths.shape[1]
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.5), squeeze=False)
    lines = []
    traces = []
    for idx, ax in enumerate(axes.ravel()):
        if idx < n_panels:
            _setup_arm_axis(ax, scenario, title=f"epoch {int(epochs[idx])}")
            line, = ax.plot([], [], marker="o", linewidth=1.6)
            trace, = ax.plot([], [], linewidth=1.0)
            lines.append(line); traces.append(trace)
        else:
            ax.axis("off")
    def update(frame):
        for idx in range(n_panels):
            current = min(frame, n_frames - 1)
            pts = arm_points_from_p(paths[idx, current])
            lines[idx].set_data(pts[:,0], pts[:,1])
            traces[idx].set_data(paths[idx,:current+1,0], paths[idx,:current+1,1])
        return lines + traces
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    return output_path
