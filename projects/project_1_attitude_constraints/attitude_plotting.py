from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def summarize_attitude_run(name: str, run: dict, theta_max: float, omega_max: float, u_max: float) -> dict:
    x = run["x"]
    u = run["u"]
    theta_violation = np.maximum(np.abs(x[:, 0]) - theta_max, 0.0)
    omega_violation = np.maximum(np.abs(x[:, 1]) - omega_max, 0.0)
    input_violation = np.maximum(np.abs(u) - u_max, 0.0) if len(u) else np.array([0.0])
    return {
        "controller": name,
        "final_abs_theta": float(abs(x[-1, 0])),
        "final_abs_omega": float(abs(x[-1, 1])),
        "max_abs_u": float(np.max(np.abs(u))) if len(u) else 0.0,
        "max_theta_violation": float(np.max(theta_violation)),
        "max_omega_violation": float(np.max(omega_violation)),
        "max_input_violation": float(np.max(input_violation)),
        "solver_failures": int(run.get("solver_failures", 0)),
    }


def plot_attitude_time_histories(time: np.ndarray, runs: dict[str, dict], theta_max: float, omega_max: float):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for label, run in runs.items():
        axes[0].plot(time, run["x"][:, 0], label=label)
        axes[1].plot(time, run["x"][:, 1], label=label)
    axes[0].axhline(theta_max, linestyle="--", linewidth=1)
    axes[0].axhline(-theta_max, linestyle="--", linewidth=1)
    axes[1].axhline(omega_max, linestyle="--", linewidth=1)
    axes[1].axhline(-omega_max, linestyle="--", linewidth=1)
    axes[0].set_ylabel(r"attitude error $\theta$ [rad]")
    axes[1].set_ylabel(r"rate error $\omega$ [rad/s]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    return fig, axes


def plot_attitude_inputs(time_u: np.ndarray, runs: dict[str, dict], u_max: float):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for label, run in runs.items():
        ax.step(time_u, run["u"], where="post", label=label)
    ax.axhline(u_max, linestyle="--", linewidth=1)
    ax.axhline(-u_max, linestyle="--", linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"input $u$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_attitude_phase(runs: dict[str, dict], theta_max: float, omega_max: float):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for label, run in runs.items():
        ax.plot(run["x"][:, 0], run["x"][:, 1], marker="o", markersize=2, label=label)
    ax.axvline(theta_max, linestyle="--", linewidth=1)
    ax.axvline(-theta_max, linestyle="--", linewidth=1)
    ax.axhline(omega_max, linestyle="--", linewidth=1)
    ax.axhline(-omega_max, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax
