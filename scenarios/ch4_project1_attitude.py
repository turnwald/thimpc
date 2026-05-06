"""Chapter 4 Project 1: constrained attitude control with a double integrator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controllers.lqr import discrete_lqr, simulate_lqr
from controllers.saturated_lqr import simulate_saturated_lqr
from mpc.casadi_linear_mpc import solve_linear_mpc
from mpc.metrics import count_violations, status_counts, write_metrics, write_summary
from mpc.plotting import plot_attitude_time, savefig
from mpc.terminal_tools import ellipse_points
from systems.double_integrator import attitude_matrices


def simulate_receding_mpc(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    steps: int,
    horizon: int,
    Q: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    u_bounds: tuple[np.ndarray, np.ndarray],
    x_bounds: tuple[np.ndarray, np.ndarray],
    rate_bound: np.ndarray | None = None,
    soft_theta: bool = False,
    slack_penalty: float = 0.0,
) -> dict[str, np.ndarray | list[str] | int]:
    """Run receding-horizon MPC and expose solver status/slack histories."""
    nx = A.shape[0]
    nu = B.shape[1]
    X = np.zeros((steps + 1, nx))
    U = np.zeros((steps, nu))
    slack = np.zeros(steps)
    statuses: list[str] = []
    failures = 0
    X[0] = np.asarray(x0, dtype=float).reshape(nx)
    u_previous = np.zeros(nu)

    for k in range(steps):
        result = solve_linear_mpc(
            A,
            B,
            X[k],
            horizon,
            Q,
            R,
            P_terminal=P,
            u_bounds=u_bounds,
            x_bounds=x_bounds,
            rate_bound=rate_bound,
            u_previous=u_previous,
            soften_state_indices=[0] if soft_theta else None,
            slack_penalty=slack_penalty,
        )
        statuses.append(result.status)
        if result.success:
            U[k] = result.u0
            if result.slack.size:
                slack[k] = float(result.slack[0, 0])
        else:
            failures += 1
            U[k] = np.zeros(nu)
        X[k + 1] = A @ X[k] + B @ U[k]
        u_previous = U[k]

    return {"X": X, "U": U, "slack": slack, "statuses": statuses, "failures": failures}


def plot_phase(
    path: Path,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    x_bounds: tuple[np.ndarray, np.ndarray],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for label, (X, _) in runs.items():
        ax.plot(X[:, 0], X[:, 1], label=label)
        ax.plot(X[0, 0], X[0, 1], "o", markersize=4)
    ax.axvline(x_bounds[0][0], color="k", linestyle="--", linewidth=0.9)
    ax.axvline(x_bounds[1][0], color="k", linestyle="--", linewidth=0.9)
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


def plot_terminal_ellipses(
    path: Path,
    P: np.ndarray,
    x_bounds: tuple[np.ndarray, np.ndarray],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for level in [0.5, 1.5, 3.0, 6.0]:
        pts = ellipse_points(P, level)
        ax.plot(pts[:, 0], pts[:, 1], label=f"x^T P x = {level:g}")
    ax.axvline(x_bounds[0][0], color="k", linestyle="--", linewidth=0.9)
    ax.axvline(x_bounds[1][0], color="k", linestyle="--", linewidth=0.9)
    ax.axhline(x_bounds[0][1], color="k", linestyle=":", linewidth=0.9)
    ax.axhline(x_bounds[1][1], color="k", linestyle=":", linewidth=0.9)
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
    path: Path,
    t: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    slack: np.ndarray,
    x_bounds: tuple[np.ndarray, np.ndarray],
    u_bounds: tuple[np.ndarray, np.ndarray],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    margin_theta = np.minimum(X[:, 0] - x_bounds[0][0], x_bounds[1][0] - X[:, 0])
    margin_u = np.minimum(U[:, 0] - u_bounds[0][0], u_bounds[1][0] - U[:, 0])
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.8), sharex=True)
    axes[0].plot(t, margin_theta)
    axes[0].axhline(0.0, color="k", linewidth=0.9)
    axes[0].set_ylabel("theta margin")
    axes[1].step(t[:-1], margin_u, where="post")
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


def run_project1(steps: int, output_dir: str | Path, *, show: bool = True, close: bool = False) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    dt = 0.1
    A, B = attitude_matrices(dt=dt)
    Q = np.diag([20.0, 2.0])
    R = np.array([[0.25]])
    K, P = discrete_lqr(A, B, Q, R)
    horizon = 18
    x0 = np.array([1.0, 0.0])
    u_bounds = (np.array([-1.1]), np.array([1.1]))
    x_bounds = (np.array([-1.2, -2.5]), np.array([1.2, 2.5]))
    rate_bound = np.array([0.25])

    X_lqr, U_lqr = simulate_lqr(A, B, K, x0, steps)
    X_sat, U_sat = simulate_saturated_lqr(A, B, K, x0, steps, *u_bounds)
    mpc_run = simulate_receding_mpc(A, B, x0, steps, horizon, Q, R, P, u_bounds, x_bounds)
    rate_run = simulate_receding_mpc(A, B, x0, steps, horizon, Q, R, P, u_bounds, x_bounds, rate_bound=rate_bound)

    infeasible = solve_linear_mpc(
        A,
        B,
        np.array([1.5, 0.0]),
        horizon,
        Q,
        R,
        P_terminal=P,
        u_bounds=u_bounds,
        x_bounds=(np.array([-0.8, -2.5]), np.array([0.8, 2.5])),
    )
    soft = solve_linear_mpc(
        A,
        B,
        np.array([1.5, 0.0]),
        horizon,
        Q,
        R,
        P_terminal=P,
        u_bounds=u_bounds,
        x_bounds=(np.array([-0.8, -2.5]), np.array([0.8, 2.5])),
        soften_state_indices=[0],
        slack_penalty=5_000.0,
    )

    runs = {
        "LQR": (X_lqr, U_lqr),
        "saturated LQR": (X_sat, U_sat),
        "MPC": (mpc_run["X"], mpc_run["U"]),
        "MPC + rate": (rate_run["X"], rate_run["U"]),
    }
    t = np.arange(steps + 1) * dt
    plot_attitude_time(
        figures_dir / "time_plots.png",
        t,
        runs,
        {
            "theta_min": x_bounds[0][0],
            "theta_max": x_bounds[1][0],
            "u_min": u_bounds[0][0],
            "u_max": u_bounds[1][0],
        },
        show=show,
        close=close,
    )
    plot_phase(figures_dir / "phase_plane.png", runs, x_bounds, show=show, close=close)
    plot_terminal_ellipses(figures_dir / "terminal_ellipses.png", P, x_bounds, show=show, close=close)
    plot_constraint_activity(
        figures_dir / "constraint_activity.png",
        t,
        rate_run["X"],
        rate_run["U"],
        rate_run["slack"],
        x_bounds,
        u_bounds,
        show=show,
        close=close,
    )

    metrics = {
        "final_state_norm": {name: float(np.linalg.norm(X[-1])) for name, (X, _) in runs.items()},
        "max_abs_input": {name: float(np.max(np.abs(U))) for name, (_, U) in runs.items()},
        "input_saturation_count": {
            name: count_violations(U[:, 0], u_bounds[0][0], u_bounds[1][0]) for name, (_, U) in runs.items()
        },
        "state_violation_count": {
            name: count_violations(X[:, 0], x_bounds[0][0], x_bounds[1][0]) for name, (X, _) in runs.items()
        },
        "mpc_solver_failures": int(mpc_run["failures"]),
        "rate_mpc_solver_failures": int(rate_run["failures"]),
        "mpc_status_counts": status_counts(mpc_run["statuses"]),
        "rate_mpc_status_counts": status_counts(rate_run["statuses"]),
        "hard_infeasibility_status": infeasible.status,
        "soft_recovery_status": soft.status,
        "soft_initial_slack": float(soft.slack[0, 0]) if soft.success and soft.slack.size else None,
    }
    write_metrics(output_dir / "metrics.json", metrics)
    write_summary(
        output_dir / "summary.txt",
        [
            "Chapter 4 Project 1: constrained attitude control",
            f"steps: {steps}",
            f"MPC solver failures: {metrics['mpc_solver_failures']}",
            f"rate-limited MPC solver failures: {metrics['rate_mpc_solver_failures']}",
            f"hard infeasibility status: {metrics['hard_infeasibility_status']}",
            f"soft recovery status: {metrics['soft_recovery_status']}",
        ],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output-dir", default="outputs/ch4_project1")
    args = parser.parse_args()
    run_project1(args.steps, args.output_dir, show=False, close=True)


if __name__ == "__main__":
    main()
